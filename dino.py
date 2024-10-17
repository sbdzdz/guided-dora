# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import datetime
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import hydra
from omegaconf import DictConfig
import utils
import vision_transformer as vits
from vision_transformer import DINOHead
import logging

from tqdm import tqdm, trange
import wandb


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cfg.output_dir)
    logger.info(f"Config: {cfg}")

    # Initialize wandb if necessary
    if cfg.use_wandb and utils.is_main_process():
        wandb.init(entity=cfg.entity, project=cfg.project, name=cfg.name)

    train_dino(cfg, logger)


def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    output_dir = Path(output_dir).resolve() / datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    file_handler = logging.FileHandler(output_dir / "dino_training.log")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)


def train_dino(cfg, logger):
    utils.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        cfg.global_crops_scale,
        cfg.local_crops_scale,
        cfg.local_crops_number,
    )
    data_path = Path(cfg.data_path).resolve()
    logger.debug(data_path)
    dataset = datasets.ImageNet(root=str(data_path), split="train", transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    cfg.arch = cfg.arch.replace("deit", "vit")
    if cfg.arch in vits.__dict__.keys():
        student = vits.__dict__[cfg.arch](
            patch_size=cfg.patch_size,
            drop_path_rate=cfg.drop_path_rate,
        )
        teacher = vits.__dict__[cfg.arch](patch_size=cfg.patch_size)
        embed_dim = student.embed_dim
    elif cfg.arch in torchvision_archs:
        student = torchvision_models.__dict__[cfg.arch]()
        teacher = torchvision_models.__dict__[cfg.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {cfg.arch}")

    student = utils.MultiCropWrapper(
        student,
        DINOHead(
            embed_dim,
            cfg.out_dim,
            use_bn=cfg.use_bn_in_head,
            norm_last_layer=cfg.norm_last_layer,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, cfg.out_dim, use_bn=cfg.use_bn_in_head),
    )
    student, teacher = student.cuda(), teacher.cuda()

    if torch.cuda.is_available():
        student = torch.compile(student)
        teacher = torch.compile(teacher)

    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[cfg.local_rank]
        )
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[cfg.local_rank])
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {cfg.arch} network.")

    dino_loss = DINOLoss(
        cfg.out_dim,
        cfg.local_crops_number + 2,
        cfg.warmup_teacher_temp,
        cfg.teacher_temp,
        cfg.warmup_teacher_temp_epochs,
        cfg.epochs,
    ).cuda()

    params_groups = utils.get_params_groups(student)
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif cfg.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    fp16_scaler = torch.amp.GradScaler() if cfg.use_fp16 else None

    lr_schedule = utils.cosine_scheduler(
        cfg.lr * (cfg.batch_size_per_gpu * utils.get_world_size()) / 256.0,
        cfg.min_lr,
        cfg.epochs,
        len(data_loader),
        warmup_epochs=cfg.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        cfg.weight_decay,
        cfg.weight_decay_end,
        cfg.epochs,
        len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        cfg.momentum_teacher, 1, cfg.epochs, len(data_loader)
    )
    print("Loss, optimizer, and schedulers ready.")

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(cfg.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in trange(start_epoch, cfg.epochs, ncols=100, desc="Epochs"):
        data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            cfg,
        )

        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "cfg": cfg,
            "dino_loss": dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(cfg.output_dir, "checkpoint.pth"))
        if cfg.saveckp_freq and epoch % cfg.saveckp_freq == 0:
            utils.save_on_master(
                save_dict, os.path.join(cfg.output_dir, f"checkpoint{epoch:04}.pth")
            )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if utils.is_main_process():
            with (Path(cfg.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    cfg,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    pbar = tqdm(data_loader, ncols=100, desc="Training")

    for it, (images, _) in enumerate(pbar):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]
        with torch.amp.autocast("cuda", enabled=fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if utils.is_main_process():
            if cfg.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/wd": optimizer.param_groups[0]["weight_decay"],
                    }
                )

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if cfg.clip_grad:
                param_norms = utils.clip_gradients(student, cfg.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, cfg.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if cfg.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, cfg.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, cfg.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(
                student.module.parameters(), teacher_without_ddp.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ]
        )
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ]
        )
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == "__main__":
    main()
