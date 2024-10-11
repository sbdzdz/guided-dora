# train_dino.py

import os
import time
import datetime
import math
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead


@hydra.main(config_path=".", config_name="config")
def train_dino(cfg: DictConfig):
    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    utils.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.seed)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        cfg.global_crops_scale,
        cfg.local_crops_scale,
        cfg.local_crops_number,
    )
    dataset = datasets.ImageFolder(cfg.data_path, transform=transform)
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
    elif cfg.arch in torchvision_models.__dict__.keys():
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
        teacher, DINOHead(embed_dim, cfg.out_dim, cfg.use_bn_in_head)
    )

    student, teacher = student.cuda(), teacher.cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[cfg.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[cfg.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {cfg.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        cfg.out_dim,
        cfg.local_crops_number + 2,
        cfg.warmup_teacher_temp,
        cfg.teacher_temp,
        cfg.warmup_teacher_temp_epochs,
        cfg.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif cfg.optimizer == "lars":
        optimizer = utils.LARS(params_groups)

    fp16_scaler = None
    if cfg.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        cfg.lr * (cfg.batch_size_per_gpu * utils.get_world_size()) / 256.0,
        cfg.min_lr,
        cfg.epochs,
        len(data_loader),
        warmup_epochs=cfg.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        cfg.weight_decay, cfg.weight_decay_end, cfg.epochs, len(data_loader)
    )
    momentum_schedule = utils.cosine_scheduler(
        cfg.momentum_teacher, 1, cfg.epochs, len(data_loader)
    )
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
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
    for epoch in range(start_epoch, cfg.epochs):
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
    print(f"Training time: {str(datetime.timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    train_dino()
