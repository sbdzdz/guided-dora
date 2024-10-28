import pathlib
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import Cityscapes

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE

from seguidora.methods import Seguidora

import os
import wandb
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="base")
def main(cfg: DictConfig):
    print(f"Configuration: {cfg}")

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.use_wandb:
        wandb_logger = WandbLogger(
            project="nat_aug",
            name=f"dino_petface-nat" if cfg.natural_augmentation else "dino_petface",
            save_dir=cfg.log_dir,
        )
    else:
        wandb_logger = None

    transform = DINOTransform(
        global_crop_scale=(0.14, 1),
        local_crop_scale=(0.05, 0.14)
    )

    cfg.data_dir = pathlib.Path(cfg.data_dir).expanduser().resolve()
    train_dataset = Cityscapes(
        root=cfg.data_dir,
        split='train',
        mode='fine',
        target_type='semantic',
        transform=transform,
    )
    cfg.num_classes = len(train_dataset.classes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_per_device,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )

    if cfg.skip_validation:
        val_dataloader = None
    else:
        val_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
            ]
        )
        val_dataset = Cityscapes(
            root=cfg.data_dir,
            split='val',
            mode='fine',
            target_type='semantic',
            transform=val_transform,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size_per_device,
            shuffle=False,
            num_workers=cfg.num_workers,
            persistent_workers=False,
        )

    model = Dora(backbone=cfg.backbone, batch_size_per_device=cfg.batch_size_per_device, num_classes=cfg.num_classes)

    name = "dino-petface-nat" if cfg.natural_augmentation else "dino-petface"
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=10, dirpath=f'logs/nat_aug/{name}')

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        limit_train_batches=cfg.limit_train_batches,
        fast_dev_run=cfg.fast_dev_run,
        default_root_dir=cfg.log_dir,
        devices=cfg.devices,
        accelerator=cfg.accelerator,
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=False if cfg.seed == -1 else True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.ckpt_path
    )

if __name__ == "__main__":
    main()
