import pathlib
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.msn_transform import MSNTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE

from methods import SimCLR, VICReg, DINO, PMSN
from transforms import DINONaturalTransform, SimCLRNaturalTransform, MSNNaturalTransform
from petface import PetFaceDataset

import os
import wandb
import hydra
from omegaconf import DictConfig

os.environ["WANDB__SERVICE_WAIT"] = "300"
TRAIN_SPLIT = ['cat', 'chimp', 'chinchilla', 'degus', 'dog', 'ferret', 'guineapig', 'hamster__', 'hedgehog__', 'javasparrow__', 'parakeet', 'pig', 'rabbit']
VAL_SPLIT = ['cat', 'chimp', 'chinchilla', 'degus', 'dog', 'ferret', 'guineapig', 'hamster', 'hedgehog', 'javasparrow', 'parakeet', 'pig', 'rabbit']
TEST_SPLIT = ['hamster', 'hedgehog', 'javasparrow']

@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig):
    print(f"Configuration: {cfg}")

    #### Set seed for reproducibility
    if cfg.seed != -1:
        pl.seed_everything(cfg.seed, workers=True)

    #### Set up Weights & Biases logger
    if cfg.use_wandb:
        wandb_logger = WandbLogger(
            project="nat_aug",
            name=f"{cfg.method}_petface-nat" if cfg.natural_augmentation else f"{cfg.method}_petface",
            save_dir=cfg.log_dir,
        )
    else:
        wandb_logger = None

    #### Select the typology of augmentation to use in the experiment
    if cfg.method in ["simclr", "vicreg"]:
        if cfg.natural_augmentation:
            transform = SimCLRNaturalTransform(
                input_size=cfg.input_size,
                cj_prob=cfg.cj_prob,
                cj_strength=cfg.cj_strength,
                cj_bright=cfg.cj_bright,
                cj_contrast=cfg.cj_contrast,
                cj_sat=cfg.cj_sat,
                cj_hue=cfg.cj_hue,
                min_scale=cfg.min_scale,
                random_gray_scale=cfg.random_gray_scale,
                gaussian_blur=cfg.gaussian_blur,
            )
        else:
            transform = SimCLRTransform(
                input_size=cfg.input_size,
                cj_prob=cfg.cj_prob,
                cj_strength=cfg.cj_strength,
                cj_bright=cfg.cj_bright,
                cj_contrast=cfg.cj_contrast,
                cj_sat=cfg.cj_sat,
                cj_hue=cfg.cj_hue,
                min_scale=cfg.min_scale,
                random_gray_scale=cfg.random_gray_scale,
                gaussian_blur=cfg.gaussian_blur,
            )
    elif cfg.method == "dino":
        if cfg.natural_augmentation:
            transform = DINONaturalTransform(
                global_crop_scale=(0.14, 1),
                local_crop_scale=(0.05, 0.14)
            )
        else:
            transform = DINOTransform(
                global_crop_scale=(0.14, 1),
                local_crop_scale=(0.05, 0.14)
            )
    elif cfg.method == "pmsn":
        if cfg.natural_augmentation:
            transform = MSNNaturalTransform()
        else:
            transform = MSNTransform()
    else:
        raise ValueError(f"No augmentations implemented for {cfg.method}")

    #### Create Dataset and DataLoaders
    cfg.data_dir = pathlib.Path(cfg.data_dir).expanduser().resolve()
    train_dataset = PetFaceDataset(
        root=cfg.data_dir,
        split="train",
        transform=transform,
        natural_augmentation=cfg.natural_augmentation,
        class_names=TRAIN_SPLIT,
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
        val_dataset = PetFaceDataset(
            root=cfg.data_dir, split="val", transform=val_transform, class_names=VAL_SPLIT
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size_per_device,
            shuffle=False,
            num_workers=cfg.num_workers,
            persistent_workers=False,
        )

    if cfg.method == "simclr":
        model = SimCLR(
            backbone=cfg.backbone,
            batch_size_per_device=cfg.batch_size_per_device,
            num_classes=cfg.num_classes,
        )
    elif cfg.method == "vicreg":
        model = VICReg(
            backbone=cfg.backbone,
            batch_size_per_device=cfg.batch_size_per_device,
            num_classes=cfg.num_classes,
        )
    elif cfg.method == "dino":
        model = DINO(
            backbone=cfg.backbone,
            batch_size_per_device=cfg.batch_size_per_device,
            num_classes=cfg.num_classes
        )
    elif cfg.method == "pmsn":
        model = PMSN(
            batch_size_per_device=cfg.batch_size_per_device,
            num_classes=cfg.num_classes
        )

    name=f"{cfg.method}-petface-nat" if cfg.natural_augmentation else f"{cfg.method}-petface"
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
