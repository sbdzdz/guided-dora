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

from parser import parse_arguments
from methods import SimCLR, VICReg, DINO, PMSN
from transforms import DINONaturalTransform, SimCLRNaturalTransform, MSNNaturalTransform
from petface import PetFaceDataset

import os
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"
TRAIN_SPLIT = ['cat', 'chimp', 'chinchilla', 'degus', 'dog', 'ferret', 'guineapig', 'hamster__', 'hedgehog__', 'javasparrow__', 'parakeet', 'pig', 'rabbit']
VAL_SPLIT = ['cat', 'chimp', 'chinchilla', 'degus', 'dog', 'ferret', 'guineapig', 'hamster', 'hedgehog', 'javasparrow', 'parakeet', 'pig', 'rabbit']
TEST_SPLIT = ['hamster', 'hedgehog', 'javasparrow']


#### Parse command line arguments
args = parse_arguments()
print(f"Command line arguments {args}")

#### Set seed for reproducibility
if args.seed != -1:
    pl.seed_everything(args.seed, workers=True)

#### Set up Weights & Biases logger
if args.use_wandb:
    wandb_logger = WandbLogger(
        project="nat_aug",
        name=f"{args.method}_petface-nat" if args.natural_augmentation else f"{args.method}_petface",
        save_dir=args.log_dir,
    )
else:
    wandb_logger = None

#### Select the typology of augmentation to use in the experiment
if args.method in ["simclr", "vicreg"]:
    if args.natural_augmentation:
        # Applies the SimCLR view on each one of the "natural" augmentations
        # Note: This alters the default SimCLR augmentation because uses different
        # images.
        transform = SimCLRNaturalTransform(
            input_size=args.input_size,
            cj_prob=args.cj_prob,
            cj_strength=args.cj_strength,
            cj_bright=args.cj_bright,
            cj_contrast=args.cj_contrast,
            cj_sat=args.cj_sat,
            cj_hue=args.cj_hue,
            min_scale=args.min_scale,
            random_gray_scale=args.random_gray_scale,
            gaussian_blur=args.gaussian_blur,
        )
    else:
        # Applies the default SimCLR transforms and generates two views
        transform = SimCLRTransform(
            input_size=args.input_size,
            cj_prob=args.cj_prob,
            cj_strength=args.cj_strength,
            cj_bright=args.cj_bright,
            cj_contrast=args.cj_contrast,
            cj_sat=args.cj_sat,
            cj_hue=args.cj_hue,
            min_scale=args.min_scale,
            random_gray_scale=args.random_gray_scale,
            gaussian_blur=args.gaussian_blur,
        )
elif args.method == "dino":
    if args.natural_augmentation:
        transform = DINONaturalTransform(
            global_crop_scale=(0.14, 1),
            local_crop_scale=(0.05, 0.14)
        )
    else:
        transform = DINOTransform(
            global_crop_scale=(0.14, 1), 
            local_crop_scale=(0.05, 0.14)
        )
if args.method == "pmsn":
    if args.natural_augmentation:
        transform = MSNNaturalTransform()
    else:
        transform = MSNTransform()
else:
    raise ValueError(f"No augmentations implemented for {args.method}")

#### Create Dataset and DataLoaders
# DATA_PATH = pathlib.Path("~/projects/ocl/data/PetFace/").expanduser().resolve()
args.data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
train_dataset = PetFaceDataset(
    root=args.data_dir,
    split="train",
    transform=transform,
    natural_augmentation=args.natural_augmentation,
    class_names=TRAIN_SPLIT,
)
args.num_classes = len(train_dataset.classes)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size_per_device,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
)

if args.skip_validation:
    val_dataloader = None
else:   
    # Setup validation data.
    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    val_dataset = PetFaceDataset(
        root=args.data_dir, split="val", transform=val_transform, class_names=VAL_SPLIT
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False,
    )

if args.method == "simclr": 
    model = SimCLR(
        backbone=args.backbone,
        batch_size_per_device=args.batch_size_per_device,
        num_classes=args.num_classes,
    )
elif args.method == "vicreg":
    model = VICReg(
        backbone=args.backbone,
        batch_size_per_device=args.batch_size_per_device,
        num_classes=args.num_classes,
    )
elif args.method == "dino":
    model = DINO(
        backbone=args.backbone,
        batch_size_per_device=args.batch_size_per_device,
        num_classes=args.num_classes
    )
elif args.method == "pmsn":
    model = PMSN(
        batch_size_per_device=args.batch_size_per_device,
        num_classes=args.num_classes
    )

# Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
# calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
name=f"{args.method}-petface-nat" if args.natural_augmentation else f"{args.method}-petface"
checkpoint_callback = ModelCheckpoint(every_n_train_steps=10, dirpath=f'logs/nat_aug/{name}')

trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    limit_train_batches=args.limit_train_batches,
    fast_dev_run=args.fast_dev_run,
    # profiler="simple",
    default_root_dir=args.log_dir,
    devices=args.devices, 
    accelerator=args.accelerator,
    strategy="ddp_find_unused_parameters_true",
    sync_batchnorm=True,
    # use_distributed_sampler=True,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    deterministic=False if args.seed == -1 else True,
    # log_every_n_steps=5,
    # enable_progress_bar=False
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    ckpt_path=args.ckpt_path
)
