from typing import Dict
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.cuda.amp import GradScaler
import torchvision.transforms as tv

from data.datasets import ImageNet
from utils.dist_utils import is_distributed_set


def build_transform(img_size: int = 224, is_train: bool = False) -> tv.Compose:
    if is_train:
        transform = tv.Compose([
            tv.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                                 interpolation=tv.InterpolationMode.BICUBIC),
            tv.RandomHorizontalFlip(p=0.5),
            tv.ToTensor()  # [0, 255] -> [0, 1] by dividing 255.
        ])
    else:
        pre_size = 256 if (img_size <= 224) else img_size
        transform = tv.Compose([
            tv.Resize(pre_size, interpolation=tv.InterpolationMode.BICUBIC),
            tv.CenterCrop(img_size),
            tv.ToTensor()  # [0, 255] -> [0, 1] by dividing 255.
        ])
    return transform


def build_dataset(data_dir, is_train: bool, transform) -> ImageNet:
    dataset = ImageNet(data_dir, train=is_train, transform=transform)
    return dataset


def build_dataloader(dataset: ImageNet, is_train: bool, cfg: Dict) -> DataLoader:
    if is_train:
        if is_distributed_set():
            sampler = DistributedSampler(dataset, shuffle=True, seed=0, drop_last=True)
            shuffle, drop_last = False, False
        else:
            sampler = None
            shuffle, drop_last = True, True
    else:
        if is_distributed_set():
            sampler = DistributedSampler(dataset, shuffle=False, seed=0, drop_last=False)
            shuffle, drop_last = False, False
        else:
            sampler = None
            shuffle, drop_last = False, False

    # When using DistributedSampler, don't forget to call dataloader.sampler.set_epoch(epoch)

    kwargs = dict(
        batch_size=cfg["batch_size"],  # per-process (=per-GPU)
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.get("num_workers", 1),  # per-process
        collate_fn=ImageNet.fast_collate_imagenet,
        pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=2,
    )
    dataloader = DataLoader(dataset, **kwargs)
    return dataloader


def build_optimizer(model: nn.Module, cfg: Dict) -> SGD:
    params_default = []
    params_no_decay = []
    for p in model.parameters():
        if p.ndim <= 1:  # BN weight/bias, Conv bias, etc.
            params_no_decay.append(p)
        else:
            params_default.append(p)
    params_opt = [
        {"params": params_default},
        {"params": params_no_decay, "weight_decay": 0.0}
    ]
    optimizer_type = cfg["name"].lower()

    if optimizer_type == "sgd":
        optimizer = SGD(
            params_opt,
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg.get("weight_decay", 0.0),
            nesterov=False
        )
    elif optimizer_type == "adam":
        optimizer = Adam(params_opt,
                         lr=cfg["lr"])
    elif optimizer_type == "adamw":
        optimizer = AdamW(params_opt,
                          lr=cfg["lr"],
                          betas=cfg.get("betas", (0.9, 0.999)),
                          weight_decay=cfg.get("weight_decay", 0.0))
    else:
        raise ValueError(f"Unsupported optimizer type {optimizer_type}.")

    return optimizer


def build_scheduler(optimizer: SGD, cfg: Dict, iter_per_epoch: int, num_epoch: int = 1, num_accum: int = 1):
    scheduler_type = cfg["name"].lower()
    iter_per_epoch = iter_per_epoch // num_accum  # actual update count
    if scheduler_type == "constant":
        scheduler = ConstantLR(optimizer,
                               factor=cfg.get("factor", 1.0),
                               total_iters=0)
    elif scheduler_type == "cos":
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=num_epoch * iter_per_epoch,
                                      eta_min=cfg.get("min_lr", 0.0),
                                      last_epoch=-1)
    elif scheduler_type == "custom":
        warmup_cfg = cfg["warmup"]
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_cfg["start_factor"],
            end_factor=1.0,
            total_iters=warmup_cfg["epochs"] * iter_per_epoch,
        )
        decay_cfg = cfg["decay"]
        decay = CosineAnnealingLR(
            optimizer,
            T_max=decay_cfg["epochs"] * iter_per_epoch,
            eta_min=decay_cfg["min_lr"],
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_cfg["epochs"] * iter_per_epoch]
        )
    else:
        raise ValueError(f"Unsupported optimizer type {scheduler_type}.")

    return scheduler


def build_scaler(is_fp16: bool = False) -> GradScaler:
    scaler = GradScaler(init_scale=2048, growth_interval=1000, enabled=is_fp16)
    return scaler