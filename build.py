from typing import Dict, List
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.cuda.amp import GradScaler

from data.dataset import OpenWorldDataset, ConcatDataset, TwoStreamBatchSampler
from utils.dist_utils import is_distributed_set

from model.nach import NACH
from wrapper.NACHWrapper import NACHWrapper


def build_model(cfg: Dict) -> nn.Module:
    # cfg
    model_name = cfg["name"].lower()

    if "nach" in model_name:
        model = NACHWrapper(cfg,
                            NACH(cfg["model"], cfg["loss"],
                                 num_classes=cfg["dataset"]["num_class"],
                                 num_seen=cfg["dataset"]["label_num"]))
    else:
        raise ValueError(f"Unsupported model type {model_name}.")

    return model


def build_dataset(data_dir: str, seed: int, cfg: Dict) -> OpenWorldDataset:
    # # cfg = cfg["dataset"]

    train_label_dataset = OpenWorldDataset(
        dataset_name=cfg["name"].lower(),
        label_num=cfg["label_num"],
        label_ratio=cfg["label_ratio"],
        data_dir=data_dir,
        is_train=True,
        is_label=True,
        unlabeled_idxs=None
    )

    train_unlabel_dataset = OpenWorldDataset(
        dataset_name=cfg["name"].lower(),
        label_num=cfg["label_num"],
        label_ratio=cfg["label_ratio"],
        data_dir=data_dir,
        is_train=True,
        is_label=False,
        unlabeled_idxs=train_label_dataset.unlabeled_idxs
    )

    train_dataset = ConcatDataset((train_label_dataset, train_unlabel_dataset))

    # Following previous works, we test unlabeled dataset in 'trainset'
    valid_dataset = OpenWorldDataset(
        dataset_name=cfg["name"].lower(),
        label_num=cfg["label_num"],
        label_ratio=cfg["label_ratio"],
        data_dir=data_dir,
        is_train=False,
        is_label=False,
        unlabeled_idxs=train_label_dataset.unlabeled_idxs
    )

    return train_dataset, valid_dataset, len(train_label_dataset), len(train_unlabel_dataset)


def build_dataloader(dataset: OpenWorldDataset, batch_size: int,
                     is_train: bool, cfg: Dict,
                     labeled_len: int = 1,
                     unlabeled_len: int = 1,
                     labeled_batch_size: int = 1,
                     ) -> DataLoader:
    if is_train:
        if is_distributed_set():
            sampler = TwoStreamBatchSampler(range(labeled_len), range(labeled_len, labeled_len + unlabeled_len),
                                            batch_size, labeled_batch_size)
            # sampler = DistributedSampler(dataset, shuffle=True, seed=0, drop_last=True)
            shuffle, drop_last = False, True
        else:
            sampler = None
            shuffle, drop_last = True, True
        kwargs = dict(batch_sampler=sampler, num_workers=cfg.get("num_workers", 1))

    else:
        if is_distributed_set():
            sampler = DistributedSampler(dataset, shuffle=False, seed=0, drop_last=False)
            shuffle, drop_last = False, False
        else:
            sampler = None
            shuffle, drop_last = False, False

        kwargs = dict(
            batch_size=batch_size if not is_train else 1,  # per-process (=per-GPU)
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 1),  # per-process
            # collate_fn=ImageNet.fast_collate_imagenet,
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
                                      T_max=num_epoch,
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
