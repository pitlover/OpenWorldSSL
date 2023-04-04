from typing import Dict
from collections import OrderedDict
import os
import time
import pprint

import wandb
import numpy as np
import torch
import torch.cuda.amp as amp
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.config_utils import prepare_config
from utils.wandb_utils import set_wandb
from utils.random_utils import set_seed
from utils.dist_utils import set_dist, is_distributed_set, is_master, barrier, get_world_size
from utils.dist_utils import all_reduce_dict
from utils.print_utils import time_log
from utils.param_utils import count_params, compute_param_norm

from itertools import cycle
from build import (build_dataset, build_dataloader, build_model,
                   build_optimizer, build_scheduler, build_scaler)
from wrapper.NACHWrapper import NACHWrapper
from model.functions import AverageMeter
from model.others.metric import OpenSSLMetric


def train_epoch(
        model: NACHWrapper,
        optimizer,
        scheduler,
        scaler,
        label_dataloader,
        unlabel_dataloader,
        cfg: Dict,
        device: torch.device,
        current_iter: int
) -> int:
    print_interval = cfg["print_interval_iters"]
    fp16 = cfg.get("fp16", False)
    num_accum = cfg.get("num_accum", 1)
    clip_grad = cfg.get("clip_grad", 10.0)

    model.train()
    torch.set_grad_enabled(True)  # same as 'with torch.enable_grad():'
    grad_norm = torch.tensor(0.0, dtype=torch.float32, device=device)  # placeholder

    forward_time = 0.0
    backward_time = 0.0
    step_time = 0.0

    unlabel_dataloader_iter = cycle(unlabel_dataloader)
    data_start_time = time.time()

    for it, label_data in enumerate(label_dataloader):
        s = time_log()
        s += f"Current iter: {current_iter} (epoch done: {it / len(label_dataloader) * 100:.2f} %)\n"

        # -------------------------------- data -------------------------------- #
        (img1, img2, aug_weak, aug_strong), label = label_data
        (uimg1, uimg2, uaug_weak, uaug_strong), ulabel = next(unlabel_dataloader_iter)

        img1 = torch.cat([img1, uimg1], 0)
        img2 = torch.cat([img2, uimg2], 0)
        aug_weak = torch.cat([aug_weak, uaug_weak], 0)
        aug_strong = torch.cat([aug_strong, uaug_strong], 0)

        img1, img2, aug_weak, aug_strong, label = img1.to(device, non_blocking=True), \
                                                  img2.to(device, non_blocking=True), \
                                                  aug_weak.to(device, non_blocking=True), \
                                                  aug_strong.to(device, non_blocking=True), \
                                                  label.to(device, non_blocking=True)
        data_time = time.time() - data_start_time

        # -------------------------------- loss -------------------------------- #
        if it % num_accum == 0:
            optimizer.zero_grad(set_to_none=True)

        if it % num_accum == (num_accum - 1):  # update step
            forward_start_time = time.time()
            with amp.autocast(enabled=fp16):
                _, output = model(img1=img1, label=label, img2=img2, aug_weak=aug_weak,
                                  aug_strong=aug_strong, iter=it, max_iter=len(label_dataloader))  # {"loss", "acc1"}
            forward_time = time.time() - forward_start_time

            backward_start_time = time.time()
            loss = output["loss"]
            loss = loss / num_accum
            scaler.scale(loss).backward()
            backward_time = time.time() - backward_start_time

            step_start_time = time.time()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()
            step_time = time.time() - step_start_time

        elif isinstance(model, DistributedDataParallel):  # non-update step and DDP
            with model.no_sync():
                with amp.autocast(enabled=fp16):
                    _, output = model(img1=img1, label=label, img2=img2, aug_weak=aug_weak,
                                      aug_strong=aug_strong,
                                      iter=it, max_iter=len(label_dataloader))  # {"loss", "acc1"}

                loss = output["loss"]
                loss = loss / num_accum
                scaler.scale(loss).backward()

        else:  # non-update step and not DDP
            with amp.autocast(enabled=fp16):
                _, output = model(img1=img1, label=label, img2=img2, aug_weak=aug_weak,
                                  aug_strong=aug_strong, iter=it, max_iter=len(label_dataloader))  # {"loss", "acc1"}

            loss = output["loss"]
            loss = loss / num_accum
            scaler.scale(loss).backward()

        # -------------------------------- print -------------------------------- #

        if (it > 0) and (it % print_interval == 0):
            output = all_reduce_dict(output, op="mean")
            param_norm = compute_param_norm(model.parameters())
            lr = scheduler.get_last_lr()[0]

            for k, v in output.items():
                s += f"... {k}: {v.item() if isinstance(v, torch.Tensor) else v:.6f}\n"
            s += f"... LR: {lr:.6f}\n"
            s += f"... grad/param norm: {grad_norm.item():.3f} / {param_norm.item():.3f}\n"
            s += f"... batch_size x num_accum x gpus = " \
                 f"{int(label.shape[0])} x {num_accum} x {get_world_size()}\n"
            s += f"... data/fwd/bwd/step time: " \
                 f"{data_time:.3f} / {forward_time:.3f} / {backward_time:.3f} / {step_time:.3f}"

            if is_master():
                print(s)
                log_dict = {
                    "grad_norm": grad_norm.item(),
                    "param_norm": param_norm.item(),
                    "lr": lr,
                    "iterations": current_iter,
                }
                for k, v in output.items():
                    log_dict[k] = v.item() if isinstance(v, torch.Tensor) else v
                wandb.log(log_dict)

        current_iter += 1
        data_start_time = time.time()

    return current_iter


def valid_epoch(
        model: NACHWrapper,
        dataloader,
        cfg: Dict,
        num_seen,
        device: torch.device,
        current_iter: int,
) -> Dict:
    fp16 = cfg.get("fp16", False)

    model.eval()
    torch.set_grad_enabled(False)  # same as 'with torch.no_grad():'

    targets, preds = np.array([]), np.array([])
    output = {}
    for it, data in enumerate(dataloader):
        # -------------------------------- data -------------------------------- #
        (img, _, _, _), label = data
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # -------------------------------- loss -------------------------------- #
        with amp.autocast(enabled=fp16):
            prob = model(img1=img, label=label, iter=it)
            _, pred = prob.max(1)

        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    targets, preds = targets.astype(int), preds.astype(int)

    seen_acc, unseen_acc, all_acc, unseen_nmi = OpenSSLMetric(targets, preds, labeled_num=num_seen)

    output["all_acc"] = all_acc
    output["unseen_acc"] = unseen_acc
    output["seen_acc"] = seen_acc
    output["unseen_nmi"] = unseen_nmi

    if is_master():
        wandb.log({
            "val_all_acc": output["all_acc"],
            "val_unseen_acc": output["unseen_acc"],
            "val_seen_acc": output["seen_acc"],
            "val_unseen_nmi": output["unseen_nmi"],
            "iterations": current_iter,
        })

    return output


def run(cfg: Dict, debug: bool = False, eval: bool = False) -> None:
    # ======================================================================================== #
    # Initialize
    # ======================================================================================== #
    device, local_rank = set_dist(device_type="cuda")

    if is_master():
        pprint.pprint(cfg)  # print config to check if all arguments are correctly given.

    save_dir = set_wandb(cfg, force_mode="disabled" if debug else None)
    set_seed(seed=cfg["seed"] + local_rank)

    # ======================================================================================== #
    # Data
    # ======================================================================================== #
    data_dir = cfg["data_dir"]

    train_label_dataset = build_dataset(data_dir, is_train=True, is_label=True, seed=cfg["seed"],
                                        cfg=cfg["dataset"])
    train_unlabel_dataset = build_dataset(data_dir, is_train=True, is_label=False, seed=cfg["seed"],
                                          cfg=cfg["dataset"], unlabeled_idxs=train_label_dataset.unlabeled_idxs)

    labeled_len, unlabeled_len = len(train_label_dataset), len(train_unlabel_dataset)
    labeled_batch_size = int(cfg["dataloader"]["train"]["batch_size"] * (labeled_len / (labeled_len + unlabeled_len)))
    train_label_dataloader = build_dataloader(train_label_dataset, batch_size=labeled_batch_size, is_train=True,
                                              cfg=cfg["dataloader"]["train"])
    train_unlabel_dataloader = build_dataloader(train_unlabel_dataset,
                                                batch_size=cfg["dataloader"]["train"][
                                                               "batch_size"] - labeled_batch_size,
                                                is_train=True, cfg=cfg["dataloader"]["train"])

    # Following previous works, we test unlabeled dataset in 'trainset'
    valid_dataset = build_dataset(data_dir, is_train=False, is_label=False, seed=cfg["seed"],
                                  cfg=cfg["dataset"],
                                  unlabeled_idxs=train_label_dataset.unlabeled_idxs)
    valid_dataloader = build_dataloader(valid_dataset, batch_size=cfg["dataloader"]["valid"]["batch_size"],
                                        is_train=False, cfg=cfg["dataloader"]["valid"])

    # ======================================================================================== #
    # Model
    # ======================================================================================== #
    model = build_model(cfg)
    model = model.to(device)

    if is_distributed_set():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=device)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # optional for ImageNet
        model_m = model.module  # actual model without wrapping
    else:
        model_m = model

    if is_master():
        print(model)
        p1, p2 = count_params(model_m.parameters())
        print(f"Model parameters: {p1} tensors, {p2} elements.")

        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

    if eval:
        if cfg["resume"]["checkpoint"] != None:
            ckpt = torch.load(cfg["resume"]["checkpoint"], map_location=device)
        else:
            raise ValueError(f"Eval mode requires checkpoint.")
    elif cfg["resume"]["checkpoint"] is not None:
        # checkpoint: {"model", "optimizer", "scheduler", "stats"}
        ckpt = torch.load(cfg["resume"]["checkpoint"], map_location=device)
    else:
        ckpt = None

    if ckpt is not None:
        model_m.load_state_dict(ckpt["model"], strict=cfg["resume"].get("strict", True))

    # ======================================================================================== #
    # Optimizer & Scheduler
    # ======================================================================================== #
    optimizer = build_optimizer(model_m, cfg=cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg=cfg["scheduler"],
                                iter_per_epoch=len(train_label_dataset) + len(train_unlabel_dataset),
                                num_epoch=cfg["trainer"]["max_epochs"],
                                num_accum=cfg["trainer"].get("num_accum", 1))
    scaler = build_scaler(is_fp16=cfg["trainer"].get("fp16", False))

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

    # ======================================================================================== #
    # Trainer
    # ======================================================================================== #

    # -------- config -------- #
    train_cfg = cfg["trainer"]
    max_epochs = train_cfg["max_epochs"]
    valid_interval = train_cfg["valid_interval_epochs"]
    num_seen = cfg["dataset"]["num_class"] * cfg["dataset"]["label_ratio"]

    # -------- status -------- #
    current_epoch = 0
    current_iter = 0
    current_best_unseen = 0.0
    best_epoch, best_iter = 0, 0

    if ckpt is not None:  # Eval
        current_epoch = ckpt["stats"]["epoch"]
        current_iter = ckpt["stats"]["iter"]
        current_best_unseen = ckpt["stats"]["best_unseen"]
        max_epochs = ckpt["stats"]["max_epochs"]

        # -------- check -------- #
        valid_result = valid_epoch(model, valid_dataloader, train_cfg, num_seen, device, current_iter)
        s = time_log()
        s += f"Resume valid epoch {current_epoch} / {max_epochs}\n"
        s += f"... seen_acc: {valid_result['seen_acc']:.4f}\n"
        s += f"... unseen_acc: {valid_result['unseen_acc']:.4f}\n"
        s += f"... all_acc: {valid_result['all_acc']:.4f}\n"
        if is_master():
            print(s)
        if eval:
            print("Best Eval Finish...")
            return
            # -------- main loop -------- #
    while current_epoch < max_epochs:
        if is_master():
            s = time_log()
            s += f"Start train epoch {current_epoch} / {max_epochs} (iter: {current_iter})"
            print(s)

        if is_distributed_set():
            # reset random seed of sampler, sampler should be DistributedSampler.
            train_label_dataloader.sampler.set_epoch(current_epoch)  # noqa
            train_unlabel_dataloader.sampler.set_epoch(current_epoch)  # noqa

        # -------- train body -------- #
        epoch_start_time = time.time()  # second
        current_iter = train_epoch(model, optimizer, scheduler, scaler, train_label_dataloader,
                                   train_unlabel_dataloader,
                                   train_cfg, device, current_iter)
        epoch_time = time.time() - epoch_start_time
        if is_master():
            s = time_log()
            s += f"End train epoch {current_epoch} / {max_epochs}, time: {epoch_time:.3f} s\n"
            # save checkpoint
            ckpt = OrderedDict()
            ckpt["model"] = model_m.state_dict()
            ckpt["optimizer"] = optimizer.state_dict()
            ckpt["scheduler"] = scheduler.state_dict()
            ckpt["scaler"] = scaler.state_dict()
            ckpt["stats"] = OrderedDict(epoch=current_epoch, iter=current_iter,
                                        best_unseen=current_best_unseen)
            torch.save(ckpt, os.path.join(save_dir, "latest.ckpt"))
            s += f"... save checkpoint to {os.path.join(save_dir, 'latest.ckpt')}"
            print(s)

        barrier()
        # -------- valid body -------- #
        if current_epoch % valid_interval == 0 or current_epoch == max_epochs - 1:

            s = time_log()
            s += f"| ***** Start valid epoch {current_epoch} / {max_epochs} (iter: {current_iter})"
            if is_master():
                print(s)

            valid_start_time = time.time()  # second
            valid_result = valid_epoch(model, valid_dataloader, train_cfg, num_seen, device, current_iter)
            valid_time = time.time() - valid_start_time

            s = time_log()
            s += f"| ***** End valid epoch {current_epoch} / {max_epochs}, time: {valid_time:.3f}s\n"
            s += f"| ... seen_acc: {valid_result['seen_acc']:.4f}\n"
            s += f"| ... unseen_acc: {valid_result['unseen_acc']:.4f}\n"
            s += f"| ... all_acc: {valid_result['all_acc']:.4f}\n"
            s += f"| ... unseen_nmi: {valid_result['unseen_nmi']:.4f}\n"
            if is_master():
                print(s)

            current_unseen_acc = valid_result['unseen_acc']
            if current_best_unseen <= current_unseen_acc:
                s = time_log()
                s += f"| ***** Best updated!\n" \
                     f"| ... previous best was at {best_epoch} epoch, {best_iter} iters\n" \
                     f"| ... unseen_acc: {current_best_unseen:.4f} (prev) -> {current_unseen_acc:.4f} (new)\n"
                current_best_unseen = current_unseen_acc
                best_iter = current_iter
                best_epoch = current_epoch

                if is_master():
                    # save checkpoint
                    ckpt = OrderedDict()
                    ckpt["model"] = model_m.state_dict()
                    ckpt["optimizer"] = optimizer.state_dict()
                    ckpt["scheduler"] = scheduler.state_dict()
                    ckpt["scaler"] = scaler.state_dict()
                    ckpt["stats"] = OrderedDict(epoch=current_epoch, iter=current_iter, max_epochs=max_epochs,
                                                best_unseen=current_best_unseen)
                    torch.save(ckpt, os.path.join(save_dir, "best.ckpt"))
                    s += f"| ... save checkpoint to {os.path.join(save_dir, 'best.ckpt')}"
                    print(s)
            else:
                s = time_log()
                s += f"| ***** Best not updated\n" \
                     f"| ... previous best was at {best_epoch} epoch, {best_iter} iters\n" \
                     f"| ... unseen_acc: {current_best_unseen:.4f} (best) vs. {current_unseen_acc:.4f} (now)\n"
                if is_master():
                    print(s)

            barrier()

        scheduler.step()
        current_epoch += 1


if __name__ == '__main__':
    args, config = prepare_config()
    run(config, args.debug, args.eval)