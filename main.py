from typing import Dict
from collections import OrderedDict
import os
import time
import pprint
import wandb
import torch
import torch.cuda.amp as amp
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils.clip_grad import clip_grad_norm_
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from utils.config_utils import prepare_config
from utils.wandb_utils import set_wandb
from utils.random_utils import set_seed
from utils.dist_utils import set_dist, is_distributed_set, is_master, barrier, get_world_size
from utils.dist_utils import all_reduce_dict
from utils.print_utils import time_log
from utils.param_utils import count_params, compute_param_norm

from build import (build_transform, build_dataset, build_dataloader,
                   build_optimizer, build_scheduler, build_scaler)
from wrapper import ImageNet1KClassifier


def train_epoch(
        model: ImageNet1KClassifier,
        optimizer,
        scheduler,
        scaler,
        dataloader,
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

    data_start_time = time.time()
    for it, (img, label) in enumerate(dataloader):

        s = time_log()
        s += f"Current iter: {current_iter} (epoch done: {it / len(dataloader) * 100:.2f} %)\n"

        # -------------------------------- data -------------------------------- #
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        data_time = time.time() - data_start_time

        # -------------------------------- loss -------------------------------- #
        if it % num_accum == 0:
            optimizer.zero_grad(set_to_none=True)

        if it % num_accum == (num_accum - 1):  # update step
            forward_start_time = time.time()
            with amp.autocast(enabled=fp16):
                output = model(img, label)  # {"loss", "acc1", "acc5"}
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
            scheduler.step()
            step_time = time.time() - step_start_time

        elif isinstance(model, DistributedDataParallel):  # non-update step and DDP
            with model.no_sync():
                with amp.autocast(enabled=fp16):
                    output = model(img, label)  # {"loss", "acc1", "acc5"}

                loss = output["loss"]
                loss = loss / num_accum
                scaler.scale(loss).backward()

        else:  # non-update step and not DDP
            with amp.autocast(enabled=fp16):
                output = model(img, label)  # {"loss", "acc1", "acc5"}

            loss = output["loss"]
            loss = loss / num_accum
            scaler.scale(loss).backward()

        # -------------------------------- print -------------------------------- #

        if (it > 0) and (it % print_interval == 0):
            output = all_reduce_dict(output, op="mean")
            param_norm = compute_param_norm(model.parameters())
            lr = scheduler.get_last_lr()[0]

            s += f"... loss: {output['loss'].item():.6f}\n"
            s += f"... acc1: {output['acc1'].item():.4f}\n"
            s += f"... acc5: {output['acc5'].item():.4f}\n"
            s += f"... LR: {lr:.6f}\n"
            s += f"... grad/param norm: {grad_norm.item():.3f} / {param_norm.item():.3f}\n"
            s += f"... batch_size x num_accum x gpus = " \
                 f"{int(label.shape[0])} x {num_accum} x {get_world_size()}\n"
            s += f"... data/fwd/bwd/step time: " \
                 f"{data_time:.3f} / {forward_time:.3f} / {backward_time:.3f} / {step_time:.3f}"

            if is_master():
                print(s)
                wandb.log({
                    "train_loss": output["loss"].item(),
                    "train_acc1": output["acc1"].item(),
                    "train_acc5": output["acc5"].item(),
                    "grad_norm": grad_norm.item(),
                    "param_norm": param_norm.item(),
                    "lr": lr,
                    "iterations": current_iter,
                })

        current_iter += 1
        data_start_time = time.time()

    return current_iter


def valid_epoch(
        model: ImageNet1KClassifier,
        dataloader,
        cfg: Dict,
        device: torch.device,
        current_iter: int,
) -> Dict:
    print_interval = 4  # heuristic
    fp16 = cfg.get("fp16", False)

    model.eval()
    torch.set_grad_enabled(False)  # same as 'with torch.no_grad():'

    result = dict(loss=0.0, acc1=0.0, acc5=0.0, count=0)
    for it, (img, label) in enumerate(dataloader):
        # -------------------------------- data -------------------------------- #
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # -------------------------------- loss -------------------------------- #
        with amp.autocast(enabled=fp16):
            output = model(img, label)  # {"loss", "acc1", "acc5"}

        result["loss"] += output["loss"]
        result["acc1"] += output["acc1"]
        result["acc5"] += output["acc5"]
        result["count"] += 1

        if (it > 0) and (it % print_interval == 0):
            s = f"... samples: {result['count'] * int(label.shape[0]) * get_world_size()} " \
                f"(valid done: {it / len(dataloader) * 100:.2f} %)"
            if is_master():
                print(s)

    result["loss"] /= result["count"]
    result["acc1"] /= result["count"]
    result["acc5"] /= result["count"]
    result.pop("count")
    result = all_reduce_dict(result, op="mean")

    output = dict()
    output["loss"] = result["loss"].item()
    output["acc1"] = result["acc1"].item()
    output["acc5"] = result["acc5"].item()

    if is_master():
        wandb.log({
            "valid_loss": output["loss"],
            "valid_acc1": output["acc1"],
            "valid_acc5": output["acc5"],
            "iterations": current_iter,
        })

    return output


def run(cfg: Dict, debug: bool = False) -> None:
    # ======================================================================================== #
    # Initialize
    # ======================================================================================== #
    device, local_rank = set_dist(device_type="cuda")
    print(local_rank)

    if is_master():
        pprint.pprint(cfg)  # print config to check if all arguments are correctly given.

    print(f"{local_rank}: wandbset start")
    save_dir = set_wandb(cfg, force_mode="disabled" if debug else None)
    print(f"{local_rank}: wandbset end")
    set_seed(seed=cfg["seed"] + local_rank)
    print(f"{local_rank}: wandbset end")

    # ======================================================================================== #
    # Data
    # ======================================================================================== #
    print("{local_rank}: dataset start")
    img_size = cfg["img_size"]  # 224
    data_dir = cfg["data_dir"]

    train_label_dataset = build_dataset(data_dir, is_train=True, is_label=True, cfg=cfg["dataset"])
    train_unlabel_dataset = build_dataset(data_dir, is_train=True, is_label=False, cfg=cfg["dataset"],
                                          unlabeled_idxs=train_label_dataset.unlabeled_idxs)
    # Loader 도 바껴야함..
    train_dataloader = build_dataloader(train_dataset, is_train=True, cfg=cfg["dataloader"]["train"])

    valid_dataset = build_dataset(data_dir, is_train=False, transform=valid_transform)
    valid_dataloader = build_dataloader(valid_dataset, is_train=False, cfg=cfg["dataloader"]["valid"])

    # ======================================================================================== #
    # Model
    # ======================================================================================== #
    model = ImageNet1KClassifier(cfg)
    model = model.to(device)
    print(f"{local_rank}{is_distributed_set()}")
    if is_distributed_set():
        print(f"model DDP {local_rank}")
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=device)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # optional for ImageNet
        model_m = model.module  # actual model without wrapping
    else:
        model_m = model

    if is_master():
        print(model)
        p1, p2 = count_params(model_m.parameters())
        print(f"Model parameters: {p1} tensors, {p2} elements.")

    if cfg["resume"]["checkpoint"] is not None:
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
                                iter_per_epoch=len(train_dataloader),
                                num_epoch=cfg["trainer"]["max_peochs"],
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

    # -------- status -------- #
    current_epoch = 0
    current_iter = 0
    current_best1 = 0.0  # accuracy top-1
    current_best5 = 0.0  # accuracy top-5
    if ckpt is not None:
        current_epoch = ckpt["stats"]["epoch"]
        current_iter = ckpt["stats"]["iter"]
        current_best1 = ckpt["stats"]["best1"]
        current_best5 = ckpt["stats"]["best5"]

    # -------- check -------- #
    if ckpt is not None:  # resumed
        valid_result = valid_epoch(model, valid_dataloader, train_cfg, device, current_iter)
        s = time_log()
        s += f"Resume valid epoch {current_epoch} / {max_epochs}\n"
        s += f"... loss: {valid_result['loss']:.6f}\n"
        s += f"... acc1: {valid_result['acc1']:.4f}\n"
        s += f"... acc5: {valid_result['acc5']:.4f}"
        if is_master():
            print(s)

    # -------- main loop -------- #
    while current_epoch < max_epochs:
        if is_master():
            s = time_log()
            s += f"Start train epoch {current_epoch} / {max_epochs} (iter: {current_iter})"
            print(s)

        if is_distributed_set():
            # reset random seed of sampler, sampler should be DistributedSampler.
            train_dataloader.sampler.set_epoch(current_epoch)  # noqa

        # -------- train body -------- #
        epoch_start_time = time.time()  # second
        current_iter = train_epoch(model, optimizer, scheduler, scaler, train_dataloader,
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
                                        best1=current_best1, best5=current_best5)
            torch.save(ckpt, os.path.join(save_dir, "latest.ckpt"))
            s += f"... save checkpoint to {os.path.join(save_dir, 'latest.ckpt')}"
            print(s)

        barrier()
        # -------- valid body -------- #
        if current_epoch % valid_interval == 0:

            s = time_log()
            s += f"Start valid epoch {current_epoch} / {max_epochs} (iter: {current_iter})"
            if is_master():
                print(s)

            valid_start_time = time.time()  # second
            valid_result = valid_epoch(model, valid_dataloader, train_cfg, device, current_iter)
            valid_time = time.time() - valid_start_time

            s = time_log()
            s += f"End valid epoch {current_epoch} / {max_epochs}, time: {valid_time:.3f} s\n"
            s += f"... loss: {valid_result['loss']:.6f}\n"
            s += f"... acc1: {valid_result['acc1']:.4f}\n"
            s += f"... acc5: {valid_result['acc5']:.4f}"
            if is_master():
                print(s)

            current_acc1 = valid_result['acc1']
            current_acc5 = valid_result['acc5']
            if current_best1 <= current_acc1:
                s = time_log()
                s += f"Best updated!\n" \
                     f"... acc1: {current_best1:.4f} (prev) -> {current_acc1:.4f} (new)\n" \
                     f"... acc5: {current_best5:.4f} (prev) -> {current_acc5:.4f} (new)\n"
                current_best1 = current_acc1
                current_best5 = current_acc5
                if is_master():
                    # save checkpoint
                    ckpt = OrderedDict()
                    ckpt["model"] = model_m.state_dict()
                    ckpt["optimizer"] = optimizer.state_dict()
                    ckpt["scheduler"] = scheduler.state_dict()
                    ckpt["scaler"] = scaler.state_dict()
                    ckpt["stats"] = OrderedDict(epoch=current_epoch, iter=current_iter,
                                                best1=current_best1, best5=current_best5)
                    torch.save(ckpt, os.path.join(save_dir, "best.ckpt"))
                    s += f"... save checkpoint to {os.path.join(save_dir, 'latest.ckpt')}"
                    print(s)
            else:
                s = time_log()
                s += f"Best not updated.\n" \
                     f"... acc1: {current_best1:.4f} (best) vs. {current_acc1:.4f} (now)\n" \
                     f"... acc5: {current_best5:.4f} (best) vs. {current_best5:.4f} (now)"
                if is_master():
                    print(s)

            barrier()
        current_epoch += 1


if __name__ == '__main__':
    args, config = prepare_config()
    run(config, args.debug)
