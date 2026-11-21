# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
import re
import os.path as osp
import numpy as np
import torch
from accelerate.state import DistributedType
from diffusion.utils.logger import get_root_logger
from termcolor import colored


def find_model(model_path: str):
    """Load a checkpoint from a local .pth path.

    Previously this delegated to a Sana-era ``tools/download.py`` helper
    that supported HuggingFace download URLs; this project only uses
    local checkpoints, so it now reduces to a plain ``torch.load``."""
    assert os.path.isfile(model_path), f"Could not find checkpoint at {model_path}"
    return torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def cleanup_checkpoints_by_metric(work_dir: str, keep_best_n: int, mode: str = "max"):
    assert mode in ("max", "min")
    if keep_best_n is None or keep_best_n <= 0:
        return
    # epoch_12_step_300_0.123.pth
    # epoch_12_0.123.pth
    # epoch_12_step_300_0.123_suffix.pth
    pattern = re.compile(
        rf"^epoch_(\d+)(?:_step_(\d+))?_(?P<metric>{_NUM})(?:_.*)?\.pth$"
    )

    ckpts = []
    for fn in os.listdir(work_dir):
        if not (fn.startswith("epoch_") and fn.endswith(".pth")):
            continue
        m = pattern.match(fn)
        if not m:
            continue
        metric = float(m.group("metric"))
        ckpts.append((fn, metric))

    if len(ckpts) <= keep_best_n:
        return

    ckpts.sort(key=lambda x: x[1], reverse=(mode == "max"))
    for fn, _ in ckpts[keep_best_n:]:
        path = os.path.join(work_dir, fn)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

def cleanup_fsdp_checkpoints_by_metric(work_dir: str, keep_best_n: int, mode: str = "max"):
    assert mode in ("max", "min")
    if keep_best_n is None or keep_best_n <= 0:
        return

    items = []
    for d in os.listdir(work_dir):
        ckpt_dir = os.path.join(work_dir, d)
        if not (d.startswith("epoch_") and os.path.isdir(ckpt_dir)):
            continue
        meta_path = os.path.join(ckpt_dir, "metadata.pth")
        if not os.path.isfile(meta_path):
            continue
        meta = torch.load(meta_path, map_location="cpu")
        if "best_metric" not in meta:
            continue
        metric = float(meta["best_metric"])
        items.append((d, metric))

    if len(items) <= keep_best_n:
        return

    items.sort(key=lambda x: x[1], reverse=(mode == "max"))
    for d, _ in items[keep_best_n:]:
        ckpt_dir = os.path.join(work_dir, d)
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        # 同步删除平铺的 .pth
        pth_path = ckpt_dir + ".pth"
        if os.path.isfile(pth_path) or os.path.islink(pth_path):
            try:
                os.remove(pth_path)
            except FileNotFoundError:
                pass

def extract_epoch_step(filename):
    m = re.match(r"^epoch_(\d+)(?:_step_(\d+))?(?:_.*)?\.pth$", filename)
    if not m:
        return (-1, -1)
    epoch = int(m.group(1))
    step = int(m.group(2)) if m.group(2) is not None else -1
    return (epoch, step)

def save_checkpoint(
    work_dir,
    epoch,
    model,
    accelerator=None,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    generator=torch.Generator(device="cpu").manual_seed(42),
    keep_last=False,
    keep_last_n=3, # only useful when keep_last is True
    step=None,
    add_symlink=False,
    add_suffix=None,
    best_metric=None,
    best_metric_mode=None,
    rng_state=None,
):
    if accelerator is not None and accelerator.distributed_type == DistributedType.FSDP:
        return save_checkpoint_fsdp(
            work_dir=work_dir,
            epoch=epoch,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            generator=generator,
            keep_last=keep_last,
            keep_last_n=keep_last_n,
            step=step,
            add_symlink=add_symlink,
            add_suffix=add_suffix,
            best_metric=best_metric,
            best_metric_mode=best_metric_mode,
            rng_state=rng_state,
        )
    else:
        return save_checkpoint_ddp(
            work_dir=work_dir,
            epoch=epoch,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            generator=generator,
            keep_last=keep_last,
            keep_last_n=keep_last_n,
            step=step,
            add_symlink=add_symlink,
            add_suffix=add_suffix,
            best_metric=best_metric,
            best_metric_mode=best_metric_mode,
            rng_state=rng_state,
        )


def save_checkpoint_ddp(
    work_dir,
    epoch,
    model,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    generator=torch.Generator(device="cpu").manual_seed(42),
    keep_last=False,
    keep_last_n=3,
    step=None,
    add_symlink=False,
    add_suffix=None,
    best_metric=None,
    best_metric_mode=None,
    rng_state=None,
):
    os.makedirs(work_dir, exist_ok=True)
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict["state_dict_ema"] = model_ema.state_dict()
    if optimizer is not None:
        state_dict["optimizer"] = [optimizer_i.state_dict() for optimizer_i in optimizer] \
            if isinstance(optimizer, list) else optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict["scheduler"] = [lr_scheduler_i.state_dict() for lr_scheduler_i in lr_scheduler] \
            if isinstance(lr_scheduler, list) else lr_scheduler.state_dict()
    if epoch is not None:
        state_dict["epoch"] = epoch
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split(".pth")[0] + f"_step_{step}.pth"
    if best_metric is not None:
        file_path = file_path.replace(".pth", f"_{float(best_metric)}.pth") # 记下具体的指标值
    if add_suffix is not None:
        file_path = file_path.replace(".pth", f"_{add_suffix}.pth")
    if rng_state is None:
        rng_state = {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "generator": generator.get_state(),
        }
    state_dict["rng_state"] = rng_state

    logger = get_root_logger()
    torch.save(state_dict, file_path)
    logger.info(f"Saved checkpoint of epoch {epoch} to {file_path.format(epoch)}.")
    if keep_last:
        # for i in range(epoch):
        #     previous_ckgt = file_path.format(i)
        #     if os.path.exists(previous_ckgt):
        #         os.remove(previous_ckgt)
        # 提取epoch和step数值进行排序
        if best_metric is None:
            checkpoints = sorted(
                [d for d in os.listdir(work_dir) if d.startswith("epoch_") and d.endswith(".pth")],
                key=extract_epoch_step,
            )
            for ckpt_file in checkpoints[:-keep_last_n]:
                os.remove(os.path.join(work_dir, ckpt_file))
        else: # 按指标来排序
            best_metric_mode = "max" if best_metric_mode is None else best_metric_mode
            cleanup_checkpoints_by_metric(work_dir, keep_last_n, best_metric_mode)
    if add_symlink:
        link_path = os.path.join(os.path.dirname(file_path), "latest.pth")
        if os.path.exists(link_path) or os.path.islink(link_path):
            os.remove(link_path)
        os.symlink(os.path.abspath(file_path), link_path)

    return file_path


def save_checkpoint_fsdp(
    work_dir,
    epoch,
    accelerator=None,
    lr_scheduler=None,
    generator=torch.Generator(device="cpu").manual_seed(42),
    keep_last=False,
    keep_last_n=3,
    step=None,
    add_symlink=False,
    add_suffix=None,
    best_metric=None,
    best_metric_mode=None,
    rng_state=None,
):
    """FSDP checkpoint save function, sharding"""
    logger = get_root_logger()

    checkpoint_dir = os.path.join(work_dir, f"epoch_{epoch}")
    if step is not None:
        checkpoint_dir = checkpoint_dir + f"_step_{step}"
    if add_suffix is not None:
        checkpoint_dir = checkpoint_dir + f"_{add_suffix}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_dir = os.path.join(checkpoint_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    accelerator.save_state(model_dir)

    if accelerator.is_main_process:
        metadata = dict()
        if rng_state is None:
            rng_state = {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "generator": generator.get_state(),
            }
        metadata["rng_state"] = rng_state
        if lr_scheduler is not None:
            metadata["scheduler"] = lr_scheduler.state_dict()
        if epoch is not None:
            metadata["epoch"] = epoch

        torch.save(metadata, os.path.join(checkpoint_dir, "metadata.pth"))

        if keep_last:
            if best_metric is None:
                checkpoints = sorted(
                    [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d)) and d.startswith("epoch_")],
                    key=extract_epoch_step,
                )
                for old_ckpt in checkpoints[:-keep_last_n]:
                    old_path = os.path.join(work_dir, old_ckpt)
                    if os.path.exists(old_path):
                        import shutil
                        shutil.rmtree(old_path)
            else:
                raise NotImplementedError("Best metric deletion is not supported for FSDP mode")

        if add_symlink:
            link_path = os.path.join(work_dir, "latest.pth")
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.remove(link_path)
            os.symlink(os.path.abspath(checkpoint_dir), link_path)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # add model symlink
        model_link_path = checkpoint_dir + ".pth"
        state_dict = torch.load(os.path.join(model_dir, "pytorch_model_fsdp.bin"), map_location="cpu")
        torch.save({"state_dict": state_dict}, model_link_path)

    accelerator.wait_for_everyone()
    return checkpoint_dir

def _parse_epoch_step_from_name(name: str):
    """
    支持：
      epoch_12.pth
      epoch_12_step_300.pth
      epoch_12_step_300_0.123.pth
      epoch_12_step_300_0.123_suffix.pth
    """
    m = re.match(r"^epoch_(\d+)(?:_step_(\d+))?(?:_.*)?(?:\.pth)?$", name)
    if not m:
        return None
    epoch = int(m.group(1))
    step = int(m.group(2)) if m.group(2) is not None else 0
    return epoch, step

def _resolve_latest_ckpt_path(ckpt_dir: str):
    latest_link = osp.join(ckpt_dir, "latest.pth")
    if osp.exists(latest_link):
        return osp.realpath(latest_link)

    # os.listdir 返回顺序是 arbitrary，不可依赖 :contentReference[oaicite:3]{index=3}
    cands = []
    for fn in os.listdir(ckpt_dir):
        if not (fn.startswith("epoch_") and fn.endswith(".pth")):
            continue
        if fn == "latest.pth":
            continue
        parsed = _parse_epoch_step_from_name(fn)
        if parsed is None:
            continue
        epoch, step = parsed
        full = osp.join(ckpt_dir, fn)
        try:
            mtime = osp.getmtime(full)
        except FileNotFoundError:
            continue
        cands.append((epoch, step, mtime, full))

    if not cands:
        return None
    cands.sort(key=lambda x: (x[0], x[1], x[2]))
    return cands[-1][3]


def load_checkpoint(
    checkpoint,
    model,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    load_ema=False,
    resume_optimizer=True,
    resume_lr_scheduler=True,
    # null_embed_path=None,
    FSDP=False,
):
    if FSDP:
        return load_checkpoint_fsdp(
            checkpoint=checkpoint,
            model=model,
        )
    else:
        return load_checkpoint_ddp(
            checkpoint=checkpoint,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            load_ema=load_ema,
            resume_optimizer=resume_optimizer,
            resume_lr_scheduler=resume_lr_scheduler,
            # null_embed_path=null_embed_path,
        )


def load_checkpoint_ddp(
    checkpoint,
    model,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    load_ema=False,
    resume_optimizer=True,
    resume_lr_scheduler=True,
    # null_embed_path=None,
):
    assert isinstance(checkpoint, str)
    logger = get_root_logger()
    ckpt_file = checkpoint
    checkpoint = find_model(ckpt_file)

    # state_dict_keys = ["pos_embed", "base_model.pos_embed", "model.pos_embed"]
    # for key in state_dict_keys:
    #     if key in checkpoint["state_dict"]:
    #         del checkpoint["state_dict"][key]
    #         if "state_dict_ema" in checkpoint and key in checkpoint["state_dict_ema"]:
    #             del checkpoint["state_dict_ema"][key]
    #         break

    if load_ema:
        state_dict = checkpoint["state_dict_ema"]
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)  # to be compatible with the official checkpoint
    
    # if null_embed_path is not None:
    #     null_embed = torch.load(null_embed_path, map_location="cpu")
    #     state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
    rng_state = checkpoint.get("rng_state", None)

    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    if model_ema is not None:
        if "state_dict_ema" in checkpoint:
            model_ema.load_state_dict(checkpoint["state_dict_ema"], strict=False)
        else:
            logger.warning( # 打一条警告
                colored("No state_dict_ema found in checkpoint, using model.state_dict() instead", "yellow", attrs=["bold"])
            )
            model_ema.load_state_dict(model.state_dict(), strict=False)
    if optimizer is not None and resume_optimizer:
        if isinstance(optimizer, list):
            for i in range(len(optimizer)):
                optimizer[i].load_state_dict(checkpoint["optimizer"][i])
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None and resume_lr_scheduler:
        if isinstance(lr_scheduler, list):
            for i in range(len(lr_scheduler)):
                lr_scheduler[i].load_state_dict(checkpoint["scheduler"][i])
        else:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = None
    if optimizer is not None and resume_optimizer:
        epoch_match = re.search(r"epoch_(\d+)", ckpt_file)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        else:
            epoch = checkpoint.get("epoch", 0)
        logger.info(
            f"Resume checkpoint of epoch {epoch} from {ckpt_file}. Load ema: {load_ema}, "
            f"resume optimizer： {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}."
        )
        return epoch, missing, unexpect, rng_state
    logger.info(f"Load checkpoint from {ckpt_file}. Load ema: {load_ema}.")
    return epoch, missing, unexpect, rng_state


def load_checkpoint_fsdp(
    checkpoint,
    model,
):
    assert isinstance(checkpoint, str)
    logger = get_root_logger()

    # 1 load model
    if ".pth" in checkpoint:
        state_dict_model = find_model(checkpoint)
        state_dict_model = state_dict_model.get("state_dict", state_dict_model)
    else:
        if os.path.isfile(checkpoint):
            checkpoint = os.path.dirname(checkpoint)
        assert os.path.isdir(checkpoint), f"Checkpoint directory {checkpoint} does not exist!"

        state_dict_model = find_model(os.path.join(checkpoint, "model", "pytorch_model_fsdp.bin"))

    # state_dict_keys = ["pos_embed", "base_model.pos_embed", "model.pos_embed"]
    # for key in state_dict_keys:
    #     if key in state_dict_model:
    #         del state_dict_model[key]
    #         break

    missing, unexpect = model.load_state_dict(state_dict_model, strict=False)
    logger.info(f"Load checkpoint of {checkpoint}.")

    return None, missing, unexpect, None
