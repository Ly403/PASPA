#!/usr/bin/env python3
"""Train the TalkSHOW WavLM face model with padded batches and optional DDP.

This is an opt-in runtime for per-device batches larger than one and/or more
than one GPU.  The released single-GPU, batch-size-one trainer remains the
default path in ``train_face.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader

try:
    from .face_batching import (
        DistributedLengthBucketBatchSampler,
        collate_face_whole_video,
    )
    from .face_resume import (
        capture_rng_state,
        get_rank_rng_state,
        get_resume_progress,
        load_face_checkpoint,
        restore_rng_state,
        restore_wrapper_state,
        validate_distributed_resume,
    )
except ImportError:
    # Direct execution places this script's directory on sys.path.
    from face_batching import (
        DistributedLengthBucketBatchSampler,
        collate_face_whole_video,
    )
    from face_resume import (
        capture_rng_state,
        get_rank_rng_state,
        get_resume_progress,
        load_face_checkpoint,
        restore_rng_state,
        restore_wrapper_state,
        validate_distributed_resume,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--speakers", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier() -> None:
    if _distributed():
        dist.barrier()


def _setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _masked_utterance_normalize(
    waveform: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    valid = attention_mask.to(dtype=waveform.dtype)
    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (waveform * valid).sum(dim=-1, keepdim=True) / counts
    centered = (waveform - mean) * valid
    variance = centered.square().sum(dim=-1, keepdim=True) / counts
    return centered / torch.sqrt(variance + 1e-7)


def _time_mask(
    motion_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return motion_mask[:, None, :].to(dtype=dtype)


def _masked_audio_middle(
    audio_middle: nn.Module,
    features: torch.Tensor,
    identity: torch.Tensor,
    motion_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the face audio head while clearing padded activations every layer."""

    mask = _time_mask(motion_mask, dtype=features.dtype)
    hidden = audio_middle.dropout(features) * mask
    if audio_middle.identity:
        identity_features = identity.reshape(identity.shape[0], -1, 1)
        identity_features = identity_features.repeat(1, 1, hidden.shape[2])
        identity_features = audio_middle.id_mlp(identity_features.to(torch.float32))
        hidden = torch.cat([hidden, identity_features], dim=1) * mask

    for layer in audio_middle.first_net.conv_layers:
        if getattr(layer, "norm_type", None) != "ln":
            raise RuntimeError(
                "variable-length face batching requires channel-wise LayerNorm"
            )
        hidden = layer(hidden) * mask
    return hidden


class PaddedWavLMFaceGenerator(nn.Module):
    """Mask-aware adapter around the pinned TalkSHOW face generator."""

    def __init__(self, generator: nn.Module) -> None:
        super().__init__()
        if getattr(generator, "audio_encoder_name", None) != "wavlm":
            raise RuntimeError(
                "variable-length face batching currently supports WavLM only"
            )
        if not getattr(generator, "freeze_audio_encoder", False):
            raise RuntimeError(
                "large-batch WavLM training requires the audio encoder to stay frozen"
            )
        if (
            getattr(generator.audio_encoder.config, "feat_extract_norm", None)
            != "layer"
        ):
            raise RuntimeError(
                "padded WavLM batching requires feat_extract_norm='layer'; "
                "use the documented WavLM Large checkpoint"
            )
        self.generator = generator

    def forward(
        self,
        waveform: torch.Tensor,
        gt_poses: torch.Tensor,
        identity: torch.Tensor,
        audio_lengths: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        motion_lengths: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        generator = self.generator

        batch_size = waveform.shape[0]
        waveform = waveform.reshape(batch_size, -1)
        if generator.audio_do_normalize:
            waveform = _masked_utterance_normalize(
                waveform, audio_attention_mask
            )
        else:
            waveform = waveform * audio_attention_mask.to(waveform.dtype)

        generator.audio_encoder.eval()
        with torch.no_grad():
            hidden_states = generator.audio_encoder(
                waveform,
                attention_mask=audio_attention_mask,
            ).last_hidden_state
            feature_lengths = (
                generator.audio_encoder._get_feat_extract_output_lengths(
                    audio_lengths
                )
            )

        mapped = generator.audio_feature_map(hidden_states)
        max_motion_length = gt_poses.shape[1]
        aligned_features: list[torch.Tensor] = []
        for index in range(batch_size):
            feature_length = min(
                int(feature_lengths[index].item()), mapped.shape[1]
            )
            motion_length = int(motion_lengths[index].item())
            if feature_length < 1 or motion_length < 1:
                raise RuntimeError("audio and motion lengths must be positive")
            feature = mapped[index, :feature_length].transpose(0, 1)[None]
            feature = F.interpolate(
                feature,
                size=motion_length,
                mode="linear",
                align_corners=True,
            )
            feature = F.pad(
                feature, (0, max_motion_length - motion_length)
            )
            aligned_features.append(feature)

        features = torch.cat(aligned_features, dim=0)
        mask = _time_mask(motion_mask, dtype=features.dtype)
        features = features * mask
        features = _masked_audio_middle(
            generator.audio_middle,
            features,
            identity,
            motion_mask,
        )

        outputs: list[torch.Tensor] = []
        for decoder, final_out in zip(generator.decoder, generator.final_out):
            decoded = features
            for layer in decoder:
                if getattr(layer, "norm_type", None) != "ln":
                    raise RuntimeError(
                        "variable-length face batching requires channel-wise LayerNorm"
                    )
                decoded = layer(decoded) * mask
            outputs.append(final_out(decoded) * mask)

        prediction = torch.cat(outputs, dim=1).transpose(1, 2)
        return prediction, None


def _masked_face_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    motion_mask: torch.Tensor,
    *,
    jaw_width: int,
    expression: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = motion_mask.to(dtype=prediction.dtype)
    valid_counts = valid.sum(dim=1).clamp_min(1.0)

    jaw_error = torch.abs(
        prediction[:, :, :jaw_width] - target[:, :, :jaw_width]
    ).mean(dim=-1)
    jaw_per_clip = (jaw_error * valid).sum(dim=1) / valid_counts
    jaw_loss = jaw_per_clip.mean()

    losses = {"MSELoss": jaw_loss}
    if expression:
        expression_error = (
            prediction[:, :, -100:] - target[:, :, -100:]
        ).square().mean(dim=-1)
        expression_per_clip = (
            expression_error * valid
        ).sum(dim=1) / valid_counts
        expression_loss = expression_per_clip.mean()
        losses["exp_loss"] = expression_loss
    else:
        expression_loss = jaw_loss.new_zeros(())

    return jaw_loss + expression_loss, losses


def _assert_fp32_model(model: nn.Module) -> None:
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            if torch.is_floating_point(value) and value.dtype != torch.float32:
                raise RuntimeError(
                    f"{kind} {name} has dtype {value.dtype}; "
                    "TalkSHOW face training must remain float32"
                )


def _build_train_dataset(
    config: Any,
    speakers: list[str],
) -> tuple[Any, list[int], str]:
    from data_utils import torch_data

    train_set = torch_data(
        data_root=config.Data.data_root,
        speakers=speakers,
        split="train",
        limbscaling=config.Data.pose.augmentation,
        normalization=config.Data.pose.normalization,
        norm_method=config.Data.pose.norm_method,
        split_trans_zero=False,
        num_pre_frames=config.Data.pose.pre_pose_length,
        num_frames=config.Data.pose.generate_length,
        num_generate_length=config.Data.pose.generate_length,
        aud_feat_win_size=config.Data.aud.aud_feat_win_size,
        aud_feat_dim=config.Data.aud.aud_feat_dim,
        feat_method=config.Data.aud.feat_method,
        context_info=config.Data.aud.context_info,
        smplx=True,
        audio_sr=22000,
        convert_to_6d=config.Data.pose.convert_to_6d,
        expression=config.Data.pose.expression,
        config=config,
    )
    train_set.get_dataset()

    audio_lengths: list[int] = []
    clip_datasets: list[Any] = []
    fingerprint = hashlib.sha256()
    data_root = Path(os.path.abspath(str(config.Data.data_root)))
    minimum_motion_length = int(config.Data.pose.generate_length)
    for key in sorted(train_set.dataset, key=str):
        clip = train_set.dataset[key]
        motion_length = int(clip.complete_data.shape[0])
        if motion_length < minimum_motion_length:
            continue
        if len(clip.all_dataset) != 1:
            raise RuntimeError(
                "whole-video face batching expects one item per clip"
            )
        audio_length = int(clip.audio_feat.shape[0])
        audio_lengths.append(audio_length)
        clip_datasets.append(clip.all_dataset)
        try:
            stable_key = Path(os.path.abspath(str(key))).relative_to(data_root)
        except ValueError as exc:
            raise RuntimeError(
                f"TalkSHOW clip path is outside data_root: {key}"
            ) from exc
        record = (
            f"{stable_key.as_posix()}\0speaker={clip.speaker}"
            f"\0audio={audio_length}\0motion={motion_length}\n"
        ).encode("utf-8")
        fingerprint.update(len(record).to_bytes(8, "little"))
        fingerprint.update(record)

    if not audio_lengths:
        raise RuntimeError("TalkSHOW face training dataset is empty")
    dataset = ConcatDataset(clip_datasets)
    if len(audio_lengths) != len(dataset):
        raise RuntimeError(
            "could not match TalkSHOW clip lengths to the concatenated dataset"
        )
    return dataset, audio_lengths, fingerprint.hexdigest()


def _assert_distributed_dataset_consistency(
    sample_count: int,
    fingerprint: str,
    device: torch.device,
) -> None:
    if not _distributed():
        return

    count = torch.tensor([sample_count], dtype=torch.long, device=device)
    counts = [torch.empty_like(count) for _ in range(dist.get_world_size())]
    dist.all_gather(counts, count)

    digest = torch.tensor(
        list(bytes.fromhex(fingerprint)),
        dtype=torch.uint8,
        device=device,
    )
    digests = [torch.empty_like(digest) for _ in range(dist.get_world_size())]
    dist.all_gather(digests, digest)

    count_values = [int(value.item()) for value in counts]
    digest_values = [bytes(value.cpu().tolist()).hex() for value in digests]
    if len(set(count_values)) != 1 or len(set(digest_values)) != 1:
        summaries = ", ".join(
            f"rank{rank}:count={rank_count},sha256={rank_digest[:12]}"
            for rank, (rank_count, rank_digest) in enumerate(
                zip(count_values, digest_values)
            )
        )
        raise RuntimeError(
            "TalkSHOW dataset count/order differs across ranks: " + summaries
        )


def _reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    names = sorted(metrics)
    values = torch.tensor(
        [metrics[name] for name in names],
        dtype=torch.float32,
        device=device,
    )
    if _distributed():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {name: float(value) for name, value in zip(names, values.cpu())}


def _active_face_loss_mode() -> str:
    return (
        "legacy"
        if os.environ.get("TALKSHOW_LEGACY_FACE_LOSS", "1") == "1"
        else "corrected"
    )


def _validate_face_loss_config(config: Any) -> None:
    configured = getattr(
        getattr(config, "Reproduction", None),
        "face_loss",
        None,
    )
    active = _active_face_loss_mode()
    if configured != active:
        raise RuntimeError(
            "run config and TALKSHOW_LEGACY_FACE_LOSS disagree: "
            f"config={configured!r}, active={active!r}"
        )


def _build_resume_contract(
    config: Any,
    wrapper: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_audio_model_files_sha256 = getattr(
        config.Model,
        "audio_model_files_sha256",
        None,
    )
    if raw_audio_model_files_sha256 is None:
        audio_model_files_sha256 = None
    elif isinstance(raw_audio_model_files_sha256, Mapping):
        audio_model_files_sha256 = dict(raw_audio_model_files_sha256)
    elif hasattr(raw_audio_model_files_sha256, "__dict__"):
        # trainer.config.load_JsonConfig recursively converts JSON mappings
        # into attribute objects, including the filename -> digest mapping.
        audio_model_files_sha256 = dict(
            vars(raw_audio_model_files_sha256)
        )
    else:
        raise RuntimeError(
            "Model.audio_model_files_sha256 must be a mapping-like object"
        )
    if audio_model_files_sha256 is not None:
        if not audio_model_files_sha256:
            raise RuntimeError(
                "Model.audio_model_files_sha256 cannot be empty"
            )
        for filename, digest in audio_model_files_sha256.items():
            if (
                not isinstance(filename, str)
                or not filename
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise RuntimeError(
                    "Model.audio_model_files_sha256 contains an invalid "
                    f"entry for {filename!r}"
                )
            try:
                int(digest, 16)
            except ValueError as error:
                raise RuntimeError(
                    "Model.audio_model_files_sha256 contains a non-hex "
                    f"digest for {filename!r}"
                ) from error
            audio_model_files_sha256[filename] = digest.lower()
    has_weight_contract = audio_model_files_sha256 is not None
    contract = {
        "schema_version": 2 if has_weight_contract else 1,
        "trainer": "talkshow_face_large_batch",
        "seed": args.seed,
        "speakers": list(args.speakers),
        "length_bucket": int(config.DataLoader.length_bucket),
        "drop_last": bool(config.DataLoader.drop_last),
        "num_workers": int(config.DataLoader.num_workers),
        "audio_encoder": str(config.Model.audio_encoder),
        "audio_model_revision": getattr(
            config.Model, "audio_model_revision", None
        ),
        "audio_hidden_size": int(config.Model.audio_hidden_size),
        "audio_config_sha256": getattr(
            config.Model, "audio_config_sha256", None
        ),
        "audio_preprocessor_sha256": getattr(
            config.Model, "audio_preprocessor_sha256", None
        ),
        "audio_do_normalize": bool(config.Model.audio_do_normalize),
        "freeze_audio_encoder": bool(config.Model.freeze_audio_encoder),
        "num_classes": int(wrapper.num_classes),
        "expression": bool(wrapper.expression),
        "optimizer": type(wrapper.generator_optimizer).__name__,
        "learning_rate": float(
            wrapper.generator_optimizer.param_groups[0]["lr"]
        ),
        "momentum": float(
            wrapper.generator_optimizer.param_groups[0].get("momentum", 0.0)
        ),
        "max_gradient_norm": float(config.Train.max_gradient_norm),
        "face_loss": _active_face_loss_mode(),
    }
    if has_weight_contract:
        contract["audio_model_files_sha256"] = dict(
            audio_model_files_sha256
        )
    return contract


def _preserve_legacy_audio_contract_on_resume(
    payload: Mapping[str, Any],
    current_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Never claim schema-2 weight provenance for a schema-1 lineage.

    A legacy checkpoint restores its serialized frozen ``audio_encoder.*``
    tensors.  Because schema 1 did not fingerprint those tensors, hashes from
    the current config directory cannot retroactively prove their origin.
    Keep the resumed lineage at schema 1 and omit file hashes.
    """

    saved_contract = payload.get("resume_contract")
    normalized = dict(current_contract)
    has_proven_schema_two_lineage = (
        isinstance(saved_contract, Mapping)
        and saved_contract.get("schema_version") == 2
        and isinstance(
            saved_contract.get("audio_model_files_sha256"),
            Mapping,
        )
        and bool(saved_contract["audio_model_files_sha256"])
    )
    if not has_proven_schema_two_lineage:
        normalized["schema_version"] = 1
        normalized.pop("audio_model_files_sha256", None)
    return normalized


def _gather_rng_states(device: torch.device) -> list[dict[str, Any]]:
    local_state = capture_rng_state(device)
    if not _distributed():
        return [local_state]
    states: list[dict[str, Any] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError("could not gather all rank RNG states")
    return [state for state in states if state is not None]


def _save_checkpoint(
    path: Path,
    wrapper: Any,
    *,
    epoch: int,
    global_steps: int,
    samples_seen: int,
    world_size: int,
    per_device_batch_size: int,
    steps_per_epoch: int,
    samples_per_epoch: int,
    dataset_fingerprint: str | None = None,
    resume_contract: dict[str, Any] | None = None,
    rng_states: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "generator": wrapper.state_dict(),
        "epoch": epoch,
        "global_steps": global_steps,
        "samples_seen": samples_seen,
        "distributed": {
            "world_size": world_size,
            "per_device_batch_size": per_device_batch_size,
            "global_batch_size": world_size * per_device_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "samples_per_epoch": samples_per_epoch,
        },
    }
    if dataset_fingerprint is not None:
        payload["dataset_fingerprint"] = dataset_fingerprint
    if resume_contract is not None:
        payload["resume_contract"] = resume_contract
    if rng_states is not None:
        payload["rng_states"] = rng_states
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _configure_rank_zero_logging(train_dir: Path) -> logging.Logger:
    logger = logging.getLogger("talkshow.face.large_batch")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s-%(lineno)d-%(message)s",
        datefmt="%m/%d %I:%M:%S %p",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(train_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def train(args: argparse.Namespace) -> None:
    upstream_dir = args.upstream_dir.resolve()
    config_file = args.config_file.resolve()
    save_dir = args.save_dir.resolve()
    os.chdir(upstream_dir)
    sys.path.insert(0, str(upstream_dir))

    from nets.init_model import init_model
    from trainer.config import load_JsonConfig

    config = load_JsonConfig(str(config_file))
    if config.Model.model_name != "s2g_face":
        raise RuntimeError("large-batch entry point only supports s2g_face")
    if config.Model.audio_encoder != "wavlm":
        raise RuntimeError(
            "batch sizes larger than one currently require --audio-encoder wavlm"
        )
    if not config.Data.whole_video:
        raise RuntimeError("large-batch face training requires whole_video=true")
    if not config.DataLoader.variable_length_batching:
        raise RuntimeError(
            "large-batch face config must enable variable_length_batching"
        )
    _validate_face_loss_config(config)

    requested_world_size = int(config.Distributed.requested_world_size)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != requested_world_size:
        raise RuntimeError(
            f"torchrun world size is {world_size}, "
            f"but the run config requests {requested_world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TalkSHOW face training")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds visible CUDA device count "
            f"{torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")

    try:
        _setup_seed(args.seed)
        for name in (
            "smplx_npz_path",
            "extra_joint_path",
            "j14_regressor_path",
        ):
            os.environ[name] = str(getattr(config, name))

        date = os.environ.get("TALKSHOW_RUN_DATE") or time.strftime(
            "%Y-%m-%d", time.localtime()
        )
        train_dir = save_dir / f"{date}-{args.exp_name}-{config.Log.name}"
        if rank == 0:
            train_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_file, train_dir / config_file.name)
        _barrier()
        logger = (
            _configure_rank_zero_logging(train_dir)
            if rank == 0
            else logging.getLogger("talkshow.face.large_batch.null")
        )

        epochs = int(config.Train.epochs)
        model_args = SimpleNamespace(gpu=str(device), infer=False)
        wrapper = init_model(config.Model.model_name, model_args, config)
        resume_contract = _build_resume_contract(config, wrapper, args)

        resume_payload = None
        resume_progress = None
        if args.resume_from is not None:
            resume_payload = load_face_checkpoint(args.resume_from)
            resume_contract = _preserve_legacy_audio_contract_on_resume(
                resume_payload,
                resume_contract,
            )
            resume_progress = get_resume_progress(
                resume_payload,
                total_epochs=epochs,
                require_samples_seen=True,
            )

        if resume_payload is not None and resume_progress is not None:
            restore_wrapper_state(wrapper, resume_payload, resume_progress)
            # The full checkpoint includes WavLM Large.  Its CPU copy is no
            # longer needed after the wrapper/optimizer restore; retain only
            # the small resume metadata through dataset validation.
            del resume_payload["generator"]
        _assert_fp32_model(wrapper.generator)

        adapter = PaddedWavLMFaceGenerator(wrapper.generator).to(device)
        if world_size > 1:
            forward_model: nn.Module = DistributedDataParallel(
                adapter,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            forward_model = adapter

        # Model initialization must be identical before DDP broadcasts rank 0.
        # Rank-specific training RNG keeps dropout streams independent.
        _setup_seed(args.seed + rank)

        dataset, audio_lengths, dataset_fingerprint = _build_train_dataset(
            config, args.speakers
        )
        _assert_distributed_dataset_consistency(
            len(dataset), dataset_fingerprint, device
        )
        per_device_batch_size = int(config.DataLoader.batch_size)
        bucket_multiplier = int(config.DataLoader.length_bucket)
        batch_sampler = DistributedLengthBucketBatchSampler(
            audio_lengths,
            batch_size=per_device_batch_size,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
            bucket_multiplier=bucket_multiplier,
            drop_last=bool(config.DataLoader.drop_last),
        )
        train_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_face_whole_video,
            num_workers=int(config.DataLoader.num_workers),
            pin_memory=True,
        )
        steps_per_epoch = len(batch_sampler)
        samples_per_epoch = (
            sum(len(batch) for batch in batch_sampler) * world_size
        )
        if steps_per_epoch < 1 or samples_per_epoch < 1:
            raise RuntimeError("TalkSHOW face batch sampler produced no data")

        start_epoch = 0
        global_steps = 0
        samples_seen = 0
        if resume_payload is not None and resume_progress is not None:
            warnings = validate_distributed_resume(
                resume_payload,
                resume_progress,
                world_size=world_size,
                per_device_batch_size=per_device_batch_size,
                steps_per_epoch=steps_per_epoch,
                samples_per_epoch=samples_per_epoch,
                dataset_fingerprint=dataset_fingerprint,
                expected_contract=resume_contract,
            )
            rank_rng_state, rng_warning = get_rank_rng_state(
                resume_payload,
                rank=rank,
                world_size=world_size,
            )
            if rank_rng_state is not None:
                restore_rng_state(rank_rng_state, device)
            if rng_warning is not None:
                warnings.append(rng_warning)
            if rank == 0:
                for warning in warnings:
                    logger.warning(warning)
            start_epoch = resume_progress.start_epoch
            global_steps = resume_progress.global_steps
            if resume_progress.samples_seen is None:
                raise RuntimeError(
                    "large-batch resume requires samples_seen"
                )
            samples_seen = resume_progress.samples_seen
            logger.info(
                "resume checkpoint=%s checkpoint_epoch=%d "
                "start_epoch=%d global_steps=%d samples_seen=%d",
                args.resume_from.resolve(),
                resume_progress.checkpoint_epoch,
                start_epoch,
                global_steps,
                samples_seen,
            )
            del resume_payload

        initial_global_steps = global_steps
        initial_samples_seen = samples_seen
        logger.info(
            "start_training world_size=%d per_device_batch=%d "
            "global_batch=%d start_epoch=%d epochs=%d steps_per_epoch=%d "
            "dataset=%d precision=float32",
            world_size,
            per_device_batch_size,
            world_size * per_device_batch_size,
            start_epoch,
            epochs,
            len(train_loader),
            len(dataset),
        )
        logger.info(
            "dataset_fingerprint=%s samples_per_epoch=%d",
            dataset_fingerprint,
            samples_per_epoch,
        )

        for epoch in range(start_epoch, epochs):
            batch_sampler.set_epoch(epoch)
            forward_model.train()
            epoch_totals: dict[str, float] = {}
            epoch_steps = 0

            for batch in train_loader:
                waveform = batch["aud_feat"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
                poses = batch["poses"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
                target = poses.permute(0, 2, 1)
                if wrapper.expression:
                    expression = batch["expression"].to(
                        device, dtype=torch.float32, non_blocking=True
                    )
                    target = torch.cat(
                        [target, expression.permute(0, 2, 1)], dim=2
                    )

                speaker = batch["speaker"].to(device, non_blocking=True) - 20
                identity = F.one_hot(speaker, wrapper.num_classes)
                audio_lengths_batch = batch["audio_lengths"].to(
                    device, non_blocking=True
                )
                audio_attention_mask = batch["audio_attention_mask"].to(
                    device, non_blocking=True
                )
                motion_lengths = batch["motion_lengths"].to(
                    device, non_blocking=True
                )
                motion_mask = batch["motion_mask"].to(
                    device, non_blocking=True
                )

                prediction, _ = forward_model(
                    waveform,
                    target,
                    identity,
                    audio_lengths_batch,
                    audio_attention_mask,
                    motion_lengths,
                    motion_mask,
                )
                jaw_width = (
                    6
                    if os.environ.get("TALKSHOW_LEGACY_FACE_LOSS", "1") == "1"
                    else wrapper.dim_list[1]
                )
                loss, loss_tensors = _masked_face_loss(
                    prediction,
                    target,
                    motion_mask,
                    jaw_width=jaw_width,
                    expression=wrapper.expression,
                )

                wrapper.generator_optimizer.zero_grad()
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    wrapper.generator.parameters(),
                    config.Train.max_gradient_norm,
                )
                wrapper.generator_optimizer.step()

                global_steps += 1
                wrapper.global_step += 1
                epoch_steps += 1
                local_batch_size = int(waveform.shape[0])
                samples_seen += local_batch_size * world_size

                metrics = {
                    name: float(value.detach())
                    for name, value in loss_tensors.items()
                }
                metrics["grad"] = float(gradient.detach())
                metrics = _reduce_metrics(metrics, device)
                for name, value in metrics.items():
                    epoch_totals[name] = epoch_totals.get(name, 0.0) + value

                if (
                    rank == 0
                    and global_steps % int(config.Log.print_every) == 0
                ):
                    logger.info(
                        "epoch=%d global_steps=%d %s",
                        epoch,
                        global_steps,
                        ",".join(
                            f"{name}={epoch_totals[name] / epoch_steps:.6f}"
                            for name in sorted(epoch_totals)
                        ),
                    )

            completed_epochs = epoch - start_epoch + 1
            expected_global_steps = (
                initial_global_steps + completed_epochs * steps_per_epoch
            )
            expected_samples_seen = (
                initial_samples_seen + completed_epochs * samples_per_epoch
            )
            if (
                epoch_steps != steps_per_epoch
                or global_steps != expected_global_steps
                or samples_seen != expected_samples_seen
            ):
                raise RuntimeError(
                    "face training schedule mismatch: "
                    f"epoch_steps={epoch_steps}/{steps_per_epoch}, "
                    f"global_steps={global_steps}/{expected_global_steps}, "
                    f"samples_seen={samples_seen}/{expected_samples_seen}"
                )

            if rank == 0:
                logger.info(
                    "epoch=%d global_steps=%d samples_seen=%d steps=%d %s",
                    epoch,
                    global_steps,
                    samples_seen,
                    epoch_steps,
                    ",".join(
                        f"{name}={epoch_totals[name] / max(epoch_steps, 1):.6f}"
                        for name in sorted(epoch_totals)
                    ),
                )

            should_save = (
                (epoch + 1) % int(config.Log.save_every) == 0
                or (epoch + 1) == 30
                or (epoch + 1) == epochs
            )
            if should_save:
                _barrier()
                rng_states = _gather_rng_states(device)
                if rank == 0:
                    _save_checkpoint(
                        train_dir / f"ckpt-{epoch}.pth",
                        wrapper,
                        epoch=epoch,
                        global_steps=global_steps,
                        samples_seen=samples_seen,
                        world_size=world_size,
                        per_device_batch_size=per_device_batch_size,
                        steps_per_epoch=steps_per_epoch,
                        samples_per_epoch=samples_per_epoch,
                        dataset_fingerprint=dataset_fingerprint,
                        resume_contract=resume_contract,
                        rng_states=rng_states,
                    )
                _barrier()
    finally:
        if _distributed():
            dist.destroy_process_group()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
