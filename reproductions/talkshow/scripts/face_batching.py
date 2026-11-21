#!/usr/bin/env python3
"""Batching utilities for variable-length TalkSHOW whole-video face samples.

This module is deliberately independent from the TalkSHOW upstream package so it
can be tested before a patched checkout is created.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler, default_collate


_SEQUENCE_KEYS = frozenset(("aud_feat", "poses", "expression"))
_GENERATED_KEYS = frozenset(
    (
        "audio_lengths",
        "audio_attention_mask",
        "motion_lengths",
        "motion_mask",
    )
)


def _as_float32_matrix(value: Any, *, key: str, sample_index: int) -> torch.Tensor:
    """Convert one channel-first sequence to a contiguous CPU FP32 tensor."""

    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sample {sample_index} field {key!r} is not a numeric array"
        ) from exc
    if tensor.ndim != 2:
        raise ValueError(
            f"sample {sample_index} field {key!r} must have shape [C, T], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(
            f"sample {sample_index} field {key!r} must be non-empty, "
            f"got {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


def _cast_collated_floats_to_fp32(value: Any) -> Any:
    """Keep metadata structure while normalizing floating tensors to FP32."""

    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {
            key: _cast_collated_floats_to_fp32(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_cast_collated_floats_to_fp32(child) for child in value)
    if isinstance(value, list):
        return [_cast_collated_floats_to_fp32(child) for child in value]
    return value


def _collate_metadata(values: list[Any], *, key: str) -> Any:
    """Stack regular metadata and retain path-like or unsupported values as lists."""

    if all(isinstance(value, Path) for value in values):
        return list(values)
    try:
        return _cast_collated_floats_to_fp32(default_collate(values))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        # Strings are already handled by default_collate. For opaque metadata,
        # preserving the per-sample values is safer than silently dropping them.
        if any(
            torch.is_tensor(value)
            or isinstance(value, (np.ndarray, Mapping, int, float, bool))
            for value in values
        ):
            raise ValueError(f"cannot collate metadata field {key!r}") from exc
        return list(values)


def collate_face_whole_video(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Right-pad a batch of channel-first whole-video face samples.

    Required input fields have the upstream TalkSHOW shapes:

    * ``aud_feat``: ``[1, audio_samples]``
    * ``poses``: ``[pose_channels, motion_frames]``
    * ``expression``: ``[expression_channels, motion_frames]``

    The returned tensors stay on CPU. Floating arrays are converted to FP32.
    ``audio_attention_mask`` is integer-valued for direct use with Hugging Face
    WavLM, while ``motion_mask`` is boolean for masked head/loss operations.
    """

    if not samples:
        raise ValueError("cannot collate an empty face batch")
    if any(not isinstance(sample, Mapping) for sample in samples):
        raise TypeError("every face sample must be a mapping")

    expected_keys = set(samples[0].keys())
    missing_required = _SEQUENCE_KEYS.difference(expected_keys)
    if missing_required:
        raise KeyError(
            "face sample is missing required fields: "
            + ", ".join(sorted(missing_required))
        )
    reserved = _GENERATED_KEYS.intersection(expected_keys)
    if reserved:
        raise KeyError(
            "face sample contains collate-generated fields: "
            + ", ".join(sorted(reserved))
        )

    audio_items: list[torch.Tensor] = []
    pose_items: list[torch.Tensor] = []
    expression_items: list[torch.Tensor] = []

    for sample_index, sample in enumerate(samples):
        sample_keys = set(sample.keys())
        if sample_keys != expected_keys:
            missing = sorted(expected_keys.difference(sample_keys))
            extra = sorted(sample_keys.difference(expected_keys))
            raise KeyError(
                f"sample {sample_index} has inconsistent keys; "
                f"missing={missing}, extra={extra}"
            )

        audio = _as_float32_matrix(
            sample["aud_feat"], key="aud_feat", sample_index=sample_index
        )
        poses = _as_float32_matrix(
            sample["poses"], key="poses", sample_index=sample_index
        )
        expression = _as_float32_matrix(
            sample["expression"], key="expression", sample_index=sample_index
        )
        if audio.shape[0] != 1:
            raise ValueError(
                f"sample {sample_index} aud_feat must contain one waveform "
                f"channel, got {audio.shape[0]}"
            )
        if poses.shape[1] != expression.shape[1]:
            raise ValueError(
                f"sample {sample_index} motion length mismatch: "
                f"poses={poses.shape[1]}, expression={expression.shape[1]}"
            )

        audio_items.append(audio)
        pose_items.append(poses)
        expression_items.append(expression)

    pose_channels = {item.shape[0] for item in pose_items}
    expression_channels = {item.shape[0] for item in expression_items}
    if len(pose_channels) != 1:
        raise ValueError(f"pose channel counts differ across samples: {pose_channels}")
    if len(expression_channels) != 1:
        raise ValueError(
            "expression channel counts differ across samples: "
            f"{expression_channels}"
        )

    batch_size = len(samples)
    audio_lengths = torch.tensor(
        [item.shape[1] for item in audio_items], dtype=torch.long
    )
    motion_lengths = torch.tensor(
        [item.shape[1] for item in pose_items], dtype=torch.long
    )
    max_audio_length = int(audio_lengths.max().item())
    max_motion_length = int(motion_lengths.max().item())
    pose_channels_value = pose_items[0].shape[0]
    expression_channels_value = expression_items[0].shape[0]

    audio_batch = torch.zeros(
        (batch_size, 1, max_audio_length), dtype=torch.float32
    )
    pose_batch = torch.zeros(
        (batch_size, pose_channels_value, max_motion_length), dtype=torch.float32
    )
    expression_batch = torch.zeros(
        (batch_size, expression_channels_value, max_motion_length),
        dtype=torch.float32,
    )
    for index, (audio, poses, expression) in enumerate(
        zip(audio_items, pose_items, expression_items)
    ):
        audio_batch[index, :, : audio.shape[1]] = audio
        pose_batch[index, :, : poses.shape[1]] = poses
        expression_batch[index, :, : expression.shape[1]] = expression

    audio_steps = torch.arange(max_audio_length).unsqueeze(0)
    motion_steps = torch.arange(max_motion_length).unsqueeze(0)
    audio_attention_mask = (audio_steps < audio_lengths.unsqueeze(1)).to(torch.long)
    motion_mask = motion_steps < motion_lengths.unsqueeze(1)

    result: dict[str, Any] = {
        "aud_feat": audio_batch,
        "poses": pose_batch,
        "expression": expression_batch,
        "audio_lengths": audio_lengths,
        "audio_attention_mask": audio_attention_mask,
        "motion_lengths": motion_lengths,
        "motion_mask": motion_mask,
    }
    for key in sorted(expected_keys.difference(_SEQUENCE_KEYS)):
        result[key] = _collate_metadata(
            [sample[key] for sample in samples], key=key
        )
    return result


class DistributedLengthBucketBatchSampler(Sampler[list[int]]):
    """Deterministic length-aware global batches split evenly across ranks.

    ``batch_size`` is the maximum batch size on each of ``num_replicas`` ranks.
    With ``drop_last=False``, the final global batch is padded only to a multiple
    of ``num_replicas``. Consequently every rank performs the same number of
    synchronized steps and receives the same local size, without padding the
    tail all the way to a full global batch.
    """

    def __init__(
        self,
        lengths: Sequence[int | float],
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        bucket_multiplier: int = 8,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        if isinstance(num_replicas, bool) or not isinstance(num_replicas, int):
            raise TypeError("num_replicas must be an integer")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if isinstance(bucket_multiplier, bool) or not isinstance(
            bucket_multiplier, int
        ):
            raise TypeError("bucket_multiplier must be an integer")
        if bucket_multiplier < 1:
            raise ValueError("bucket_multiplier must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        normalized_lengths: list[float] = []
        for index, value in enumerate(lengths):
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"length at index {index} is not numeric") from exc
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(
                    f"length at index {index} must be finite and positive, "
                    f"got {value!r}"
                )
            normalized_lengths.append(normalized)

        self.lengths = tuple(normalized_lengths)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = bool(drop_last)
        self.bucket_multiplier = bucket_multiplier
        self.shuffle = bool(shuffle)
        self.epoch = 0

    @property
    def global_batch_size(self) -> int:
        return self.num_replicas * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.global_batch_size
        return math.ceil(len(self.lengths) / self.global_batch_size)

    @staticmethod
    def _pad_to_world_size(batch: list[int], world_size: int) -> list[int]:
        if not batch:
            return batch
        padding = (-len(batch)) % world_size
        if padding:
            original = tuple(batch)
            batch.extend(original[index % len(original)] for index in range(padding))
        return batch

    def _global_batches(self) -> list[list[int]]:
        if not self.lengths:
            return []

        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        pool_size = self.global_batch_size * self.bucket_multiplier
        full_batches: list[list[int]] = []
        tail: list[int] | None = None
        for pool_start in range(0, len(indices), pool_size):
            pool = indices[pool_start : pool_start + pool_size]
            # Python sorting is stable, so equal-length examples retain their
            # epoch-specific shuffled order.
            pool.sort(key=self.lengths.__getitem__)
            for batch_start in range(0, len(pool), self.global_batch_size):
                batch = pool[
                    batch_start : batch_start + self.global_batch_size
                ]
                if len(batch) == self.global_batch_size:
                    full_batches.append(batch)
                else:
                    tail = batch

        if self.shuffle:
            rng.shuffle(full_batches)
        if not self.drop_last and tail:
            full_batches.append(
                self._pad_to_world_size(tail, self.num_replicas)
            )
        return full_batches

    def __iter__(self) -> Iterator[list[int]]:
        for global_batch in self._global_batches():
            local_batch = global_batch[self.rank :: self.num_replicas]
            if local_batch:
                yield local_batch
