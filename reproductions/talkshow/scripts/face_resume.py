#!/usr/bin/env python3
"""Shared validation and RNG helpers for TalkSHOW face-training resume."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class ResumeProgress:
    """Validated progress counters read from one epoch-boundary checkpoint."""

    checkpoint_epoch: int
    start_epoch: int
    global_steps: int
    samples_seen: int | None


def _require_int(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(
            f"checkpoint {name} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def validate_face_wrapper_state(generator_state: Any) -> None:
    """Require both model and optimizer state for true training continuation."""

    if not isinstance(generator_state, dict) or not generator_state:
        raise RuntimeError(
            "resume checkpoint has no non-empty generator/wrapper state"
        )
    model_state = generator_state.get("generator")
    if not isinstance(model_state, Mapping) or not model_state:
        raise RuntimeError(
            "resume checkpoint wrapper state has no non-empty generator model"
        )
    optimizer_state = generator_state.get("generator_optim")
    if not isinstance(optimizer_state, Mapping):
        raise RuntimeError(
            "resume checkpoint wrapper state has no generator_optim; "
            "weight-only checkpoints cannot be used for resume"
        )
    optimizer_values = optimizer_state.get("state")
    optimizer_groups = optimizer_state.get("param_groups")
    if not isinstance(optimizer_values, Mapping):
        raise RuntimeError(
            "resume checkpoint generator_optim.state must be a dictionary"
        )
    if not isinstance(optimizer_groups, list) or not optimizer_groups:
        raise RuntimeError(
            "resume checkpoint generator_optim.param_groups must be non-empty"
        )
    for index, group in enumerate(optimizer_groups):
        if (
            not isinstance(group, Mapping)
            or not isinstance(group.get("params"), list)
            or not group["params"]
        ):
            raise RuntimeError(
                "resume checkpoint optimizer param group "
                f"{index} has no parameter IDs"
            )
    uses_momentum = any(
        float(group.get("momentum", 0.0)) != 0.0
        for group in optimizer_groups
    )
    if uses_momentum and not optimizer_values:
        raise RuntimeError(
            "resume checkpoint generator_optim.state is empty despite "
            "non-zero optimizer momentum"
        )


def load_face_checkpoint(path: Path) -> dict[str, Any]:
    """Load and validate the common outer TalkSHOW face checkpoint contract."""

    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"resume checkpoint does not exist: {checkpoint_path}"
        )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("resume checkpoint payload must be a dictionary")
    validate_face_wrapper_state(payload.get("generator"))
    return payload


def get_resume_progress(
    payload: Mapping[str, Any],
    *,
    total_epochs: int,
    require_samples_seen: bool,
) -> ResumeProgress:
    """Validate counters and return the first epoch that still needs training."""

    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int):
        raise TypeError("total_epochs must be an integer")
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")

    checkpoint_epoch = _require_int(
        payload.get("epoch"),
        name="epoch",
        minimum=0,
    )
    global_steps = _require_int(
        payload.get("global_steps"),
        name="global_steps",
        minimum=1,
    )
    samples_value = payload.get("samples_seen")
    if samples_value is None and not require_samples_seen:
        samples_seen = None
    else:
        samples_seen = _require_int(
            samples_value,
            name="samples_seen",
            minimum=1,
        )

    start_epoch = checkpoint_epoch + 1
    if start_epoch >= total_epochs:
        raise RuntimeError(
            f"resume checkpoint already completed epoch {checkpoint_epoch}; "
            f"--epochs must be greater than {start_epoch}, got {total_epochs}"
        )
    return ResumeProgress(
        checkpoint_epoch=checkpoint_epoch,
        start_epoch=start_epoch,
        global_steps=global_steps,
        samples_seen=samples_seen,
    )


def restore_wrapper_state(
    wrapper: Any,
    payload: Mapping[str, Any],
    progress: ResumeProgress,
) -> None:
    """Restore model/optimizer state and the wrapper's internal step counter."""

    wrapper.load_state_dict(payload["generator"])
    wrapper.global_step = progress.global_steps
    optimizer = getattr(wrapper, "generator_optimizer", None)
    generator = getattr(wrapper, "generator", None)
    if optimizer is not None and generator is not None:
        try:
            device = next(generator.parameters()).device
        except StopIteration:
            device = None
        if device is not None:
            for optimizer_state in optimizer.state.values():
                for name, value in tuple(optimizer_state.items()):
                    if torch.is_tensor(value):
                        optimizer_state[name] = value.to(device)


def validate_distributed_resume(
    payload: Mapping[str, Any],
    progress: ResumeProgress,
    *,
    world_size: int,
    per_device_batch_size: int,
    steps_per_epoch: int,
    samples_per_epoch: int,
    dataset_fingerprint: str,
    expected_contract: Mapping[str, Any],
) -> list[str]:
    """Fail closed on schedule changes that would make DDP resume ambiguous."""

    distributed = payload.get("distributed")
    if not isinstance(distributed, Mapping):
        raise RuntimeError(
            "large-batch resume requires checkpoint distributed metadata; "
            "use the released single-GPU resume path for legacy checkpoints"
        )

    expected_distributed = {
        "world_size": world_size,
        "per_device_batch_size": per_device_batch_size,
        "global_batch_size": world_size * per_device_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "samples_per_epoch": samples_per_epoch,
    }
    for name, expected in expected_distributed.items():
        observed = _require_int(
            distributed.get(name),
            name=f"distributed.{name}",
            minimum=1,
        )
        if observed != expected:
            raise RuntimeError(
                "resume configuration differs from the checkpoint: "
                f"distributed.{name}={observed}, current={expected}"
            )

    expected_global_steps = (
        progress.start_epoch * steps_per_epoch
    )
    if progress.global_steps != expected_global_steps:
        raise RuntimeError(
            "checkpoint global_steps is inconsistent with its completed epoch: "
            f"{progress.global_steps} != {expected_global_steps}"
        )
    if progress.samples_seen is None:
        raise RuntimeError(
            "large-batch resume checkpoint has no samples_seen counter"
        )
    expected_samples_seen = (
        progress.start_epoch * samples_per_epoch
    )
    if progress.samples_seen != expected_samples_seen:
        raise RuntimeError(
            "checkpoint samples_seen is inconsistent with its completed epoch: "
            f"{progress.samples_seen} != {expected_samples_seen}"
        )

    warnings: list[str] = []
    saved_fingerprint = payload.get("dataset_fingerprint")
    if saved_fingerprint is None:
        warnings.append(
            "resume checkpoint predates dataset_fingerprint metadata; "
            "sample count/schedule were checked, but dataset layout "
            "could not be verified"
        )
    elif saved_fingerprint != dataset_fingerprint:
        raise RuntimeError(
            "resume dataset fingerprint differs from the checkpoint: "
            f"{saved_fingerprint} != {dataset_fingerprint}"
        )

    saved_contract = payload.get("resume_contract")
    if saved_contract is None:
        warnings.append(
            "resume checkpoint predates resume_contract metadata; "
            "model state and distributed schedule were checked, but all "
            "training options could not be verified"
        )
    elif not isinstance(saved_contract, Mapping):
        raise RuntimeError("checkpoint resume_contract must be a dictionary")
    else:
        if saved_contract.get("schema_version") == 1:
            warnings.append(
                "resume checkpoint uses audio contract schema 1 and has no "
                "WavLM weight-file hashes; its resumed lineage must remain "
                "schema 1 because current files cannot prove the origin of "
                "the serialized frozen audio encoder"
            )
        for name, expected in expected_contract.items():
            if name not in saved_contract:
                raise RuntimeError(
                    f"checkpoint resume_contract has no {name!r} field"
                )
            observed = saved_contract[name]
            if observed != expected:
                raise RuntimeError(
                    "resume training contract differs from the checkpoint: "
                    f"{name}={observed!r}, current={expected!r}"
                )
    return warnings


def capture_rng_state(device: torch.device | None) -> dict[str, Any]:
    """Capture all RNGs that affect the epoch-boundary training continuation."""

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Keep the payload compatible with torch.load(weights_only=True), which
        # rejects a pickled NumPy ndarray even when the checkpoint is trusted.
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(
                numpy_state[1].astype(np.int64, copy=True)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if device is not None:
        if device.type != "cuda":
            raise ValueError("RNG device must be CUDA or None")
        state["torch_cuda"] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_rng_state(
    state: Mapping[str, Any],
    device: torch.device | None,
) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    if not isinstance(state, Mapping):
        raise RuntimeError("checkpoint RNG state must be a dictionary")
    required = ("python", "numpy", "torch_cpu")
    missing = [name for name in required if name not in state]
    if missing:
        raise RuntimeError(
            "checkpoint RNG state is incomplete; missing: "
            + ", ".join(missing)
        )
    torch_cpu = state["torch_cpu"]
    if not torch.is_tensor(torch_cpu) or torch_cpu.dtype != torch.uint8:
        raise RuntimeError("checkpoint torch_cpu RNG state is invalid")

    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise RuntimeError("checkpoint NumPy RNG state is invalid")
    numpy_values = numpy_state.get("state")
    if (
        not torch.is_tensor(numpy_values)
        or numpy_values.dtype != torch.int64
        or numpy_values.ndim != 1
    ):
        raise RuntimeError("checkpoint NumPy RNG values are invalid")
    try:
        normalized_numpy_state = (
            str(numpy_state["bit_generator"]),
            numpy_values.cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint NumPy RNG state is incomplete") from exc

    random.setstate(state["python"])
    np.random.set_state(normalized_numpy_state)
    torch.set_rng_state(torch_cpu.cpu())
    if device is not None:
        if device.type != "cuda":
            raise ValueError("RNG device must be CUDA or None")
        torch_cuda = state.get("torch_cuda")
        if not torch.is_tensor(torch_cuda) or torch_cuda.dtype != torch.uint8:
            raise RuntimeError("checkpoint torch_cuda RNG state is invalid")
        torch.cuda.set_rng_state(torch_cuda.cpu(), device=device)


def get_rank_rng_state(
    payload: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Select one rank's RNG state, retaining compatibility with old checkpoints."""

    states = payload.get("rng_states")
    if states is None:
        return (
            None,
            "resume checkpoint predates per-rank RNG metadata; optimizer/model "
            "progress will resume correctly, but the continuation is not "
            "bitwise identical to an uninterrupted run",
        )
    if not isinstance(states, (list, tuple)) or len(states) != world_size:
        raise RuntimeError(
            "checkpoint rng_states count does not match the current world size"
        )
    state = states[rank]
    if not isinstance(state, Mapping):
        raise RuntimeError(f"checkpoint RNG state for rank {rank} is invalid")
    return state, None
