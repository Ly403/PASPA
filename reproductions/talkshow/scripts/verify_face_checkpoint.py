#!/usr/bin/env python3
"""Check that a TalkSHOW face checkpoint is complete and numerically finite."""

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from .face_resume import validate_face_wrapper_state
except ImportError:
    from face_resume import validate_face_wrapper_state


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}")
    return value


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_resume_metadata(payload: Mapping[str, Any]) -> None:
    """Validate metadata required for exact same-protocol DDP continuation."""

    distributed = payload.get("distributed")
    if not isinstance(distributed, Mapping):
        raise RuntimeError("checkpoint has no distributed resume metadata")
    world_size = _positive_int(
        distributed.get("world_size"),
        "distributed.world_size",
    )
    per_device_batch = _positive_int(
        distributed.get("per_device_batch_size"),
        "distributed.per_device_batch_size",
    )
    global_batch = _positive_int(
        distributed.get("global_batch_size"),
        "distributed.global_batch_size",
    )
    steps_per_epoch = _positive_int(
        distributed.get("steps_per_epoch"),
        "distributed.steps_per_epoch",
    )
    samples_per_epoch = _positive_int(
        distributed.get("samples_per_epoch"),
        "distributed.samples_per_epoch",
    )
    if global_batch != world_size * per_device_batch:
        raise RuntimeError(
            "distributed.global_batch_size is inconsistent with world/batch"
        )

    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError("checkpoint epoch must be a non-negative integer")
    global_steps = _positive_int(payload.get("global_steps"), "global_steps")
    samples_seen = _positive_int(payload.get("samples_seen"), "samples_seen")
    if global_steps != (epoch + 1) * steps_per_epoch:
        raise RuntimeError(
            "checkpoint global_steps is inconsistent with epoch metadata"
        )
    if samples_seen != (epoch + 1) * samples_per_epoch:
        raise RuntimeError(
            "checkpoint samples_seen is inconsistent with epoch metadata"
        )

    fingerprint = payload.get("dataset_fingerprint")
    if not _is_sha256(fingerprint):
        raise RuntimeError(
            "checkpoint dataset_fingerprint must be a SHA-256 hex digest"
        )

    contract = payload.get("resume_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("checkpoint has no resume_contract dictionary")
    required_contract = {
        "schema_version",
        "trainer",
        "seed",
        "speakers",
        "length_bucket",
        "drop_last",
        "num_workers",
        "audio_encoder",
        "audio_model_revision",
        "audio_hidden_size",
        "audio_config_sha256",
        "audio_preprocessor_sha256",
        "audio_do_normalize",
        "freeze_audio_encoder",
        "num_classes",
        "expression",
        "optimizer",
        "learning_rate",
        "momentum",
        "max_gradient_norm",
        "face_loss",
    }
    missing_contract = sorted(required_contract.difference(contract))
    if missing_contract:
        raise RuntimeError(
            "checkpoint resume_contract is incomplete; missing: "
            + ", ".join(missing_contract)
        )
    schema_version = contract["schema_version"]
    if schema_version not in (1, 2):
        raise RuntimeError("unsupported resume_contract schema_version")
    if schema_version == 2:
        model_hashes = contract.get("audio_model_files_sha256")
        if not isinstance(model_hashes, Mapping) or not model_hashes:
            raise RuntimeError(
                "schema 2 resume_contract has no audio model file hashes"
            )
        for filename, digest in model_hashes.items():
            if (
                not isinstance(filename, str)
                or not filename
                or not _is_sha256(digest)
            ):
                raise RuntimeError(
                    "resume_contract audio model file hashes are invalid"
                )
    if contract["trainer"] != "talkshow_face_large_batch":
        raise RuntimeError("checkpoint resume_contract has the wrong trainer")
    if contract["audio_encoder"] != "wavlm":
        raise RuntimeError("large-batch resume_contract must use WavLM")
    if not _is_sha256(contract["audio_config_sha256"]):
        raise RuntimeError("resume_contract audio config digest is invalid")
    if not _is_sha256(contract["audio_preprocessor_sha256"]):
        raise RuntimeError(
            "resume_contract audio preprocessor digest is invalid"
        )
    if contract["face_loss"] not in ("legacy", "corrected"):
        raise RuntimeError("resume_contract face_loss is invalid")
    speakers = contract["speakers"]
    if (
        not isinstance(speakers, list)
        or len(speakers) != 4
        or not all(isinstance(name, str) and name for name in speakers)
    ):
        raise RuntimeError("resume_contract speakers must contain four names")

    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, list) or len(rng_states) != world_size:
        raise RuntimeError(
            "checkpoint rng_states count does not match distributed.world_size"
        )
    for rank, state in enumerate(rng_states):
        if not isinstance(state, Mapping):
            raise RuntimeError(f"checkpoint RNG state for rank {rank} is invalid")
        missing_rng = {
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
        }.difference(state)
        if missing_rng:
            raise RuntimeError(
                f"checkpoint RNG state for rank {rank} is incomplete"
            )
        numpy_state = state["numpy"]
        if not isinstance(numpy_state, Mapping):
            raise RuntimeError(
                f"checkpoint NumPy RNG state for rank {rank} is invalid"
            )
        numpy_values = numpy_state.get("state")
        torch_cpu = state["torch_cpu"]
        torch_cuda = state["torch_cuda"]
        if (
            not torch.is_tensor(numpy_values)
            or numpy_values.dtype != torch.int64
            or numpy_values.ndim != 1
        ):
            raise RuntimeError(
                f"checkpoint NumPy RNG values for rank {rank} are invalid"
            )
        for name, value in (
            ("torch_cpu", torch_cpu),
            ("torch_cuda", torch_cuda),
        ):
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.uint8
                or value.ndim != 1
            ):
                raise RuntimeError(
                    f"checkpoint {name} RNG state for rank {rank} is invalid"
                )


def iter_tensors(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from iter_tensors(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_tensors(child, f"{prefix}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--min-global-steps", type=int, default=1)
    parser.add_argument("--require-face-wrapper-state", action="store_true")
    parser.add_argument("--require-resume-metadata", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise RuntimeError(f"checkpoint does not exist: {args.checkpoint}")
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if payload.get("epoch") != args.expected_epoch:
        raise RuntimeError(
            f"checkpoint epoch is {payload.get('epoch')}, "
            f"expected {args.expected_epoch}"
        )
    global_steps = payload.get("global_steps")
    if not isinstance(global_steps, int) or global_steps < args.min_global_steps:
        raise RuntimeError(
            f"checkpoint global_steps is {global_steps}, "
            f"expected at least {args.min_global_steps}"
        )
    state = payload.get("generator")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint has no non-empty generator state")
    if args.require_face_wrapper_state or args.require_resume_metadata:
        validate_face_wrapper_state(state)
    if args.require_resume_metadata:
        validate_resume_metadata(payload)

    tensors = list(iter_tensors(state))
    if not tensors:
        raise RuntimeError("checkpoint generator state contains no tensors")
    nonfinite = [
        name
        for name, tensor in tensors
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all()
    ]
    if nonfinite:
        raise RuntimeError(
            "checkpoint contains non-finite tensors: " + ", ".join(nonfinite[:10])
        )
    print(
        f"verified checkpoint epoch={args.expected_epoch}, "
        f"global_steps={global_steps}, generator_tensors={len(tensors)}, "
        f"resume_metadata={args.require_resume_metadata}, "
        f"path={args.checkpoint}"
    )


if __name__ == "__main__":
    main()
