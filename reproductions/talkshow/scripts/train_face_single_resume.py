#!/usr/bin/env python3
"""Resume the released single-GPU TalkSHOW face trainer at an epoch boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from .face_resume import (
        get_resume_progress,
        load_face_checkpoint,
        restore_wrapper_state,
    )
except ImportError:
    from face_resume import (
        get_resume_progress,
        load_face_checkpoint,
        restore_wrapper_state,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--speakers", nargs="+", required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _nested_value(
    value: Mapping[str, Any],
    path: tuple[str, ...],
) -> tuple[bool, Any]:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            return False, None
        current = current[name]
    return True, current


def validate_previous_run_config(
    checkpoint: Path,
    current_config: Mapping[str, Any],
) -> list[str]:
    """Compare semantic single-GPU options when the old run config is present."""

    previous_config_path = checkpoint.resolve().parent / "face.json"
    if not previous_config_path.is_file():
        return [
            "resume checkpoint has no adjacent face.json; model state will be "
            "checked strictly, but all previous run options cannot be verified"
        ]
    with previous_config_path.open("r", encoding="utf-8") as handle:
        previous_config = json.load(handle)
    if not isinstance(previous_config, Mapping):
        raise RuntimeError("previous run face.json must contain a dictionary")

    paths = (
        ("Model", "model_name"),
        ("Model", "encoder_choice"),
        ("Model", "audio_encoder"),
        ("Model", "audio_model_revision"),
        ("Model", "audio_hidden_size"),
        ("Model", "audio_config_sha256"),
        ("Model", "audio_preprocessor_sha256"),
        ("Model", "audio_do_normalize"),
        ("Model", "freeze_audio_encoder"),
        ("Data", "data_root"),
        ("Data", "pose", "convert_to_6d"),
        ("Data", "pose", "expression"),
        ("DataLoader", "batch_size"),
        ("Reproduction", "face_loss"),
    )
    warnings: list[str] = []
    for path in paths:
        previous_found, previous = _nested_value(previous_config, path)
        current_found, current = _nested_value(current_config, path)
        dotted = ".".join(path)
        if not previous_found:
            warnings.append(
                f"previous face.json predates {dotted}; this option could not "
                "be checked"
            )
            continue
        if not current_found:
            raise RuntimeError(f"current face config has no {dotted}")
        if previous != current:
            raise RuntimeError(
                "resume configuration differs from the previous run: "
                f"{dotted}={previous!r}, current={current!r}"
            )
    return warnings


def validate_single_epoch_progress(
    *,
    start_epoch: int,
    global_steps: int,
    steps_per_epoch: int,
) -> None:
    if steps_per_epoch < 1:
        raise RuntimeError("single-GPU face train loader produced no batches")
    expected_global_steps = start_epoch * steps_per_epoch
    if global_steps != expected_global_steps:
        raise RuntimeError(
            "checkpoint global_steps is inconsistent with its completed epoch: "
            f"{global_steps} != {expected_global_steps}"
        )


def main() -> None:
    args = parse_args()
    upstream_dir = args.upstream_dir.resolve()
    config_file = args.config_file.resolve()
    resume_from = args.resume_from.resolve()

    with config_file.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    try:
        total_epochs = raw_config["Train"]["epochs"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("face config has no Train.epochs value") from exc

    config_warnings = validate_previous_run_config(
        resume_from,
        raw_config,
    )

    os.chdir(upstream_dir)
    sys.path.insert(0, str(upstream_dir))
    sys.argv = [
        str(upstream_dir / "scripts" / "train.py"),
        "--gpu",
        str(args.gpu),
        "--save_dir",
        str(args.save_dir.resolve()),
        "--exp_name",
        args.exp_name,
        "--speakers",
        *args.speakers,
        "--seed",
        str(args.seed),
        "--config_file",
        str(config_file),
    ]

    from trainer.Trainer import Trainer

    trainer = Trainer()
    payload = load_face_checkpoint(resume_from)
    if "distributed" in payload:
        raise RuntimeError(
            "a large-batch/DDP checkpoint must be resumed with the same "
            "--num-gpus and --batch-size-per-gpu settings"
        )
    progress = get_resume_progress(
        payload,
        total_epochs=total_epochs,
        require_samples_seen=False,
    )
    validate_single_epoch_progress(
        start_epoch=progress.start_epoch,
        global_steps=progress.global_steps,
        steps_per_epoch=len(trainer.train_loader),
    )
    restore_wrapper_state(trainer.generator, payload, progress)
    del payload
    trainer.start_epoch = progress.start_epoch
    trainer.global_steps = progress.global_steps
    for warning in config_warnings:
        print(f"resume warning: {warning}", file=sys.stderr)
    print(
        "resuming TalkSHOW face training "
        f"from={resume_from} checkpoint_epoch={progress.checkpoint_epoch} "
        f"start_epoch={progress.start_epoch} "
        f"global_steps={progress.global_steps}"
    )
    trainer.train()


if __name__ == "__main__":
    main()
