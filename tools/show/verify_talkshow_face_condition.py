#!/usr/bin/env python3
"""Validate a TalkSHOW WavLM face checkpoint for the main diffusion model."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "diffusion/model/audio_encoder/talkshow_face.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "talkshow_face_condition_verifier",
    MODULE_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
TalkSHOWFaceConditionModel = MODULE.TalkSHOWFaceConditionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="TalkSHOW ckpt-*.pth produced by reproductions/talkshow/train_face.sh",
    )
    parser.add_argument(
        "--wavlm-dir",
        type=Path,
        required=True,
        help="The exact local WavLM Large directory used by the main model.",
    )
    parser.add_argument(
        "--allow-legacy-audio-contract",
        action="store_true",
        help=(
            "Allow checkpoints without resume_contract metadata. This only "
            "checks tensor compatibility and is not recommended for formal runs."
        ),
    )
    parser.add_argument(
        "--required-face-loss",
        choices=("corrected", "legacy"),
        default="corrected",
        help="Required face-loss protocol recorded by modern checkpoints.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    wavlm_dir = args.wavlm_dir.expanduser().resolve()
    if not wavlm_dir.is_dir():
        raise SystemExit(f"WavLM directory does not exist: {wavlm_dir}")

    model = TalkSHOWFaceConditionModel(
        wavlm_model_path=wavlm_dir,
        require_audio_contract=not args.allow_legacy_audio_contract,
        required_face_loss=args.required_face_loss,
    )
    report = model.load_pretrained_checkpoint(checkpoint)
    model.eval()

    features = torch.randn(2, 8, 1024, dtype=torch.float32)
    # The main TalkSHOW dataset normalizes raw LMDB/rest IDs 21..24 to the
    # shared face/backbone runtime identity space 0..3.
    pid = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        expression = model(features, pid=pid)
        full_face = model.forward_face(features, pid=pid)

    if expression.shape != (2, 8, 100):
        raise RuntimeError(
            f"unexpected expression shape: {tuple(expression.shape)}"
        )
    if full_face.shape != (2, 8, 103):
        raise RuntimeError(
            f"unexpected full face shape: {tuple(full_face.shape)}"
        )
    torch.testing.assert_close(expression, full_face[..., 3:])

    print(f"checkpoint: {checkpoint}")
    print(f"layout: {report['layout']}")
    print(f"loaded_keys: {report['loaded_keys']}")
    print(f"ignored_prefix_counts: {report['ignored_prefix_counts']}")
    print(f"audio_contract: {report['audio_contract']}")
    print("expression_shape: (2, 8, 100)")
    print("full_face_shape: (2, 8, 103)")
    print("PASS: TalkSHOW checkpoint is compatible with the main diffusion model")


if __name__ == "__main__":
    main()
