#!/usr/bin/env python3
"""Reject a TalkSHOW checkpoint with an incompatible saved run contract.

The main trainer stores its resolved configuration beside each checkpoint:

    <run>/output/checkpoints/epoch_*.pth
    <run>/output/config.yaml

This verifier intentionally uses only the Python standard library so it can
fail before Accelerate launches workers.  In addition to scale fields, it
requires the current raw-21..24 to model-0..3 identity contract, a four-person
main backbone, and a face adapter that receives model-local IDs directly.
Older TalkSHOW main-model checkpoints without this explicit identity contract
are deliberately rejected because their style-identity weights used a
different index space.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


IDENTITY_CONTRACT = "raw21_24_to_model0_3"
EXPECTED_IDENTITY_RANGE = {
    "raw_pid_min": 21,
    "raw_pid_max": 24,
    "model_pid_min": 0,
    "model_pid_max": 3,
}
EXPECTED_MAIN_NUM_PERSON = 4
EXPECTED_FACE_SPEAKER_ID_OFFSET = 0


class ResumeProfileError(RuntimeError):
    """The saved run and requested TalkSHOW scale profiles are incompatible."""


def _path_scalar(
    text: str,
    *,
    path: tuple[str, ...],
    required: bool = True,
) -> str | None:
    """Read a scalar at a simple, two-space-indented YAML mapping path."""

    if not path:
        raise ValueError("path cannot be empty")
    lines = text.splitlines()
    start = 0
    end = len(lines)

    for depth, key in enumerate(path[:-1]):
        indent = "  " * depth
        mapping_pattern = re.compile(
            rf"^{re.escape(indent)}{re.escape(key)}:\s*(?:#.*)?$"
        )
        mapping_index = next(
            (
                index
                for index in range(start, end)
                if mapping_pattern.fullmatch(lines[index])
            ),
            None,
        )
        if mapping_index is None:
            if required:
                raise ResumeProfileError(
                    "saved config has no mapping "
                    + ".".join(path[: depth + 1])
                )
            return None

        child_start = mapping_index + 1
        child_end = end
        parent_indent = len(indent)
        for index in range(child_start, end):
            line = lines[index]
            stripped = line.lstrip(" ")
            if not stripped or stripped.startswith("#"):
                continue
            line_indent = len(line) - len(stripped)
            if line_indent <= parent_indent:
                child_end = index
                break
        start, end = child_start, child_end

    scalar_indent = "  " * (len(path) - 1)
    scalar_pattern = re.compile(
        rf"^{re.escape(scalar_indent)}{re.escape(path[-1])}:"
        r"\s*(.*?)\s*(?:#.*)?$"
    )
    for line in lines[start:end]:
        match = scalar_pattern.fullmatch(line)
        if match is None:
            continue
        value = match.group(1).strip()
        if not value:
            raise ResumeProfileError(
                f"saved config field {'.'.join(path)} is not a scalar"
            )
        return value.strip("'\"")
    if required:
        raise ResumeProfileError(
            f"saved config has no scalar {'.'.join(path)}"
        )
    return None


def _section_scalar(
    text: str,
    *,
    section: str,
    key: str,
    required: bool = True,
) -> str | None:
    """Backward-compatible helper for a direct top-level mapping child."""

    return _path_scalar(
        text,
        path=(section, key),
        required=required,
    )


def verify_resume_profile(
    checkpoint: Path,
    *,
    profile: str,
    expected_seq_size: int,
    expected_block_size: int,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ResumeProfileError(
            f"resume checkpoint does not exist: {checkpoint}"
        )
    config_path = checkpoint.parent.parent / "config.yaml"
    if not config_path.is_file():
        raise ResumeProfileError(
            "resolved run config is missing beside the checkpoint: "
            f"{config_path}"
        )
    text = config_path.read_text(encoding="utf-8")
    try:
        saved_seq_size = int(
            _section_scalar(text, section="data", key="pose_length")
        )
        saved_block_size = int(
            _section_scalar(
                text,
                section="train",
                key="num_frame_per_block",
            )
        )
    except ValueError as error:
        raise ResumeProfileError(
            "saved pose_length/num_frame_per_block is not an integer"
        ) from error

    saved_profile = _section_scalar(
        text,
        section="talkshow_profile",
        key="name",
        required=False,
    )
    saved_identity_contract = _section_scalar(
        text,
        section="talkshow_profile",
        key="identity_contract",
        required=False,
    )
    if saved_identity_contract is None:
        raise ResumeProfileError(
            "saved TalkSHOW checkpoint has no identity contract; legacy "
            "global-ID checkpoints are incompatible with "
            f"{IDENTITY_CONTRACT}"
        )
    if saved_identity_contract != IDENTITY_CONTRACT:
        raise ResumeProfileError(
            "saved TalkSHOW checkpoint identity contract mismatch: "
            f"{saved_identity_contract!r} != {IDENTITY_CONTRACT!r}"
        )

    try:
        saved_identity_range = {
            name: int(
                _section_scalar(
                    text,
                    section="talkshow_profile",
                    key=name,
                )
            )
            for name in EXPECTED_IDENTITY_RANGE
        }
        saved_main_num_person = int(
            _path_scalar(
                text,
                path=("model", "model", "params", "num_person"),
            )
        )
        saved_face_speaker_id_offset = int(
            _path_scalar(
                text,
                path=(
                    "model",
                    "speech_model",
                    "params",
                    "speaker_id_offset",
                ),
            )
        )
    except ValueError as error:
        raise ResumeProfileError(
            "saved TalkSHOW identity/model contract contains a non-integer "
            "scalar"
        ) from error

    mismatches = []
    if saved_seq_size != expected_seq_size:
        mismatches.append(
            f"pose_length={saved_seq_size} (requested {expected_seq_size})"
        )
    if saved_block_size != expected_block_size:
        mismatches.append(
            "num_frame_per_block="
            f"{saved_block_size} (requested {expected_block_size})"
        )
    if saved_profile != profile:
        mismatches.append(
            f"profile={saved_profile!r} (requested {profile!r})"
        )
    for name, expected in EXPECTED_IDENTITY_RANGE.items():
        observed = saved_identity_range[name]
        if observed != expected:
            mismatches.append(
                f"{name}={observed} (required {expected})"
            )
    if saved_main_num_person != EXPECTED_MAIN_NUM_PERSON:
        mismatches.append(
            "model.model.params.num_person="
            f"{saved_main_num_person} "
            f"(required {EXPECTED_MAIN_NUM_PERSON})"
        )
    if (
        saved_face_speaker_id_offset
        != EXPECTED_FACE_SPEAKER_ID_OFFSET
    ):
        mismatches.append(
            "model.speech_model.params.speaker_id_offset="
            f"{saved_face_speaker_id_offset} "
            f"(required {EXPECTED_FACE_SPEAKER_ID_OFFSET})"
        )
    if mismatches:
        raise ResumeProfileError(
            "TalkSHOW checkpoint/profile mismatch: " + "; ".join(mismatches)
        )
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--expected-seq-size", type=int, required=True)
    parser.add_argument("--expected-block-size", type=int, required=True)
    args = parser.parse_args()
    try:
        config_path = verify_resume_profile(
            args.checkpoint,
            profile=args.profile,
            expected_seq_size=args.expected_seq_size,
            expected_block_size=args.expected_block_size,
        )
    except ResumeProfileError as error:
        parser.exit(2, f"error: {error}\n")
    print(
        "[ok] TalkSHOW checkpoint profile matches: "
        f"{args.profile} ({config_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
