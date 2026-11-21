#!/usr/bin/env python3
"""Strict offline evaluator matching TalkSHOW's original body/face protocol.

This evaluator consumes two independent PASPA/BLD test outputs from the same
checkpoint.  It intentionally keeps TalkSHOW's original B=2 body sampling,
lower-body replacement, face isolation, SMPL-X joint metrics, FGD and beat
consistency behavior.

Heavy dependencies are imported only after input validation, so ``--help`` and
``--validate-input-only`` work in a lightweight environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


POSE_DIM = 165
EXPRESSION_DIM = 100
TALKSHOW_DIM = 265
BODY_DIM = 129
FACE_DIM = 103
POSE_FPS = 30
AUDIO_SR = 16000
BODY_SAMPLE_COUNT = 2
OFFICIAL_GENERATE_LENGTH = 88
TALKSHOW_SPEAKERS = ("oliver", "seth", "conan", "chemistry")
REFERENCE_COMMIT = "9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
OFFICIAL_SPLIT_SHA256 = (
    "2df6e745cdf7473f13ce3ae2ed759c3cceb60c9197e7f3fd65110e7bc20b6f2d"
)
OFFICIAL_TEST_CLIP_COUNT = 1708
OFFICIAL_SPLIT_MISSING_COUNT = 55

# TalkSHOW data_utils/lower_body.py:c_index_3d at REFERENCE_COMMIT.
# Keep the ranges explicit: they are part of the published evaluation protocol.
BODY_INDEX = np.asarray(
    [
        *range(18, 21),
        *range(27, 30),
        *range(36, 45),
        *range(51, 165),
    ],
    dtype=np.int64,
)

# SHA-256 after CRLF -> LF normalization.  The local reference checkout uses
# CRLF in the working tree, while the Git blobs use LF.
EXPECTED_TALKSHOW_SOURCE_SHA256_LF = {
    "config/body_pixel.json": (
        "c06bd4d6bce5f538318fe6fb050c4a32590990471851aff318cd41c82b4e82f5"
    ),
    "config/face.json": (
        "fc9bee9e4594e2dd5cd3fc02ccd1ce749c759dd0d196e6618f0dd44647313a9f"
    ),
    "scripts/test_body.py": (
        "e460ea5842ddb19c353df3bb9c56a504e6f19d687c9333ddce42baa2a0147774"
    ),
    "scripts/test_face.py": (
        "bc90341b552d9d698b5d7a65055a6949e20d8b9935ee539804f4fff23082b052"
    ),
    "evaluation/FGD.py": (
        "2949b8d5a5e697d792e7e9d1546d46943b92e27db7641da3dc194a9fad0a8d39"
    ),
    "evaluation/metrics.py": (
        "746d44c2304743294ac43922425ee0799ca6e67ec6b25b83056e0d696a52dfea"
    ),
    "data_utils/lower_body.py": (
        "6eae37ef17760b5ebad60bf25ae7149aa5244cba8d9627eafe77b71616931163"
    ),
    "data_utils/get_j.py": (
        "393f5f1f5943c123e165871c98834885cf41fa7f73e0821cee8e161ce53db09a"
    ),
    "data_utils/dataloader_torch.py": (
        "93b9c47dae13ffd4be71f0e44451719e2bd57b691b43292b25cc1470bc439f99"
    ),
    "data_utils/utils.py": (
        "75a8c844a62d962a88d61af745e0231ac6249138b23ebb602b3de000bd2a47f3"
    ),
    "nets/body_ae.py": (
        "a4065c7a086b48dcb7b161202ebe9756dd8d7a4c0df379e16ad68532368fdc4b"
    ),
    "nets/base.py": (
        "e5b1931dfdcba45a068818c5543a75f8dcfdc843cfa733e846483b401a0d19bd"
    ),
    "nets/spg/vqvae_1d.py": (
        "c4f8f2a7862a3835b250403e8533c3387523a057e3b244e73266b6f1034a783f"
    ),
    "nets/spg/vqvae_modules.py": (
        "6ae6e1c5860afc420876942616cbc547aceb245aa97ba4bf94f4ca51080fa012"
    ),
    "nets/spg/s2glayers.py": (
        "ad2f5d56d25f0e75131e3b524e14bdc7c9f0fa22a696539c89563f62c29d68d7"
    ),
    "data_utils/rotation_conversion.py": (
        "4f9cd089f2cdc031e16be435f116c10a5498772d834b4da9a03947041fa09f82"
    ),
}
# The repository's reproducible bootstrap applies a modern-runtime-only guard
# around torchaudio imports.  It does not change get_mfcc_ta or the onset
# branch used here, so this one known hash remains protocol-compatible.
KNOWN_COMPATIBLE_SOURCE_SHA256_LF = {
    "data_utils/utils.py": {
        "75c45a29fb5adee307ebe4e7cdf0bc36091f5b694b4b9551330b0bdeea2a2ce3",
    },
}


class ProtocolError(RuntimeError):
    """Raised when an input cannot satisfy the official protocol."""


@dataclass(frozen=True)
class MotionArrays:
    poses: np.ndarray
    expressions: np.ndarray
    betas: np.ndarray
    fps: int


@dataclass(frozen=True)
class ClipPaths:
    clip_id: str
    prediction_paths: tuple[Path, Path]
    ground_truth_path: Path
    audio_path: Path
    audio_protocol: str


@dataclass(frozen=True)
class InputManifest:
    clips: tuple[ClipPaths, ...]
    source_test_clip_count: int | None
    source_view_clip_count: int | None
    official_short_source_clip_ids: tuple[str, ...]
    missing_source_clip_ids: tuple[str, ...]
    split_view_summary: Mapping[str, Any] | None
    full_split_view_verified: bool


@dataclass(frozen=True)
class InputValidation:
    clips: tuple[ClipPaths, ...]
    total_frames: int
    clip_manifest_sha256: str
    warnings: tuple[str, ...]
    source_test_clip_count: int | None
    source_view_clip_count: int | None
    official_short_source_clip_ids: tuple[str, ...]
    missing_source_clip_ids: tuple[str, ...]
    split_view_summary: Mapping[str, Any] | None
    full_split_view_verified: bool


def beat2_to_talkshow(
    poses_165: np.ndarray,
    expressions_100: np.ndarray,
) -> np.ndarray:
    """Convert PASPA/BEAT2 pose order to TalkSHOW's 265-D parameter order."""

    poses = np.asarray(poses_165)
    expressions = np.asarray(expressions_100)
    if poses.ndim < 2 or poses.shape[-1] != POSE_DIM:
        raise ProtocolError(
            f"poses must end in {POSE_DIM} channels, got {poses.shape}"
        )
    if expressions.ndim != poses.ndim:
        raise ProtocolError(
            "poses and expressions must have the same rank, got "
            f"{poses.shape} and {expressions.shape}"
        )
    if expressions.shape[:-1] != poses.shape[:-1]:
        raise ProtocolError(
            "poses and expressions must have identical leading dimensions, got "
            f"{poses.shape} and {expressions.shape}"
        )
    if expressions.shape[-1] != EXPRESSION_DIM:
        raise ProtocolError(
            f"expressions must end in {EXPRESSION_DIM} channels, "
            f"got {expressions.shape}"
        )
    return np.concatenate(
        (
            poses[..., 66:75],   # jaw, left eye, right eye
            poses[..., 0:66],    # global orientation and body
            poses[..., 75:165],  # hands
            expressions,
        ),
        axis=-1,
    )


def talkshow_to_beat2(talkshow_265: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`beat2_to_talkshow`, primarily for protocol tests."""

    talkshow = np.asarray(talkshow_265)
    if talkshow.ndim < 2 or talkshow.shape[-1] != TALKSHOW_DIM:
        raise ProtocolError(
            f"TalkSHOW parameters must end in {TALKSHOW_DIM} channels, "
            f"got {talkshow.shape}"
        )
    poses = np.empty(
        (*talkshow.shape[:-1], POSE_DIM),
        dtype=talkshow.dtype,
    )
    poses[..., 66:75] = talkshow[..., 0:9]
    poses[..., 0:66] = talkshow[..., 9:75]
    poses[..., 75:165] = talkshow[..., 75:165]
    return poses, talkshow[..., 165:265].copy()


def select_official_body(talkshow_265: np.ndarray) -> np.ndarray:
    """Select the exact 129 pose channels used by TalkSHOW body generation."""

    talkshow = np.asarray(talkshow_265)
    if talkshow.ndim < 2 or talkshow.shape[-1] != TALKSHOW_DIM:
        raise ProtocolError(
            f"TalkSHOW parameters must end in {TALKSHOW_DIM} channels, "
            f"got {talkshow.shape}"
        )
    selected = talkshow[..., :POSE_DIM][..., BODY_INDEX]
    if selected.shape[-1] != BODY_DIM:
        raise AssertionError(
            f"internal body index contract produced {selected.shape[-1]} channels"
        )
    return selected


def make_body_partial_parameters(body_129: np.ndarray) -> np.ndarray:
    """Create the 232-D input expected by TalkSHOW's official ``part2full``."""

    body = np.asarray(body_129)
    if body.ndim < 2 or body.shape[-1] != BODY_DIM:
        raise ProtocolError(
            f"body parameters must end in {BODY_DIM} channels, got {body.shape}"
        )
    zeros_jaw = np.zeros((*body.shape[:-1], 3), dtype=body.dtype)
    zeros_expression = np.zeros(
        (*body.shape[:-1], EXPRESSION_DIM),
        dtype=body.dtype,
    )
    return np.concatenate((zeros_jaw, body, zeros_expression), axis=-1)


def isolate_face_parameters(talkshow_265: np.ndarray) -> np.ndarray:
    """Keep only jaw and expression, matching TalkSHOW ``test_face.py``."""

    talkshow = np.asarray(talkshow_265)
    if talkshow.ndim < 2 or talkshow.shape[-1] != TALKSHOW_DIM:
        raise ProtocolError(
            f"TalkSHOW parameters must end in {TALKSHOW_DIM} channels, "
            f"got {talkshow.shape}"
        )
    zeros_pose = np.zeros((*talkshow.shape[:-1], 162), dtype=talkshow.dtype)
    return np.concatenate(
        (
            talkshow[..., 0:3],
            zeros_pose,
            talkshow[..., 165:265],
        ),
        axis=-1,
    )


def average_clip_metrics(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Average metric dictionaries per clip, with no frame weighting."""

    if not rows:
        raise ProtocolError("cannot average an empty metric list")
    keys = tuple(rows[0].keys())
    expected = set(keys)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ProtocolError(
                f"metric row {index} has keys {sorted(row)}, "
                f"expected {sorted(expected)}"
            )
    averaged = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in keys
    }
    for key, value in averaged.items():
        if not np.isfinite(value):
            raise ProtocolError(f"metric {key} is not finite: {value}")
    return averaged


def _sha256_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@contextmanager
def _working_directory(path: Path):
    """Temporarily satisfy TalkSHOW modules that open repo-relative assets."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def verify_talkshow_sources(
    talkshow_root: Path,
    *,
    allow_unpinned: bool,
) -> dict[str, Any]:
    """Verify the source files that define the official metric protocol."""

    root = talkshow_root.expanduser().resolve()
    if not root.is_dir():
        raise ProtocolError(f"TalkSHOW root is not a directory: {root}")

    files: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    compatibility_files: list[str] = []
    for relative, expected in EXPECTED_TALKSHOW_SOURCE_SHA256_LF.items():
        path = root / relative
        observed = _sha256_lf(path) if path.is_file() else None
        known_compatible = sorted(
            KNOWN_COMPATIBLE_SOURCE_SHA256_LF.get(relative, set())
        )
        files[relative] = {
            "expected_sha256_lf": expected,
            "observed_sha256_lf": observed,
            "known_compatible_sha256_lf": known_compatible,
        }
        if observed == expected:
            continue
        if observed in known_compatible:
            compatibility_files.append(relative)
        else:
            mismatches.append(relative)

    if mismatches and not allow_unpinned:
        detail = "\n".join(
            "  "
            + relative
            + ": expected="
            + str(files[relative]["expected_sha256_lf"])
            + ", observed="
            + str(files[relative]["observed_sha256_lf"])
            for relative in mismatches
        )
        raise ProtocolError(
            "TalkSHOW metric sources differ from the pinned official "
            f"reference {REFERENCE_COMMIT}:\n{detail}\n"
            "Use the pinned checkout, or pass --allow-unpinned-talkshow only "
            "after reviewing the differences."
        )

    return {
        "root": str(root),
        "reference_commit": REFERENCE_COMMIT,
        "observed_git_head": _git_head(root),
        "source_hashes_verified": not mismatches,
        "allow_unpinned": bool(allow_unpinned),
        "mismatched_files": mismatches,
        "known_compatibility_files": compatibility_files,
        "files": files,
    }


def _discover_files(directory: Path, prefix: str, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ProtocolError(f"input directory does not exist: {directory}")
    discovered: dict[str, Path] = {}
    for path in sorted(directory.glob(f"{prefix}*{suffix}")):
        if not path.is_file():
            continue
        clip_id = path.name[len(prefix) : -len(suffix)]
        if not clip_id:
            raise ProtocolError(f"empty clip id in {path}")
        if clip_id in discovered:
            raise ProtocolError(
                f"duplicate clip id {clip_id!r} in {directory}"
            )
        discovered[clip_id] = path.resolve()
    return discovered


def _require_matching_set(
    *,
    label: str,
    observed: Iterable[str],
    expected: set[str],
) -> None:
    observed_set = set(observed)
    if observed_set == expected:
        return
    missing = sorted(expected - observed_set)
    extra = sorted(observed_set - expected)
    raise ProtocolError(
        f"{label} clip set mismatch: missing={missing[:20]}, extra={extra[:20]}"
    )


def _read_source_motion_frame_count(path: Path) -> int:
    """Read the frame count used by TalkSHOW's test-dataset length filter."""

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ProtocolError(
            f"cannot read TalkSHOW source motion {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or "body_pose_axis" not in payload:
        raise ProtocolError(
            f"{path}: TalkSHOW source motion has no body_pose_axis"
        )
    body_pose = np.asarray(payload["body_pose_axis"])
    if body_pose.ndim < 1 or body_pose.shape[0] < 1:
        raise ProtocolError(
            f"{path}: body_pose_axis has invalid shape {body_pose.shape}"
        )
    return int(body_pose.shape[0])


def _load_split_view_summary(
    source_root: Path,
    *,
    observed_test_clips: int,
) -> tuple[Mapping[str, Any] | None, bool]:
    path = source_root / "split_view_summary.json"
    if not path.is_file():
        return None, False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read split-view summary {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ProtocolError(f"split-view summary must be an object: {path}")
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise ProtocolError(f"split-view summary has invalid counts: {path}")
    test_count = counts.get("test")
    if (
        isinstance(test_count, bool)
        or not isinstance(test_count, int)
        or test_count < 0
    ):
        raise ProtocolError(
            f"split-view summary has invalid test count: {test_count!r}"
        )
    if test_count != observed_test_clips:
        raise ProtocolError(
            "split-view summary/test directory mismatch: "
            f"summary={test_count}, observed={observed_test_clips}"
        )
    full_verified = (
        summary.get("split_sha256") == OFFICIAL_SPLIT_SHA256
        and summary.get("limit_per_speaker_split") is None
        and summary.get("missing_count") == OFFICIAL_SPLIT_MISSING_COUNT
        and test_count == OFFICIAL_TEST_CLIP_COUNT
    )
    report_summary = {
        "path": str(path.resolve()),
        "split_sha256": summary.get("split_sha256"),
        "counts": counts,
        "missing_count": summary.get("missing_count"),
        "limit_per_speaker_split": summary.get(
            "limit_per_speaker_split"
        ),
    }
    return report_summary, full_verified


def build_input_manifest(
    prediction_dirs: Sequence[Path | str],
    ground_truth_dir: Path | str,
    audio_dir: Path | str | None = None,
    source_audio_root: Path | str | None = None,
    *,
    allow_source_subset: bool = False,
) -> InputManifest:
    """Pair exactly two raw prediction sets with one canonical GT/audio set."""

    if (audio_dir is None) == (source_audio_root is None):
        raise ProtocolError(
            "pass exactly one of --source-audio-root (official source WAVs) "
            "or --audio-dir (aligned PASPA output WAVs)"
        )
    if allow_source_subset and source_audio_root is None:
        raise ProtocolError(
            "--allow-source-subset requires --source-audio-root"
        )
    if len(prediction_dirs) != BODY_SAMPLE_COUNT:
        raise ProtocolError(
            "official TalkSHOW body evaluation requires exactly two ordered "
            f"--pred-dir arguments, got {len(prediction_dirs)}"
        )
    pred_dirs = tuple(
        Path(directory).expanduser().resolve()
        for directory in prediction_dirs
    )
    if pred_dirs[0] == pred_dirs[1]:
        raise ProtocolError("the two --pred-dir arguments resolve to one directory")

    gt_dir = Path(ground_truth_dir).expanduser().resolve()
    predictions = tuple(
        _discover_files(directory, "res_", ".npz")
        for directory in pred_dirs
    )
    if not predictions[0]:
        suspicious = sorted(pred_dirs[0].glob("*res_*.npz"))
        hint = (
            f"; found non-canonical files such as {suspicious[0].name!r}"
            if suspicious
            else ""
        )
        raise ProtocolError(
            f"no canonical res_*.npz files in {pred_dirs[0]}{hint}"
        )

    clip_ids = set(predictions[0])
    _require_matching_set(
        label=str(pred_dirs[1]),
        observed=predictions[1],
        expected=clip_ids,
    )

    ground_truth = _discover_files(gt_dir, "gt_", ".npz")
    _require_matching_set(
        label=str(gt_dir),
        observed=ground_truth,
        expected=clip_ids,
    )

    if source_audio_root is not None:
        source_root = Path(source_audio_root).expanduser().resolve()
        if not source_root.is_dir():
            raise ProtocolError(
                f"source audio root does not exist: {source_root}"
            )
        audio: dict[str, Path] = {}
        ordered_clip_ids: list[str] = []
        all_source_clip_ids: list[str] = []
        all_source_clip_id_set: set[str] = set()
        all_source_view_clip_ids: set[str] = set()
        official_short_source_clip_ids: list[str] = []
        # Match official dataloader_torch.py: speakers use test_body.sh order;
        # videos and sequences retain os.listdir order on this exact split
        # view.  This matters because upstream feat_dist has order-dependent
        # batch-major/truncation behavior.
        for speaker in TALKSHOW_SPEAKERS:
            speaker_root = source_root / speaker
            if not speaker_root.is_dir():
                continue
            for video_name in os.listdir(speaker_root):
                split_root = speaker_root / video_name / "test"
                if not split_root.is_dir():
                    continue
                for sequence in os.listdir(split_root):
                    clip_id = f"{speaker}__{sequence}"
                    sequence_root = split_root / sequence
                    candidate = (
                        sequence_root / f"{sequence}.wav"
                    ).resolve()
                    motion = sequence_root / f"{sequence}.pkl"
                    # Match official dataloader_torch.py: incomplete entries
                    # never enter the test DataLoader.
                    if not candidate.is_file() or not motion.is_file():
                        continue
                    if clip_id in all_source_view_clip_ids:
                        raise ProtocolError(
                            f"{clip_id}: source sequence is duplicated across "
                            "TalkSHOW video directories"
                        )
                    all_source_view_clip_ids.add(clip_id)
                    motion_frames = _read_source_motion_frame_count(motion)
                    # MultiVidData.get_dataset() applies this filter before
                    # constructing the official test DataLoader.  Both the
                    # released body and face configs use generate_length=88.
                    if motion_frames < OFFICIAL_GENERATE_LENGTH:
                        official_short_source_clip_ids.append(clip_id)
                        continue
                    all_source_clip_ids.append(clip_id)
                    all_source_clip_id_set.add(clip_id)
                    if clip_id not in clip_ids:
                        continue
                    audio[clip_id] = candidate
                    ordered_clip_ids.append(clip_id)
        unknown_predictions = sorted(clip_ids - all_source_clip_id_set)
        missing_predictions = tuple(
            clip_id
            for clip_id in all_source_clip_ids
            if clip_id not in clip_ids
        )
        if unknown_predictions:
            raise ProtocolError(
                "predictions contain clips outside the TalkSHOW official "
                "test DataLoader (including clips shorter than 88 frames): "
                f"{unknown_predictions[:20]}"
            )
        if missing_predictions and not allow_source_subset:
            raise ProtocolError(
                "predictions do not cover the complete TalkSHOW test split; "
                f"missing {len(missing_predictions)} clips, first="
                f"{list(missing_predictions[:20])}. If these clips were "
                "explicitly excluded by an audited ETL policy, rerun with "
                "--allow-source-subset; the report will be marked as a subset."
            )
        split_view_summary, full_split_view_verified = (
            _load_split_view_summary(
                source_root,
                observed_test_clips=len(all_source_view_clip_ids),
            )
        )
        source_test_clip_count: int | None = len(all_source_clip_ids)
        source_view_clip_count: int | None = len(all_source_view_clip_ids)
        audio_protocol = "talkshow_original_source"
    else:
        wav_dir = Path(audio_dir).expanduser().resolve()
        audio = _discover_files(wav_dir, "gt_sr16000_", ".wav")
        _require_matching_set(
            label=str(wav_dir),
            observed=audio,
            expected=clip_ids,
        )
        audio_protocol = "paspa_aligned_output"
        ordered_clip_ids = sorted(clip_ids)
        source_test_clip_count = None
        source_view_clip_count = None
        official_short_source_clip_ids = []
        missing_predictions = ()
        split_view_summary = None
        full_split_view_verified = False

    return InputManifest(
        clips=tuple(
            ClipPaths(
                clip_id=clip_id,
                prediction_paths=(
                    predictions[0][clip_id],
                    predictions[1][clip_id],
                ),
                ground_truth_path=ground_truth[clip_id],
                audio_path=audio[clip_id],
                audio_protocol=audio_protocol,
            )
            for clip_id in ordered_clip_ids
        ),
        source_test_clip_count=source_test_clip_count,
        source_view_clip_count=source_view_clip_count,
        official_short_source_clip_ids=tuple(
            official_short_source_clip_ids
        ),
        missing_source_clip_ids=missing_predictions,
        split_view_summary=split_view_summary,
        full_split_view_verified=full_split_view_verified,
    )


def _load_raw_full_npz(path: Path, *, role: str) -> MotionArrays:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"poses", "expressions", "betas", "mocap_frame_rate"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ProtocolError(f"{role} {path} is missing keys: {missing}")
            poses = np.asarray(archive["poses"])
            expressions = np.asarray(archive["expressions"])
            betas = np.asarray(archive["betas"]).reshape(-1)
            fps_value = np.asarray(archive["mocap_frame_rate"]).reshape(-1)
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot read {role} NPZ {path}: {exc}") from exc

    if poses.ndim != 2 or poses.shape[1] != POSE_DIM:
        raise ProtocolError(
            f"{role} {path}: poses must be [T,{POSE_DIM}], got {poses.shape}"
        )
    if expressions.ndim != 2 or expressions.shape != (
        poses.shape[0],
        EXPRESSION_DIM,
    ):
        raise ProtocolError(
            f"{role} {path}: expressions must be "
            f"[{poses.shape[0]},{EXPRESSION_DIM}], got {expressions.shape}"
        )
    if poses.shape[0] < OFFICIAL_GENERATE_LENGTH:
        raise ProtocolError(
            f"{role} {path}: TalkSHOW's official test DataLoader excludes "
            f"clips shorter than {OFFICIAL_GENERATE_LENGTH} frames, "
            f"got {poses.shape[0]}"
        )
    if betas.shape != (300,):
        raise ProtocolError(
            f"{role} {path}: betas must contain exactly 300 values, "
            f"got {betas.shape}"
        )
    if fps_value.size != 1:
        raise ProtocolError(
            f"{role} {path}: mocap_frame_rate must be scalar, "
            f"got {fps_value.shape}"
        )
    try:
        fps_float = float(fps_value[0])
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            f"{role} {path}: invalid mocap_frame_rate {fps_value!r}"
        ) from exc
    if fps_float != float(POSE_FPS):
        raise ProtocolError(
            f"{role} {path}: mocap_frame_rate must be {POSE_FPS}, "
            f"got {fps_float}"
        )

    for name, array in (
        ("poses", poses),
        ("expressions", expressions),
        ("betas", betas),
    ):
        if not np.issubdtype(array.dtype, np.floating):
            raise ProtocolError(
                f"{role} {path}: {name} must be floating point, "
                f"got {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ProtocolError(
                f"{role} {path}: {name} contains NaN or Inf"
            )

    return MotionArrays(
        poses=poses.astype(np.float32, copy=False),
        expressions=expressions.astype(np.float32, copy=False),
        betas=betas.astype(np.float64, copy=False),
        fps=POSE_FPS,
    )


def _validate_wav(
    path: Path,
    *,
    expected_frames: int,
    audio_protocol: str,
) -> None:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_count = handle.getnframes()
            compression = handle.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise ProtocolError(f"cannot read WAV {path}: {exc}") from exc

    if channels < 1 or sample_rate < 1 or sample_count < 1:
        raise ProtocolError(
            f"{path}: invalid WAV header (channels={channels}, "
            f"sample_rate={sample_rate}, samples={sample_count})"
        )
    if compression != "NONE":
        raise ProtocolError(
            f"{path}: audio must be uncompressed PCM WAV, got {compression}"
        )
    if audio_protocol == "paspa_aligned_output":
        if channels != 1:
            raise ProtocolError(
                f"{path}: aligned PASPA audio must be mono, got "
                f"{channels} channels"
            )
        if sample_rate != AUDIO_SR:
            raise ProtocolError(
                f"{path}: aligned PASPA audio sample rate must be {AUDIO_SR}, "
                f"got {sample_rate}"
            )
        expected_samples = AUDIO_SR * expected_frames // POSE_FPS
        if sample_count != expected_samples:
            raise ProtocolError(
                f"{path}: aligned PASPA audio has {sample_count} samples, "
                f"expected exactly {expected_samples} for {expected_frames} "
                f"frames at {POSE_FPS} FPS"
            )
    elif audio_protocol != "talkshow_original_source":
        raise AssertionError(f"unknown audio protocol: {audio_protocol}")


def _assert_same_array(
    left: np.ndarray,
    right: np.ndarray,
    *,
    label: str,
) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ProtocolError(f"{label} is not exactly identical")


def validate_input_manifest(manifest: InputManifest) -> InputValidation:
    """Validate all raw NPZ/WAV inputs before allocating a GPU."""

    clips = manifest.clips
    if not clips:
        raise ProtocolError("the input manifest is empty")

    total_frames = 0
    all_body_samples_identical = True
    all_face_samples_identical = True
    any_nonzero_gt_expression = False
    manifest_digest = hashlib.sha256()

    for clip_index, clip in enumerate(clips, start=1):
        if len(clips) >= 100 and (
            clip_index == 1
            or clip_index % 100 == 0
            or clip_index == len(clips)
        ):
            print(
                f"[validate] {clip_index}/{len(clips)} {clip.clip_id}",
                file=sys.stderr,
                flush=True,
            )
        gt = _load_raw_full_npz(
            clip.ground_truth_path,
            role=f"ground truth {clip.clip_id}",
        )
        predictions = tuple(
            _load_raw_full_npz(
                path,
                role=f"prediction sample{sample_index} {clip.clip_id}",
            )
            for sample_index, path in enumerate(clip.prediction_paths)
        )
        for sample_index, prediction in enumerate(predictions):
            if prediction.poses.shape[0] != gt.poses.shape[0]:
                raise ProtocolError(
                    f"{clip.clip_id}: sample{sample_index} has "
                    f"{prediction.poses.shape[0]} frames but GT has "
                    f"{gt.poses.shape[0]}; this strict adapter requires one "
                    "complete frame set for both body and face"
                )
            _assert_same_array(
                prediction.betas,
                gt.betas,
                label=f"{clip.clip_id}: sample{sample_index} betas versus GT",
            )

        gt_talkshow = beat2_to_talkshow(gt.poses, gt.expressions)
        pred_talkshow = tuple(
            beat2_to_talkshow(pred.poses, pred.expressions)
            for pred in predictions
        )
        if not np.array_equal(
            select_official_body(pred_talkshow[0]),
            select_official_body(pred_talkshow[1]),
        ):
            all_body_samples_identical = False
        if not np.array_equal(
            isolate_face_parameters(pred_talkshow[0]),
            isolate_face_parameters(pred_talkshow[1]),
        ):
            all_face_samples_identical = False
        if np.any(gt_talkshow[:, 165:265] != 0):
            any_nonzero_gt_expression = True

        # If either prediction directory also contains canonical GT files,
        # verify that they agree with --gt-dir instead of silently mixing runs.
        for sample_index, prediction_path in enumerate(clip.prediction_paths):
            local_gt = prediction_path.parent / f"gt_{clip.clip_id}.npz"
            if not local_gt.is_file() or local_gt == clip.ground_truth_path:
                continue
            other_gt = _load_raw_full_npz(
                local_gt,
                role=f"sample{sample_index} local GT {clip.clip_id}",
            )
            _assert_same_array(
                other_gt.poses,
                gt.poses,
                label=f"{clip.clip_id}: sample{sample_index} local GT poses",
            )
            _assert_same_array(
                other_gt.expressions,
                gt.expressions,
                label=(
                    f"{clip.clip_id}: sample{sample_index} local GT expressions"
                ),
            )
            _assert_same_array(
                other_gt.betas,
                gt.betas,
                label=f"{clip.clip_id}: sample{sample_index} local GT betas",
            )

        _validate_wav(
            clip.audio_path,
            expected_frames=gt.poses.shape[0],
            audio_protocol=clip.audio_protocol,
        )
        total_frames += gt.poses.shape[0]
        manifest_digest.update(clip.clip_id.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(str(gt.poses.shape[0]).encode("ascii"))
        manifest_digest.update(b"\n")

    if not any_nonzero_gt_expression:
        raise ProtocolError(
            "all GT expressions are zero. The official combined evaluator "
            "requires raw full GT; do not use legacy body-isolated "
            "preprocessed output"
        )

    warnings: list[str] = []
    audio_protocols = {clip.audio_protocol for clip in clips}
    if len(audio_protocols) != 1:
        raise AssertionError(f"mixed audio protocols: {audio_protocols}")
    audio_protocol = next(iter(audio_protocols))
    if all_body_samples_identical:
        warnings.append(
            "all official 129-D body channels are identical across sample0 "
            "and sample1; diversity will be zero. Confirm that the two test "
            "runs used independent random seeds."
        )
    if not all_face_samples_identical:
        warnings.append(
            "sample0 and sample1 have different jaw/expression predictions. "
            "The official face protocol evaluates sample0 only."
        )
    if audio_protocol == "paspa_aligned_output":
        warnings.append(
            "using cropped/padded PASPA output WAVs instead of original "
            "TalkSHOW source WAVs; body/face and the order-invariant FGD "
            "distance remain protocol-aligned, but BC can differ when source "
            "duration differs and order-dependent feat_dist is not guaranteed "
            "to match the official DataLoader order."
        )
    if manifest.official_short_source_clip_ids:
        warnings.append(
            "excluded "
            f"{len(manifest.official_short_source_clip_ids)} source clips "
            f"shorter than {OFFICIAL_GENERATE_LENGTH} frames, matching "
            "TalkSHOW MultiVidData.get_dataset()."
        )
    if manifest.missing_source_clip_ids:
        warnings.append(
            f"evaluating an explicit TalkSHOW test subset: "
            f"{len(manifest.missing_source_clip_ids)} source clips are absent."
        )

    return InputValidation(
        clips=tuple(clips),
        total_frames=total_frames,
        clip_manifest_sha256=manifest_digest.hexdigest(),
        warnings=tuple(warnings),
        source_test_clip_count=manifest.source_test_clip_count,
        source_view_clip_count=manifest.source_view_clip_count,
        official_short_source_clip_ids=(
            manifest.official_short_source_clip_ids
        ),
        missing_source_clip_ids=manifest.missing_source_clip_ids,
        split_view_summary=manifest.split_view_summary,
        full_split_view_verified=manifest.full_split_view_verified,
    )


def _make_stub_config() -> SimpleNamespace:
    """Minimal official config needed by nets.body_ae.TrainWrapper."""

    return SimpleNamespace(
        Data=SimpleNamespace(
            pose=SimpleNamespace(
                convert_to_6d=False,
                pre_pose_length=0,
                expression=False,
            )
        ),
        Train=SimpleNamespace(
            learning_rate=SimpleNamespace(
                generator_learning_rate=1e-4,
                discriminator_learning_rate=1e-4,
            )
        ),
    )


def _resolve_smplx_asset(smplx_path: Path) -> Path:
    if smplx_path.is_file():
        return smplx_path
    candidate = smplx_path / "smplx" / "SMPLX_NEUTRAL.npz"
    if candidate.is_file():
        return candidate
    raise ProtocolError(
        f"cannot find the SMPL-X asset loaded by smplx.create under "
        f"{smplx_path}; expected {candidate}, or pass the neutral NPZ "
        "itself to --smplx-path"
    )


def _load_feature_extractor(
    *,
    torch: Any,
    wrapper_class: Any,
    checkpoint_path: Path,
    device: Any,
) -> Any:
    wrapper = wrapper_class(
        SimpleNamespace(gpu=str(device)),
        _make_stub_config(),
    )
    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    inner = (
        checkpoint["generator"]
        if isinstance(checkpoint, Mapping) and "generator" in checkpoint
        else checkpoint
    )
    if isinstance(inner, Mapping) and "g" in inner:
        wrapper.load_state_dict(inner)
    elif isinstance(inner, Mapping):
        wrapper.g.load_state_dict(inner)
    else:
        raise ProtocolError(
            f"unsupported feature extractor checkpoint structure: "
            f"{checkpoint_path}"
        )
    wrapper.g.eval()
    for parameter in wrapper.g.parameters():
        parameter.requires_grad_(False)
    return wrapper


def _body_loss(lvd: Any, gt: Any, prediction: Any) -> dict[str, Any]:
    """Literal metric formula from TalkSHOW scripts/test_body.py."""

    velocity_difference = lvd(
        gt[:, :22, :],
        prediction[:, :, :22, :],
        symmetrical=False,
        weight=False,
    )
    error = (gt - prediction).norm(p=2, dim=-1).sum(dim=-1).mean()
    diversity = (
        prediction.var(dim=0).norm(p=2, dim=-1).sum(dim=-1).mean()
    )
    return {
        "LVD": velocity_difference,
        "error": error,
        "diverse": diversity,
    }


def _face_loss(
    torch: Any,
    lvd: Any,
    gt: Any,
    prediction: Any,
) -> dict[str, Any]:
    """Literal metric formula from TalkSHOW scripts/test_face.py."""

    jaw_difference = gt[:, 22:25, :] - prediction[:, 22:25, :]
    jaw_l1 = jaw_difference.norm(p=2, dim=-1).sum(dim=-1).mean()

    landmark_difference = gt[:, 74:] - prediction[:, 74:]
    landmark_l1 = (
        landmark_difference.norm(p=2, dim=-1).sum(dim=-1).mean()
    )

    face_gt = torch.cat((gt[:, 22:25], gt[:, 74:]), dim=1)
    face_prediction = torch.cat(
        (prediction[:, 22:25], prediction[:, 74:]),
        dim=1,
    )
    return {
        "jaw_l1": jaw_l1,
        "landmark_l1": landmark_l1,
        "LVD": lvd(
            face_gt,
            face_prediction,
            symmetrical=False,
            weight=False,
        ),
    }


def _finite_scalar(value: Any, *, label: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not np.isfinite(result):
        raise ProtocolError(f"official metric {label} is not finite: {result}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pred-dir",
        "--prediction-dir",
        dest="prediction_dirs",
        action="append",
        required=True,
        help=(
            "One raw PASPA/BLD npz/test directory. Pass exactly twice, in "
            "sample0 then sample1 order."
        ),
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Canonical raw PASPA/BLD npz/test directory containing gt_*.npz.",
    )
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        "--source-audio-root",
        help=(
            "TalkSHOW split-view root. Resolves original test WAVs as "
            "<speaker>/*/test/<sequence>/<sequence>.wav; this is the closest "
            "match to the official evaluator."
        ),
    )
    audio_group.add_argument(
        "--audio-dir",
        help=(
            "Canonical PASPA/BLD wav/test directory containing aligned "
            "gt_sr16000_*.wav. Use when original source WAVs are unavailable."
        ),
    )
    parser.add_argument(
        "--talkshow-root",
        help=(
            "TalkSHOW repository root. Required for metric execution; optional "
            "for input-only validation."
        ),
    )
    parser.add_argument(
        "--feature-extractor",
        help=(
            "Official TalkSHOW experiments/feature_extractor.pth. Required "
            "for metric execution."
        ),
    )
    parser.add_argument(
        "--smplx-path",
        help=(
            "Path passed to smplx.create; defaults to "
            "<talkshow-root>/visualise."
        ),
    )
    parser.add_argument(
        "--expected-feature-extractor-sha256",
        help="Optional expected SHA-256 for feature_extractor.pth.",
    )
    parser.add_argument(
        "--expected-smplx-sha256",
        help="Optional expected SHA-256 for the loaded neutral SMPL-X asset.",
    )
    parser.add_argument(
        "--allow-unpinned-talkshow",
        action="store_true",
        help=(
            "Allow TalkSHOW metric source hashes to differ from the pinned "
            "official reference. The mismatch remains recorded in JSON."
        ),
    )
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="Validate NPZ/WAV pairing and shapes without importing torch.",
    )
    parser.add_argument(
        "--allow-source-subset",
        action="store_true",
        help=(
            "Allow predictions to omit clips present in --source-audio-root. "
            "Missing IDs are recorded and the report is explicitly a subset."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for SMPL-X and feature extraction (default: cuda:0).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON. Required unless only printing input validation.",
    )
    return parser


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _generation_run_provenance(prediction_dir: Path | str) -> dict[str, Any]:
    directory = Path(prediction_dir).expanduser().resolve()
    work_dir = (
        directory.parents[1]
        if directory.name == "test"
        and directory.parent.name == "npz"
        and len(directory.parents) >= 2
        else None
    )
    candidates = (
        (
            ("config.yaml", work_dir / "config.yaml"),
            ("hydra_config.yaml", work_dir / ".hydra" / "config.yaml"),
        )
        if work_dir is not None
        else ()
    )
    config_files: dict[str, Any] = {}
    for label, path in candidates:
        config_files[label] = (
            {"path": str(path), "sha256": _sha256_file(path)}
            if path.is_file()
            else None
        )
    return {
        "prediction_dir": str(directory),
        "inferred_work_dir": str(work_dir) if work_dir is not None else None,
        "config_files": config_files,
    }


def _base_report(
    *,
    args: argparse.Namespace,
    validation: InputValidation,
) -> dict[str, Any]:
    return {
        "protocol": {
            "name": "talkshow_official_body_face",
            "version": 1,
            "input_protocol": "paspa_bld_raw_full_npz_v1",
            "reference_commit": REFERENCE_COMMIT,
            "body_samples_per_clip": BODY_SAMPLE_COUNT,
            "face_sample": 0,
            "bc_sample": 0,
            "pose_fps": POSE_FPS,
            "audio_sample_rate": AUDIO_SR,
            "minimum_source_frames": OFFICIAL_GENERATE_LENGTH,
            "per_clip_equal_weighting": True,
            "translation_ignored": True,
            "feat_dist_preserves_upstream_order_dependence": True,
            "official_clip_order": (
                validation.clips[0].audio_protocol
                == "talkshow_original_source"
            ),
            "official_source_audio": (
                validation.clips[0].audio_protocol
                == "talkshow_original_source"
            ),
            "formula_and_preprocessing_aligned": True,
            "generation_provenance_verified": False,
            "generation_provenance_note": (
                "NPZ files do not embed checkpoint/config/seed provenance; "
                "audit the two test-run commands and logs externally."
            ),
        },
        "counts": {
            "clips": len(validation.clips),
            "total_frames": validation.total_frames,
            "source_test_clips": validation.source_test_clip_count,
            "source_view_clips": validation.source_view_clip_count,
            "official_short_source_clips": len(
                validation.official_short_source_clip_ids
            ),
            "missing_source_clips": len(
                validation.missing_source_clip_ids
            ),
        },
        "inputs": {
            "prediction_dirs": [
                str(Path(path).expanduser().resolve())
                for path in args.prediction_dirs
            ],
            "generation_runs": [
                _generation_run_provenance(path)
                for path in args.prediction_dirs
            ],
            "ground_truth_dir": str(
                Path(args.gt_dir).expanduser().resolve()
            ),
            "audio_dir": (
                str(Path(args.audio_dir).expanduser().resolve())
                if args.audio_dir
                else None
            ),
            "source_audio_root": (
                str(Path(args.source_audio_root).expanduser().resolve())
                if args.source_audio_root
                else None
            ),
            "audio_protocol": validation.clips[0].audio_protocol,
            "clip_order": (
                "talkshow_test_body_os_listdir_order"
                if validation.clips[0].audio_protocol
                == "talkshow_original_source"
                else "lexicographic_clip_id"
            ),
            "clip_manifest_sha256": validation.clip_manifest_sha256,
            "official_short_source_clip_ids": list(
                validation.official_short_source_clip_ids
            ),
            "missing_source_clip_ids": list(
                validation.missing_source_clip_ids
            ),
            "full_split_view_verified": (
                validation.full_split_view_verified
            ),
            "split_view_summary": validation.split_view_summary,
        },
        "warnings": list(validation.warnings),
    }


def run_official_evaluation(
    *,
    args: argparse.Namespace,
    validation: InputValidation,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the heavy official protocol after all lightweight checks pass."""

    talkshow_root = Path(args.talkshow_root).expanduser().resolve()
    feature_extractor = Path(args.feature_extractor).expanduser().resolve()
    if not feature_extractor.is_file():
        raise ProtocolError(
            f"feature extractor checkpoint does not exist: {feature_extractor}"
        )
    feature_sha256 = _sha256_file(feature_extractor)
    if (
        args.expected_feature_extractor_sha256
        and feature_sha256.lower()
        != args.expected_feature_extractor_sha256.lower()
    ):
        raise ProtocolError(
            "feature extractor SHA-256 mismatch: expected "
            f"{args.expected_feature_extractor_sha256}, got {feature_sha256}"
        )

    smplx_path = (
        Path(args.smplx_path).expanduser().resolve()
        if args.smplx_path
        else talkshow_root / "visualise"
    )
    smplx_asset = _resolve_smplx_asset(smplx_path)
    smplx_asset_sha256 = _sha256_file(smplx_asset)
    if (
        args.expected_smplx_sha256
        and smplx_asset_sha256.lower()
        != args.expected_smplx_sha256.lower()
    ):
        raise ProtocolError(
            "SMPL-X neutral asset SHA-256 mismatch: expected "
            f"{args.expected_smplx_sha256}, got {smplx_asset_sha256}"
        )

    if str(talkshow_root) not in sys.path:
        sys.path.insert(0, str(talkshow_root))
    try:
        # TalkSHOW data_utils/mesh_dataset.py opens
        # "data_utils/hand_component.json" during package import.
        with _working_directory(talkshow_root):
            import smplx as smpl
            import torch

            from data_utils.get_j import get_joints
            from data_utils.lower_body import c_index_3d, part2full, poses2pred
            from data_utils.utils import get_mfcc_ta
            from evaluation.FGD import EmbeddingSpaceEvaluator
            from evaluation.metrics import LVD
            from nets.body_ae import TrainWrapper as BodyAutoencoder
    except (ImportError, FileNotFoundError) as exc:
        raise ProtocolError(
            "cannot import the TalkSHOW evaluation environment. Install the "
            f"official dependencies and retry: {exc}"
        ) from exc

    if not np.array_equal(np.asarray(c_index_3d), BODY_INDEX):
        raise ProtocolError(
            "imported TalkSHOW c_index_3d differs from the pinned 129-D "
            "official body index"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ProtocolError(
                f"requested {device}, but torch.cuda.is_available() is false"
            )
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")

    smplx_model = smpl.create(
        model_path=str(smplx_path),
        model_type="smplx",
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        num_betas=300,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        use_pca=False,
        flat_hand_mean=False,
        create_expression=True,
        num_expression_coeffs=100,
        num_pca_comps=12,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=torch.float64,
    ).to(device)

    autoencoder = _load_feature_extractor(
        torch=torch,
        wrapper_class=BodyAutoencoder,
        checkpoint_path=feature_extractor,
        device=device,
    )
    fgd_handler = EmbeddingSpaceEvaluator(
        autoencoder,
        None,
        str(device),
    )

    body_rows: list[dict[str, float]] = []
    face_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for clip_number, clip in enumerate(validation.clips, start=1):
            gt = _load_raw_full_npz(
                clip.ground_truth_path,
                role=f"ground truth {clip.clip_id}",
            )
            predictions = tuple(
                _load_raw_full_npz(
                    path,
                    role=f"prediction sample{index} {clip.clip_id}",
                )
                for index, path in enumerate(clip.prediction_paths)
            )
            gt_talkshow_np = beat2_to_talkshow(
                gt.poses,
                gt.expressions,
            ).astype(np.float32, copy=False)
            pred_talkshow_np = np.stack(
                [
                    beat2_to_talkshow(
                        prediction.poses,
                        prediction.expressions,
                    )
                    for prediction in predictions
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            pred_body_np = select_official_body(pred_talkshow_np)

            gt_talkshow = torch.from_numpy(gt_talkshow_np).to(device)
            pred_body = torch.from_numpy(pred_body_np).to(device)
            betas = torch.from_numpy(gt.betas).to(
                device=device,
                dtype=torch.float64,
            )

            # TalkSHOW test_body.py pushes B=2 generated 129-D parameters and
            # one raw 265-D GT sequence into the official body autoencoder.
            fgd_handler.push_samples(
                pred_body,
                gt_talkshow.unsqueeze(0),
            )

            body_partial = torch.from_numpy(
                make_body_partial_parameters(pred_body_np)
            ).to(device)
            full_body_prediction = torch.stack(
                [
                    part2full(body_partial[sample_index])
                    for sample_index in range(BODY_SAMPLE_COUNT)
                ],
                dim=0,
            )
            gt_fixed = poses2pred(gt_talkshow)
            gt_body_parameters = torch.cat(
                (
                    torch.zeros(
                        (gt_fixed.shape[0], 3),
                        dtype=gt_fixed.dtype,
                        device=device,
                    ),
                    gt_fixed[:, 3:165],
                    torch.zeros(
                        (gt_fixed.shape[0], EXPRESSION_DIM),
                        dtype=gt_fixed.dtype,
                        device=device,
                    ),
                ),
                dim=-1,
            )

            pred_body_joints = get_joints(
                smplx_model,
                betas,
                full_body_prediction,
            )
            gt_body_joints = get_joints(
                smplx_model,
                betas,
                gt_body_parameters,
            )
            fgd_handler.push_joints(pred_body_joints, gt_body_joints)
            onset = get_mfcc_ta(
                str(clip.audio_path),
                fps=POSE_FPS,
                sr=AUDIO_SR,
                am="not None",
                encoder_choice="onset",
            )
            fgd_handler.push_aud(torch.from_numpy(onset))

            body_values = _body_loss(
                LVD,
                gt_body_joints,
                pred_body_joints,
            )
            body_rows.append(
                {
                    key: _finite_scalar(
                        value,
                        label=f"{clip.clip_id}/body/{key}",
                    )
                    for key, value in body_values.items()
                }
            )

            # Official face evaluation is a separate single-sample network.
            # With two main-model runs, sample0 is the canonical face sample.
            pred_face_parameters = torch.from_numpy(
                isolate_face_parameters(pred_talkshow_np[0])
            ).to(device)
            gt_face_parameters = torch.from_numpy(
                isolate_face_parameters(gt_talkshow_np)
            ).to(device)
            pred_face_joints = get_joints(
                smplx_model,
                betas,
                pred_face_parameters,
            )
            gt_face_joints = get_joints(
                smplx_model,
                betas,
                gt_face_parameters,
            )
            face_values = _face_loss(
                torch,
                LVD,
                gt_face_joints,
                pred_face_joints,
            )
            face_rows.append(
                {
                    key: _finite_scalar(
                        value,
                        label=f"{clip.clip_id}/face/{key}",
                    )
                    for key, value in face_values.items()
                }
            )

            print(
                f"[{clip_number}/{len(validation.clips)}] "
                f"evaluated {clip.clip_id}",
                flush=True,
            )

    body_mean = average_clip_metrics(body_rows)
    face_mean = average_clip_metrics(face_rows)
    fgd, feature_distance = fgd_handler.get_scores()
    try:
        beat_consistency = fgd_handler.get_BCscore()
    except Exception as exc:
        raise ProtocolError(
            "TalkSHOW get_BCscore failed; the official BC metric is undefined "
            f"for these inputs: {exc}"
        ) from exc

    metrics = {
        "fgd": _finite_scalar(fgd, label="fgd"),
        "feat_dist": _finite_scalar(feature_distance, label="feat_dist"),
        "bc": _finite_scalar(beat_consistency, label="bc"),
        "body_lvd": body_mean["LVD"],
        "body_error": body_mean["error"],
        "diverse": body_mean["diverse"],
        "jaw_l1": face_mean["jaw_l1"],
        "landmark_l1": face_mean["landmark_l1"],
        "face_lvd": face_mean["LVD"],
    }

    report = _base_report(args=args, validation=validation)
    report["protocol"]["alignment_audit"] = {
        "talkshow_source_hashes_verified": source_provenance[
            "source_hashes_verified"
        ],
        "original_source_audio": (
            validation.clips[0].audio_protocol
            == "talkshow_original_source"
        ),
        "official_clip_order": (
            validation.clips[0].audio_protocol
            == "talkshow_original_source"
        ),
        "official_short_clip_filter_applied": True,
        "complete_prediction_coverage": (
            not validation.missing_source_clip_ids
        ),
        "full_official_split_view_verified": (
            validation.full_split_view_verified
        ),
        "feature_extractor_expected_hash_supplied": bool(
            args.expected_feature_extractor_sha256
        ),
        "smplx_expected_hash_supplied": bool(
            args.expected_smplx_sha256
        ),
    }
    report["metrics"] = metrics
    evaluator_path = Path(__file__).resolve()
    main_repo_root = evaluator_path.parents[1]
    report["provenance"] = {
        "talkshow": dict(source_provenance),
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": _sha256_file(evaluator_path),
            "repository_root": str(main_repo_root),
            "repository_git_head": _git_head(main_repo_root),
        },
        "feature_extractor": {
            "path": str(feature_extractor),
            "sha256": feature_sha256,
            "expected_sha256": args.expected_feature_extractor_sha256,
        },
        "smplx": {
            "model_path": str(smplx_path),
            "neutral_asset": str(smplx_asset),
            "neutral_asset_sha256": smplx_asset_sha256,
            "expected_sha256": args.expected_smplx_sha256,
        },
        "device": str(device),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": getattr(torch, "__version__", None),
            "scipy": _distribution_version("scipy"),
            "librosa": _distribution_version("librosa"),
            "smplx": _distribution_version("smplx"),
            "torchaudio": _distribution_version("torchaudio"),
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        clips = build_input_manifest(
            args.prediction_dirs,
            args.gt_dir,
            args.audio_dir,
            args.source_audio_root,
            allow_source_subset=args.allow_source_subset,
        )
        validation = validate_input_manifest(clips)
        for warning in validation.warnings:
            print(f"[warning] {warning}", file=sys.stderr)

        source_provenance: dict[str, Any] | None = None
        if args.talkshow_root:
            source_provenance = verify_talkshow_sources(
                Path(args.talkshow_root),
                allow_unpinned=args.allow_unpinned_talkshow,
            )

        if args.validate_input_only:
            report = _base_report(args=args, validation=validation)
            report["status"] = "input_validation_passed"
            if source_provenance is not None:
                report["provenance"] = {"talkshow": source_provenance}
            if args.output:
                _write_json(Path(args.output), report)
                print(f"[done] validation report: {Path(args.output).resolve()}")
            else:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        missing_runtime = [
            option
            for option, value in (
                ("--talkshow-root", args.talkshow_root),
                ("--feature-extractor", args.feature_extractor),
                ("--output", args.output),
            )
            if not value
        ]
        if missing_runtime:
            raise ProtocolError(
                "metric execution requires " + ", ".join(missing_runtime)
            )
        if source_provenance is None:
            raise AssertionError("TalkSHOW source provenance was not initialized")

        report = run_official_evaluation(
            args=args,
            validation=validation,
            source_provenance=source_provenance,
        )
        _write_json(Path(args.output), report)
        print("[metrics]")
        for key, value in report["metrics"].items():
            print(f"  {key}: {value}")
        print(f"[done] official metrics: {Path(args.output).resolve()}")
        return 0
    except ProtocolError as exc:
        parser.exit(2, f"[abort] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
