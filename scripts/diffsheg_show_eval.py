#!/usr/bin/env python3
r"""Evaluate canonical TalkSHOW NPZ outputs with the public DiffSHEG SHOW protocol.

This is an explicitly reconstructed adapter.  DiffSHEG publishes the SHOW data
mapping, normalization, three autoencoder checkpoints, online validation code,
and 88-frame/10-frame-overlap generation setup, but its final external
``test_talkshow_ddpm.py`` is not present in the public repository.  The report
therefore records the reconstruction boundary instead of claiming byte-for-byte
identity with that unavailable script.

The default run computes the paper-facing FMD, FED, FGD, PCM, expression
diversity, gesture diversity, and BA.  Pass ``--skip-ba`` when only the
DiffSHEG motion/AE metrics are needed and the TalkSHOW/SMPL-X runtime is not
installed.

Examples
--------
Complete metric suite using one canonical main-project test output::

    python scripts/diffsheg_show_eval.py \
      --pred-dir /path/to/work_dir/npz/test \
      --diffsheg-root /path/to/DiffSHEG \
      --talkshow-root /path/to/TalkSHOW \
      --source-audio-root /path/to/talkshow_split_view \
      --device cuda:0 \
      --output /path/to/work_dir/diffsheg_show_metrics.json

Validate pairing, dimensions, statistics and BA inputs without importing torch::

    python scripts/diffsheg_show_eval.py \
      --pred-dir /path/to/work_dir/npz/test \
      --diffsheg-root /path/to/DiffSHEG \
      --talkshow-root /path/to/TalkSHOW \
      --source-audio-root /path/to/talkshow_split_view \
      --validate-input-only

See ``configs/talkshow_profile/DIFFSHEG_EVAL_zh.md`` for the full protocol,
asset preparation, generation commands and interpretation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


PROTOCOL_NAME = "diffsheg_show_reconstructed"
PROTOCOL_VERSION = 1
DIFFSHEG_REFERENCE_COMMIT = "3ebf3058f48cba3da9146afb7623e9ec1ab9e9a5"
POSE_DIM = 165
EXPRESSION_DIM = 100
GESTURE_DIM = 129
FACE_DIM = 103
HOLISTIC_DIM = GESTURE_DIM + FACE_DIM
POSE_FPS = 30
WINDOW_LENGTH = 88
GENERATION_OVERLAP_LENGTH = 10
DEFAULT_WINDOW_STRIDE = WINDOW_LENGTH
LATENT_DIM = 300
PCM_THRESHOLD = 1.0
PUBLIC_VALIDATION_PCK_THRESHOLD = 0.5
DIVERSITY_BATCH_SIZE = 50
AUDIO_SR = 16000

# Literal index form of DiffSHEG
# datasets/show.py:ShowDataset.extract_pose.  Keep this separate from
# TalkSHOW data_utils/lower_body.py:c_index_3d: that upstream constant selects
# a different fourth-body range, while DiffSHEG explicitly concatenates
# up1/up2/up3/up4 + both hands.
GESTURE_INDEX = np.asarray(
    [
        *range(18, 21),
        *range(27, 30),
        *range(36, 39),
        *range(45, 165),
    ],
    dtype=np.int64,
)

FEATURE_SPECS = (
    ("fmd", "holistic", HOLISTIC_DIM, "gesture_expression.pth.tar"),
    ("fed", "expression", FACE_DIM, "expression.pth.tar"),
    ("fgd", "gesture", GESTURE_DIM, "gesture.pth.tar"),
)


class ProtocolError(RuntimeError):
    """Raised when inputs cannot satisfy the reconstructed protocol."""


@dataclass(frozen=True)
class MotionArrays:
    poses: np.ndarray
    expressions: np.ndarray
    betas: np.ndarray | None
    fps: int


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    prediction_path: Path
    ground_truth_path: Path
    frame_count: int
    window_starts: tuple[int, ...]
    uncovered_tail_frames: int


@dataclass(frozen=True)
class NormalizationStats:
    gesture_mean: np.ndarray
    gesture_std: np.ndarray
    expression_mean: np.ndarray
    expression_std: np.ndarray


@dataclass(frozen=True)
class InputValidation:
    clips: tuple[ClipRecord, ...]
    total_frames: int
    total_windows: int
    uncovered_tail_frames: int
    clip_manifest_sha256: str
    clip_order: str


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


def _discover_files(directory: Path, prefix: str, suffix: str) -> dict[str, Path]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ProtocolError(f"input directory does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.glob(f"{prefix}*{suffix}")):
        if not path.is_file():
            continue
        clip_id = path.name[len(prefix) : -len(suffix)]
        if not clip_id:
            raise ProtocolError(f"empty clip ID in {path}")
        if clip_id in result:
            raise ProtocolError(f"duplicate clip ID {clip_id!r} in {root}")
        result[clip_id] = path.resolve()
    return result


def _read_clip_manifest(path: Path) -> tuple[str, ...]:
    try:
        lines = path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"cannot read clip manifest {path}: {exc}") from exc
    clip_ids = tuple(
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not clip_ids:
        raise ProtocolError(f"clip manifest is empty: {path}")
    if len(set(clip_ids)) != len(clip_ids):
        raise ProtocolError(f"clip manifest contains duplicate IDs: {path}")
    return clip_ids


def _load_motion(
    path: Path,
    *,
    role: str,
    require_betas: bool,
) -> MotionArrays:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"poses", "expressions", "mocap_frame_rate"}
            if require_betas:
                required.add("betas")
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ProtocolError(f"{role} {path} is missing keys: {missing}")
            poses = np.asarray(archive["poses"])
            expressions = np.asarray(archive["expressions"])
            fps_value = np.asarray(archive["mocap_frame_rate"]).reshape(-1)
            betas = (
                np.asarray(archive["betas"]).reshape(-1)
                if "betas" in archive.files
                else None
            )
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot read {role} NPZ {path}: {exc}") from exc

    if poses.ndim != 2 or poses.shape[1] != POSE_DIM:
        raise ProtocolError(
            f"{role} {path}: poses must be [T,{POSE_DIM}], got {poses.shape}"
        )
    if expressions.shape != (poses.shape[0], EXPRESSION_DIM):
        raise ProtocolError(
            f"{role} {path}: expressions must be "
            f"[{poses.shape[0]},{EXPRESSION_DIM}], got {expressions.shape}"
        )
    if fps_value.size != 1:
        raise ProtocolError(
            f"{role} {path}: mocap_frame_rate must be scalar, got {fps_value.shape}"
        )
    try:
        fps = int(fps_value[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(
            f"{role} {path}: invalid mocap_frame_rate {fps_value!r}"
        ) from exc
    if float(fps_value[0]) != float(POSE_FPS) or fps != POSE_FPS:
        raise ProtocolError(
            f"{role} {path}: mocap_frame_rate must be {POSE_FPS}, "
            f"got {fps_value[0]}"
        )
    if poses.shape[0] < WINDOW_LENGTH:
        raise ProtocolError(
            f"{role} {path}: DiffSHEG SHOW evaluation requires at least "
            f"{WINDOW_LENGTH} frames, got {poses.shape[0]}"
        )
    if require_betas and (betas is None or betas.shape != (300,)):
        shape = None if betas is None else betas.shape
        raise ProtocolError(
            f"{role} {path}: BA requires exactly 300 betas, got {shape}"
        )

    arrays: list[tuple[str, np.ndarray]] = [
        ("poses", poses),
        ("expressions", expressions),
    ]
    if betas is not None:
        arrays.append(("betas", betas))
    for name, array in arrays:
        if not np.issubdtype(array.dtype, np.floating):
            raise ProtocolError(
                f"{role} {path}: {name} must be floating point, got {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ProtocolError(f"{role} {path}: {name} contains NaN or Inf")

    return MotionArrays(
        poses=poses.astype(np.float32, copy=False),
        expressions=expressions.astype(np.float32, copy=False),
        betas=(
            betas.astype(np.float64, copy=False)
            if betas is not None
            else None
        ),
        fps=fps,
    )


def _window_starts(frame_count: int, stride: int) -> tuple[int, ...]:
    if stride < 1 or stride > WINDOW_LENGTH:
        raise ProtocolError(
            f"window stride must be in [1,{WINDOW_LENGTH}], got {stride}"
        )
    if frame_count < WINDOW_LENGTH:
        return ()
    return tuple(range(0, frame_count - WINDOW_LENGTH + 1, stride))


def validate_inputs(
    *,
    prediction_dir: Path,
    ground_truth_dir: Path,
    clip_manifest: Path | None,
    window_stride: int,
    require_betas: bool,
) -> InputValidation:
    predictions = _discover_files(prediction_dir, "res_", ".npz")
    ground_truth = _discover_files(ground_truth_dir, "gt_", ".npz")
    if not predictions:
        suspicious = sorted(prediction_dir.expanduser().resolve().glob("*res_*.npz"))
        hint = (
            f"; found non-canonical file {suspicious[0].name!r}"
            if suspicious
            else ""
        )
        raise ProtocolError(
            f"no canonical res_*.npz files in {prediction_dir}{hint}"
        )

    if clip_manifest is None:
        pred_ids = set(predictions)
        gt_ids = set(ground_truth)
        if pred_ids != gt_ids:
            raise ProtocolError(
                "prediction/GT clip sets differ: "
                f"missing_gt={sorted(pred_ids - gt_ids)[:20]}, "
                f"extra_gt={sorted(gt_ids - pred_ids)[:20]}"
            )
        ordered_ids = tuple(sorted(pred_ids))
        clip_order = "lexicographic_clip_id"
    else:
        ordered_ids = _read_clip_manifest(clip_manifest)
        requested = set(ordered_ids)
        missing_prediction = sorted(requested.difference(predictions))
        missing_gt = sorted(requested.difference(ground_truth))
        if missing_prediction or missing_gt:
            raise ProtocolError(
                "clip manifest references unavailable inputs: "
                f"missing_prediction={missing_prediction[:20]}, "
                f"missing_gt={missing_gt[:20]}"
            )
        clip_order = f"explicit_manifest:{clip_manifest.expanduser().resolve()}"

    records: list[ClipRecord] = []
    digest = hashlib.sha256()
    total_frames = 0
    total_windows = 0
    total_tail = 0
    for index, clip_id in enumerate(ordered_ids, start=1):
        if len(ordered_ids) >= 100 and (
            index == 1 or index % 100 == 0 or index == len(ordered_ids)
        ):
            print(
                f"[validate] {index}/{len(ordered_ids)} {clip_id}",
                file=sys.stderr,
                flush=True,
            )
        pred = _load_motion(
            predictions[clip_id],
            role=f"prediction {clip_id}",
            require_betas=False,
        )
        gt = _load_motion(
            ground_truth[clip_id],
            role=f"ground truth {clip_id}",
            require_betas=require_betas,
        )
        if pred.poses.shape[0] != gt.poses.shape[0]:
            raise ProtocolError(
                f"{clip_id}: prediction has {pred.poses.shape[0]} frames, "
                f"GT has {gt.poses.shape[0]}"
            )
        starts = _window_starts(gt.poses.shape[0], window_stride)
        if not starts:
            raise ProtocolError(
                f"{clip_id}: no complete {WINDOW_LENGTH}-frame window"
            )
        tail = gt.poses.shape[0] - (starts[-1] + WINDOW_LENGTH)
        records.append(
            ClipRecord(
                clip_id=clip_id,
                prediction_path=predictions[clip_id],
                ground_truth_path=ground_truth[clip_id],
                frame_count=gt.poses.shape[0],
                window_starts=starts,
                uncovered_tail_frames=tail,
            )
        )
        total_frames += gt.poses.shape[0]
        total_windows += len(starts)
        total_tail += tail
        digest.update(clip_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(gt.poses.shape[0]).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(value) for value in starts).encode("ascii"))
        digest.update(b"\n")

    return InputValidation(
        clips=tuple(records),
        total_frames=total_frames,
        total_windows=total_windows,
        uncovered_tail_frames=total_tail,
        clip_manifest_sha256=digest.hexdigest(),
        clip_order=clip_order,
    )


def beat2_to_talkshow(
    poses_165: np.ndarray,
    expressions_100: np.ndarray,
) -> np.ndarray:
    """Convert main-project/BEAT2 order to TalkSHOW's 265-D order."""

    if poses_165.ndim != 2 or poses_165.shape[-1] != POSE_DIM:
        raise ProtocolError(f"invalid main-project pose shape: {poses_165.shape}")
    if expressions_100.shape != (poses_165.shape[0], EXPRESSION_DIM):
        raise ProtocolError(
            f"invalid main-project expression shape: {expressions_100.shape}"
        )
    return np.concatenate(
        (
            poses_165[:, 66:75],   # jaw, left eye, right eye
            poses_165[:, 0:66],    # global orientation and body
            poses_165[:, 75:165],  # hands
            expressions_100,
        ),
        axis=-1,
    )


def split_diffsheg_components(
    poses_165: np.ndarray,
    expressions_100: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return DiffSHEG's 129-D gesture and 103-D expression arrays."""

    talkshow = beat2_to_talkshow(poses_165, expressions_100)
    gesture = talkshow[:, :POSE_DIM][:, GESTURE_INDEX]
    expression = np.concatenate(
        (talkshow[:, 0:3], talkshow[:, 165:265]),
        axis=-1,
    )
    if gesture.shape[-1] != GESTURE_DIM:
        raise AssertionError(f"gesture mapping produced {gesture.shape}")
    if expression.shape[-1] != FACE_DIM:
        raise AssertionError(f"expression mapping produced {expression.shape}")
    return gesture, expression


def _as_vector(payload: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    if key not in payload:
        raise ProtocolError(f"statistics file is missing {key!r}")
    value = np.asarray(payload[key])
    if value.size != size:
        raise ProtocolError(
            f"statistics {key!r} must contain {size} values, got {value.shape}"
        )
    vector = value.reshape(size).astype(np.float32, copy=False)
    if not np.isfinite(vector).all():
        raise ProtocolError(f"statistics {key!r} contains NaN or Inf")
    return vector


def load_normalization_stats(path: Path) -> NormalizationStats:
    stats_path = path.expanduser().resolve()
    if not stats_path.is_file():
        raise ProtocolError(f"DiffSHEG statistics file does not exist: {stats_path}")
    try:
        loaded = np.load(stats_path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                payload: Any = {key: loaded[key] for key in loaded.files}
            finally:
                loaded.close()
        elif isinstance(loaded, np.ndarray) and loaded.shape == ():
            payload = loaded.item()
        else:
            payload = loaded
    except (OSError, ValueError, EOFError) as exc:
        raise ProtocolError(f"cannot load DiffSHEG statistics {stats_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError(
            f"DiffSHEG statistics must contain a mapping, got {type(payload).__name__}"
        )

    pose_mean = _as_vector(payload, "pose_mean", POSE_DIM)
    pose_std = _as_vector(payload, "pose_std", POSE_DIM)
    raw_expression_mean = _as_vector(
        payload, "expression_mean", EXPRESSION_DIM
    )
    raw_expression_std = _as_vector(payload, "expression_std", EXPRESSION_DIM)
    gesture_mean = pose_mean[GESTURE_INDEX]
    gesture_std = pose_std[GESTURE_INDEX]
    expression_mean = np.concatenate(
        (pose_mean[:3], raw_expression_mean), axis=0
    )
    expression_std = np.concatenate(
        (pose_std[:3], raw_expression_std), axis=0
    )
    for label, std in (
        ("gesture_std", gesture_std),
        ("expression_std", expression_std),
    ):
        invalid = np.flatnonzero(std <= 0)
        if invalid.size:
            raise ProtocolError(
                f"{label} must be strictly positive; invalid indices "
                f"{invalid[:20].tolist()}"
            )
    return NormalizationStats(
        gesture_mean=gesture_mean,
        gesture_std=gesture_std,
        expression_mean=expression_mean,
        expression_std=expression_std,
    )


def _standardized_components(
    motion: MotionArrays,
    stats: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray]:
    gesture, expression = split_diffsheg_components(
        motion.poses, motion.expressions
    )
    gesture = (gesture - stats.gesture_mean) / stats.gesture_std
    expression = (expression - stats.expression_mean) / stats.expression_std
    if not np.isfinite(gesture).all() or not np.isfinite(expression).all():
        raise ProtocolError("standardized motion contains NaN or Inf")
    return (
        gesture.astype(np.float32, copy=False),
        expression.astype(np.float32, copy=False),
    )


def iter_standardized_windows(
    validation: InputValidation,
    stats: NormalizationStats,
) -> Iterator[
    tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Yield pred/GT gesture and expression windows in deterministic order."""

    for clip in validation.clips:
        pred = _load_motion(
            clip.prediction_path,
            role=f"prediction {clip.clip_id}",
            require_betas=False,
        )
        gt = _load_motion(
            clip.ground_truth_path,
            role=f"ground truth {clip.clip_id}",
            require_betas=False,
        )
        pred_gesture, pred_expression = _standardized_components(pred, stats)
        gt_gesture, gt_expression = _standardized_components(gt, stats)
        for start in clip.window_starts:
            stop = start + WINDOW_LENGTH
            yield (
                clip.clip_id,
                start,
                pred_gesture[start:stop],
                gt_gesture[start:stop],
                pred_expression[start:stop],
                gt_expression[start:stop],
            )


def _pairwise_l1(samples: Sequence[np.ndarray]) -> float:
    count = len(samples)
    if count < 2:
        raise ProtocolError("diversity requires at least two sequences")
    stacked = np.stack(samples, axis=0)
    total = 0.0
    for index in range(count - 1):
        differences = np.abs(stacked[index + 1 :] - stacked[index])
        total += float(differences.mean(axis=(1, 2), dtype=np.float64).sum())
    return total * 2.0 / (count * (count - 1))


class _DiversityAccumulator:
    def __init__(self, group_size: int) -> None:
        self.group_size = group_size
        self.gesture: list[np.ndarray] = []
        self.expression: list[np.ndarray] = []
        self.weighted_gesture = 0.0
        self.weighted_expression = 0.0
        self.sample_count = 0
        self.group_sizes: list[int] = []

    def add(self, gesture: np.ndarray, expression: np.ndarray) -> None:
        self.gesture.append(np.ascontiguousarray(gesture))
        self.expression.append(np.ascontiguousarray(expression))
        if len(self.gesture) == self.group_size:
            self._flush()

    def _flush(self) -> None:
        count = len(self.gesture)
        if count < 2:
            return
        self.weighted_gesture += _pairwise_l1(self.gesture) * count
        self.weighted_expression += _pairwise_l1(self.expression) * count
        self.sample_count += count
        self.group_sizes.append(count)
        self.gesture.clear()
        self.expression.clear()

    def finish(self) -> dict[str, Any]:
        dropped_singleton = 0
        if len(self.gesture) >= 2:
            self._flush()
        elif self.gesture:
            dropped_singleton = 1
            self.gesture.clear()
            self.expression.clear()
        if self.sample_count == 0:
            raise ProtocolError("not enough windows to compute diversity")
        return {
            "gesture_diversity": self.weighted_gesture / self.sample_count,
            "expression_diversity": self.weighted_expression / self.sample_count,
            "diversity_sample_count": self.sample_count,
            "diversity_group_sizes": self.group_sizes,
            "diversity_dropped_singleton": dropped_singleton,
        }


def compute_parameter_metrics(
    validation: InputValidation,
    stats: NormalizationStats,
) -> tuple[dict[str, float], dict[str, Any]]:
    correct_pcm = 0
    pcm_joint_count = 0
    correct_scalar_at_one = 0
    correct_public_pck = 0
    scalar_count = 0
    squared_error_sum = 0.0
    diversity = _DiversityAccumulator(DIVERSITY_BATCH_SIZE)
    for clip_index, clip in enumerate(validation.clips, start=1):
        pred_motion = _load_motion(
            clip.prediction_path,
            role=f"prediction {clip.clip_id}",
            require_betas=False,
        )
        gt_motion = _load_motion(
            clip.ground_truth_path,
            role=f"ground truth {clip.clip_id}",
            require_betas=False,
        )
        pred_gesture, _ = _standardized_components(pred_motion, stats)
        gt_gesture, _ = _standardized_components(gt_motion, stats)
        difference = pred_gesture - gt_gesture
        absolute = np.abs(difference)
        joint_error = np.linalg.norm(
            difference.reshape(
                difference.shape[0], GESTURE_DIM // 3, 3
            ),
            axis=-1,
        )
        correct_pcm += int(np.count_nonzero(joint_error < PCM_THRESHOLD))
        pcm_joint_count += joint_error.size
        correct_scalar_at_one += int(
            np.count_nonzero(absolute < PCM_THRESHOLD)
        )
        correct_public_pck += int(
            np.count_nonzero(absolute < PUBLIC_VALIDATION_PCK_THRESHOLD)
        )
        scalar_count += difference.size
        squared_error_sum += float(
            np.square(difference, dtype=np.float64).sum(dtype=np.float64)
        )
        if clip_index % 100 == 0 or clip_index == len(validation.clips):
            print(
                f"[PCM] {clip_index}/{len(validation.clips)} clips",
                flush=True,
            )

    for window_index, window in enumerate(
        iter_standardized_windows(validation, stats),
        start=1,
    ):
        _, _, pred_gesture, _, pred_expression, _ = window
        diversity.add(pred_gesture, pred_expression)
        if window_index % 1000 == 0:
            print(
                f"[diversity] {window_index}/{validation.total_windows} windows",
                flush=True,
            )
    diversity_values = diversity.finish()
    metrics = {
        "pcm": correct_pcm / pcm_joint_count,
        "gesture_diversity": float(diversity_values["gesture_diversity"]),
        "expression_diversity": float(
            diversity_values["expression_diversity"]
        ),
    }
    diagnostics = {
        "gesture_scalar_accuracy_at_1_0": (
            correct_scalar_at_one / scalar_count
        ),
        "gesture_pck_at_0_5_public_validation_code": (
            correct_public_pck / scalar_count
        ),
        "gesture_mse_standardized": squared_error_sum / scalar_count,
        "pcm_frame_count": validation.total_frames,
        "pcm_threshold": PCM_THRESHOLD,
        "public_validation_pck_threshold": PUBLIC_VALIDATION_PCK_THRESHOLD,
        "diversity_batch_size": DIVERSITY_BATCH_SIZE,
        "diversity_sample_count": diversity_values[
            "diversity_sample_count"
        ],
        "diversity_group_sizes": diversity_values["diversity_group_sizes"],
        "diversity_dropped_singleton": diversity_values[
            "diversity_dropped_singleton"
        ],
    }
    return metrics, diagnostics


def _build_embedding_model(torch: Any, input_dim: int) -> Any:
    """Build the public DiffSHEG HalfEmbeddingNet exactly for SHOW."""

    nn = torch.nn

    def conv_norm_relu(
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        padding: int = 0,
        batchnorm: bool = True,
    ) -> Any:
        kernel = 4 if downsample else 3
        stride = 2 if downsample else 1
        layers: list[Any] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        ]
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    class PoseEncoderConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = LATENT_DIM
            self.net = nn.Sequential(
                conv_norm_relu(input_dim, self.base, batchnorm=True),
                conv_norm_relu(self.base, self.base * 2, batchnorm=True),
                conv_norm_relu(
                    self.base * 2,
                    self.base * 2,
                    True,
                    batchnorm=True,
                ),
                nn.Conv1d(self.base * 2, self.base, 3),
            )
            self.out_net = nn.Sequential(
                nn.Linear(39 * self.base, self.base * 12),
                nn.BatchNorm1d(self.base * 12),
                nn.Linear(12 * self.base, self.base * 4),
                nn.BatchNorm1d(self.base * 4),
                # Keep the upstream positional call exactly: True is passed
                # as negative_slope, not as inplace.
                nn.LeakyReLU(True),
                nn.Linear(self.base * 4, self.base * 2),
                nn.BatchNorm1d(self.base * 2),
                nn.LeakyReLU(True),
                nn.Linear(self.base * 2, self.base),
            )
            self.fc_mu = nn.Linear(self.base, self.base)
            self.fc_logvar = nn.Linear(self.base, self.base)

        def forward(self, poses: Any) -> tuple[Any, Any, Any]:
            output = self.net(poses.transpose(1, 2))
            output = self.out_net(output.flatten(1))
            mu = self.fc_mu(output)
            logvar = self.fc_logvar(output)
            return mu, mu, logvar

    class PoseDecoderConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_pre_poses = False
            self.feat_size = LATENT_DIM
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size * 2),
                nn.BatchNorm1d(self.feat_size * 2),
                nn.LeakyReLU(True),
                nn.Linear(
                    self.feat_size * 2,
                    self.feat_size // 8 * WINDOW_LENGTH,
                ),
            )
            self.decoder_size = self.feat_size // 8
            self.net = nn.Sequential(
                nn.ConvTranspose1d(
                    self.decoder_size, self.feat_size, 3
                ),
                nn.BatchNorm1d(self.feat_size),
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose1d(self.feat_size, self.feat_size, 3),
                nn.BatchNorm1d(self.feat_size),
                nn.LeakyReLU(0.2, True),
                nn.Conv1d(self.feat_size, self.feat_size * 2, 3),
                nn.Conv1d(self.feat_size * 2, input_dim, 3),
            )

        def forward(self, feature: Any) -> Any:
            output = self.pre_net(feature)
            output = output.view(
                feature.shape[0], self.decoder_size, -1
            )
            return self.net(output).transpose(1, 2)

    class HalfEmbeddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pose_encoder = PoseEncoderConv()
            self.decoder = PoseDecoderConv()

        def forward(self, poses: Any) -> Any:
            feature, _, _ = self.pose_encoder(poses)
            return feature

    return HalfEmbeddingNet()


def _checkpoint_state(payload: Any, path: Path) -> tuple[Mapping[str, Any], str]:
    if not isinstance(payload, Mapping):
        raise ProtocolError(
            f"checkpoint {path} must contain a state mapping, "
            f"got {type(payload).__name__}"
        )
    for key in ("model_state", "state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value, key
    return payload, "root"


def _strip_module_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def load_embedding_model(
    *,
    torch: Any,
    checkpoint_path: Path,
    input_dim: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    model = _build_embedding_model(torch, input_dim)
    try:
        payload = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # DiffSHEG's pinned PyTorch 1.13 does not expose weights_only.
        payload = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        raise ProtocolError(
            f"cannot safely load DiffSHEG checkpoint {checkpoint_path}: {exc}"
        ) from exc

    raw_state, container = _checkpoint_state(payload, checkpoint_path)
    state = {
        _strip_module_prefix(str(key)): value
        for key, value in raw_state.items()
        if isinstance(key, str)
    }
    full_expected = set(model.state_dict())
    if full_expected.issubset(state):
        selected = {key: state[key] for key in full_expected}
        try:
            model.load_state_dict(selected, strict=True)
        except Exception as exc:
            raise ProtocolError(
                f"{checkpoint_path}: incompatible full HalfEmbeddingNet "
                f"weights: {exc}"
            ) from exc
        load_mode = "full_half_embedding_net"
    else:
        encoder_expected = set(model.pose_encoder.state_dict())
        encoder_state: dict[str, Any] = {}
        for key, value in state.items():
            if key.startswith("pose_encoder."):
                encoder_state[key[len("pose_encoder.") :]] = value
            elif key in encoder_expected:
                encoder_state[key] = value
        missing = sorted(encoder_expected.difference(encoder_state))
        if missing:
            raise ProtocolError(
                f"{checkpoint_path}: missing encoder weights "
                f"{missing[:20]}; this is not a DiffSHEG SHOW AE checkpoint"
            )
        try:
            model.pose_encoder.load_state_dict(
                {key: encoder_state[key] for key in encoder_expected},
                strict=True,
            )
        except Exception as exc:
            raise ProtocolError(
                f"{checkpoint_path}: incompatible PoseEncoderConv weights: {exc}"
            ) from exc
        load_mode = "encoder_only"

    del payload, raw_state, state
    model.eval()
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    return model, {
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "state_container": container,
        "load_mode": load_mode,
        "input_dim": input_dim,
        "latent_dim": LATENT_DIM,
    }


def _component_from_window(
    window: tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, pred_gesture, gt_gesture, pred_expression, gt_expression = window
    if component == "gesture":
        return pred_gesture, gt_gesture
    if component == "expression":
        return pred_expression, gt_expression
    if component == "holistic":
        return (
            np.concatenate((pred_gesture, pred_expression), axis=-1),
            np.concatenate((gt_gesture, gt_expression), axis=-1),
        )
    raise AssertionError(f"unknown component {component}")


def extract_features(
    *,
    torch: Any,
    model: Any,
    device: Any,
    validation: InputValidation,
    stats: NormalizationStats,
    component: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_features: list[np.ndarray] = []
    gt_features: list[np.ndarray] = []
    pred_batch: list[np.ndarray] = []
    gt_batch: list[np.ndarray] = []
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not pred_batch:
            return
        pred_array = np.stack(pred_batch, axis=0)
        gt_array = np.stack(gt_batch, axis=0)
        combined = np.concatenate((pred_array, gt_array), axis=0)
        tensor = torch.from_numpy(combined).to(
            device=device,
            dtype=torch.float32,
        )
        feature = model(tensor).detach().cpu().numpy()
        count = pred_array.shape[0]
        pred_features.append(feature[:count])
        gt_features.append(feature[count:])
        processed += count
        pred_batch.clear()
        gt_batch.clear()
        if processed % 1000 < count:
            print(
                f"[{component} AE] {processed}/{validation.total_windows} windows",
                flush=True,
            )

    with torch.inference_mode():
        for window in iter_standardized_windows(validation, stats):
            pred, gt = _component_from_window(window, component)
            pred_batch.append(np.ascontiguousarray(pred))
            gt_batch.append(np.ascontiguousarray(gt))
            if len(pred_batch) == batch_size:
                flush()
        flush()
    if processed != validation.total_windows:
        raise AssertionError(
            f"feature pass processed {processed}, expected "
            f"{validation.total_windows}"
        )
    return (
        np.concatenate(pred_features, axis=0),
        np.concatenate(gt_features, axis=0),
    )


def frechet_distance(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Copy DiffSHEG's FIDCalculator formulas, including its fallback."""

    if (
        samples_a.ndim != 2
        or samples_b.ndim != 2
        or samples_a.shape != samples_b.shape
    ):
        raise ProtocolError(
            f"feature arrays must have equal [N,D] shapes, got "
            f"{samples_a.shape} and {samples_b.shape}"
        )
    if samples_a.shape[0] < 2:
        raise ProtocolError("Fréchet distance requires at least two windows")
    try:
        from scipy import linalg
    except ImportError as exc:
        raise ProtocolError("SciPy is required for FMD/FED/FGD") from exc

    mu_a = np.mean(samples_a, axis=0)
    sigma_a = np.cov(samples_a, rowvar=False)
    mu_b = np.mean(samples_b, axis=0)
    sigma_b = np.cov(samples_b, rowvar=False)
    difference = mu_a - mu_b
    try:
        try:
            sqrt_result = linalg.sqrtm(
                sigma_a.dot(sigma_b), disp=False
            )
        except TypeError:
            sqrt_result = linalg.sqrtm(sigma_a.dot(sigma_b))
        covmean = (
            sqrt_result[0]
            if isinstance(sqrt_result, tuple)
            else sqrt_result
        )
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_a.shape[0]) * 1e-6
            covmean = linalg.sqrtm(
                (sigma_a + offset).dot(sigma_b + offset)
            )
        if np.iscomplexobj(covmean):
            if not np.allclose(
                np.diagonal(covmean).imag, 0, atol=1e-3
            ):
                raise ValueError(
                    f"imaginary component {np.max(np.abs(covmean.imag))}"
                )
            covmean = covmean.real
        value = (
            difference.dot(difference)
            + np.trace(sigma_a)
            + np.trace(sigma_b)
            - 2 * np.trace(covmean)
        )
    except ValueError:
        # This is the literal public DiffSHEG fallback.
        value = 1e10
    result = float(value)
    if not np.isfinite(result):
        raise ProtocolError(f"Fréchet distance is not finite: {result}")
    return result


def _resolve_runtime_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Path]]:
    diffsheg_root = (
        Path(args.diffsheg_root).expanduser().resolve()
        if args.diffsheg_root
        else None
    )
    stats_path = (
        Path(args.stats).expanduser().resolve()
        if args.stats
        else (
            diffsheg_root / "data/SHOW/talkshow_mean_std.npy"
            if diffsheg_root is not None
            else None
        )
    )
    weights_dir = (
        Path(args.weights_dir).expanduser().resolve()
        if args.weights_dir
        else (
            diffsheg_root / "data/SHOW/ae_weights"
            if diffsheg_root is not None
            else None
        )
    )
    if stats_path is None or weights_dir is None:
        raise ProtocolError(
            "pass --diffsheg-root, or pass both --stats and --weights-dir"
        )
    if not stats_path.is_file():
        raise ProtocolError(f"DiffSHEG statistics do not exist: {stats_path}")
    if not weights_dir.is_dir():
        raise ProtocolError(f"DiffSHEG AE directory does not exist: {weights_dir}")
    checkpoints = {
        metric: (weights_dir / filename).resolve()
        for metric, _, _, filename in FEATURE_SPECS
    }
    missing = [
        f"{metric}={path}"
        for metric, path in checkpoints.items()
        if not path.is_file()
    ]
    if missing:
        raise ProtocolError(
            "missing DiffSHEG SHOW AE checkpoints: " + ", ".join(missing)
        )
    return stats_path, weights_dir, checkpoints


def _resolve_audio_paths(
    validation: InputValidation,
    *,
    audio_dir: Path | None,
    source_audio_root: Path | None,
) -> tuple[dict[str, Path], str]:
    if (audio_dir is None) == (source_audio_root is None):
        raise ProtocolError(
            "BA requires exactly one of --audio-dir or --source-audio-root"
        )
    result: dict[str, Path] = {}
    if audio_dir is not None:
        root = audio_dir.expanduser().resolve()
        discovered = _discover_files(root, "gt_sr16000_", ".wav")
        missing = [
            clip.clip_id
            for clip in validation.clips
            if clip.clip_id not in discovered
        ]
        if missing:
            raise ProtocolError(
                f"aligned audio is missing {len(missing)} clips: {missing[:20]}"
            )
        for clip in validation.clips:
            path = discovered[clip.clip_id]
            try:
                with wave.open(str(path), "rb") as handle:
                    channels = handle.getnchannels()
                    sample_rate = handle.getframerate()
                    samples = handle.getnframes()
                    compression = handle.getcomptype()
            except (OSError, EOFError, wave.Error) as exc:
                raise ProtocolError(f"cannot read WAV {path}: {exc}") from exc
            expected_samples = AUDIO_SR * clip.frame_count // POSE_FPS
            if (
                channels != 1
                or sample_rate != AUDIO_SR
                or samples != expected_samples
                or compression != "NONE"
            ):
                raise ProtocolError(
                    f"{path}: aligned WAV must be mono PCM at {AUDIO_SR} Hz "
                    f"with {expected_samples} samples; got channels={channels}, "
                    f"rate={sample_rate}, samples={samples}, "
                    f"compression={compression}"
                )
            result[clip.clip_id] = path
        return result, "paspa_aligned_output"

    root = source_audio_root.expanduser().resolve()
    if not root.is_dir():
        raise ProtocolError(f"source audio root does not exist: {root}")
    for clip in validation.clips:
        if "__" not in clip.clip_id:
            raise ProtocolError(
                f"{clip.clip_id}: source audio lookup requires "
                "<speaker>__<sequence> IDs"
            )
        speaker, sequence = clip.clip_id.split("__", 1)
        candidates = sorted(
            path.resolve()
            for path in (root / speaker).glob(
                f"*/test/{sequence}/{sequence}.wav"
            )
            if path.is_file()
        )
        if len(candidates) != 1:
            raise ProtocolError(
                f"{clip.clip_id}: expected one source WAV under {root}, "
                f"found {len(candidates)}: {candidates[:5]}"
            )
        result[clip.clip_id] = candidates[0]
    return result, "talkshow_original_source"


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_smplx_asset(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "smplx" / "SMPLX_NEUTRAL.npz"
    if candidate.is_file():
        return candidate
    raise ProtocolError(
        f"cannot find SMPLX_NEUTRAL.npz under {path}; expected {candidate}"
    )


def _body_partial_parameters(gesture: np.ndarray) -> np.ndarray:
    if gesture.ndim != 2 or gesture.shape[-1] != GESTURE_DIM:
        raise ProtocolError(f"invalid gesture shape for BA: {gesture.shape}")
    return np.concatenate(
        (
            np.zeros((gesture.shape[0], 3), dtype=gesture.dtype),
            gesture,
            np.zeros(
                (gesture.shape[0], EXPRESSION_DIM), dtype=gesture.dtype
            ),
        ),
        axis=-1,
    )


def compute_ba(
    *,
    validation: InputValidation,
    audio_paths: Mapping[str, Path],
    talkshow_root: Path,
    smplx_path: Path,
    device_name: str,
) -> tuple[float, dict[str, Any]]:
    root = talkshow_root.expanduser().resolve()
    if not root.is_dir():
        raise ProtocolError(f"TalkSHOW root does not exist: {root}")
    resolved_smplx_path = smplx_path.expanduser().resolve()
    smplx_asset = _resolve_smplx_asset(resolved_smplx_path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        with _working_directory(root):
            import smplx as smpl
            import torch

            from data_utils.get_j import get_joints
            from data_utils.lower_body import part2full
            from data_utils.utils import get_mfcc_ta
            from evaluation.FGD import EmbeddingSpaceEvaluator
    except (ImportError, FileNotFoundError) as exc:
        raise ProtocolError(
            "cannot import the TalkSHOW BA/SMPL-X environment: "
            f"{exc}"
        ) from exc
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ProtocolError(
                f"requested {device}, but torch.cuda.is_available() is false"
            )
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")

    smplx_model = smpl.create(
        model_path=str(resolved_smplx_path),
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
    handler = EmbeddingSpaceEvaluator(None, None, str(device))
    with torch.no_grad():
        for index, clip in enumerate(validation.clips, start=1):
            pred = _load_motion(
                clip.prediction_path,
                role=f"prediction {clip.clip_id}",
                require_betas=False,
            )
            gt = _load_motion(
                clip.ground_truth_path,
                role=f"ground truth {clip.clip_id}",
                require_betas=True,
            )
            gesture, _ = split_diffsheg_components(
                pred.poses, pred.expressions
            )
            partial = torch.from_numpy(
                _body_partial_parameters(gesture)
            ).to(device)
            full_parameters = part2full(partial)
            if gt.betas is None:
                raise AssertionError("validated betas disappeared")
            betas = torch.from_numpy(gt.betas).to(
                device=device, dtype=torch.float64
            )
            joints = get_joints(smplx_model, betas, full_parameters)
            handler.push_joints(joints, joints)
            onset = get_mfcc_ta(
                str(audio_paths[clip.clip_id]),
                fps=POSE_FPS,
                sr=AUDIO_SR,
                am="not None",
                encoder_choice="onset",
            )
            handler.push_aud(torch.from_numpy(onset))
            if index % 100 == 0 or index == len(validation.clips):
                print(
                    f"[BA] {index}/{len(validation.clips)} clips",
                    flush=True,
                )
    try:
        value = handler.get_BCscore()
    except Exception as exc:
        raise ProtocolError(
            "TalkSHOW get_BCscore failed; BA is undefined for these "
            f"predictions/audio: {exc}"
        ) from exc
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    result = float(value)
    if not np.isfinite(result):
        raise ProtocolError(f"BA is not finite: {result}")
    provenance = {
        "implementation": "TalkSHOW evaluation.FGD.EmbeddingSpaceEvaluator.get_BCscore",
        "talkshow_root": str(root),
        "talkshow_git_head": _git_head(root),
        "smplx_model_path": str(resolved_smplx_path),
        "smplx_neutral_asset": str(smplx_asset),
        "smplx_neutral_asset_sha256": _sha256_file(smplx_asset),
    }
    del handler, smplx_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, provenance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Canonical npz/test directory containing res_*.npz.",
    )
    parser.add_argument(
        "--gt-dir",
        help=(
            "Directory containing matching gt_*.npz. Defaults to --pred-dir, "
            "which is the normal canonical test layout."
        ),
    )
    parser.add_argument(
        "--clip-manifest",
        help=(
            "Optional ordered UTF-8 clip-ID list. Without it, all exact "
            "prediction/GT matches are evaluated in lexical ID order."
        ),
    )
    parser.add_argument(
        "--diffsheg-root",
        help=(
            "DiffSHEG repository root after assets/data.tar.gz is extracted. "
            "Infers statistics and data/SHOW/ae_weights."
        ),
    )
    parser.add_argument(
        "--stats",
        help=(
            "Explicit talkshow_mean_std.npy; overrides the path inferred from "
            "--diffsheg-root."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        help=(
            "Directory containing gesture.pth.tar, expression.pth.tar and "
            "gesture_expression.pth.tar; overrides --diffsheg-root."
        ),
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=DEFAULT_WINDOW_STRIDE,
        help=(
            f"Stride for {WINDOW_LENGTH}-frame AE windows (default: "
            f"{DEFAULT_WINDOW_STRIDE}, non-overlapping as specified by "
            "DiffSHEG's metric appendix)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Number of prediction windows per AE forward call; GT is "
            "concatenated internally, so the actual tensor has 2x rows "
            "(default: 64)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for all AE/SMPL-X calculations (default: cuda:0).",
    )
    parser.add_argument(
        "--skip-ba",
        action="store_true",
        help="Skip BA and avoid the TalkSHOW/SMPL-X/audio dependencies.",
    )
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument(
        "--source-audio-root",
        help=(
            "TalkSHOW split-view root containing "
            "<speaker>/*/test/<sequence>/<sequence>.wav; recommended for BA."
        ),
    )
    audio_group.add_argument(
        "--audio-dir",
        help=(
            "Canonical wav/test directory containing gt_sr16000_*.wav; "
            "fallback BA audio source."
        ),
    )
    parser.add_argument(
        "--talkshow-root",
        help="TalkSHOW repository root, required for BA.",
    )
    parser.add_argument(
        "--smplx-path",
        help=(
            "SMPL-X model path for BA. Defaults to "
            "<talkshow-root>/visualise."
        ),
    )
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help=(
            "Validate NPZ pairing, statistics, checkpoint paths and BA audio "
            "pairing without importing torch or loading checkpoints."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output JSON. Defaults to "
            "<pred-dir>/diffsheg_show_metrics.json."
        ),
    )
    return parser


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _base_report(
    *,
    args: argparse.Namespace,
    validation: InputValidation,
    stats_path: Path,
    weights_dir: Path,
    checkpoints: Mapping[str, Path],
    audio_protocol: str | None,
) -> dict[str, Any]:
    warnings = [
        (
            "DiffSHEG's public repository does not include the final external "
            "0_BEAT_ori/codes/audio2pose/test_talkshow_ddpm.py invoked after "
            "SHOW generation. This evaluator reconstructs the protocol from "
            "the published dataset, model, validation and paper definitions."
        )
    ]
    if args.window_stride != DEFAULT_WINDOW_STRIDE:
        warnings.append(
            f"non-default window stride {args.window_stride} was requested; "
            f"the reconstructed comparison default is {DEFAULT_WINDOW_STRIDE}."
        )
    if audio_protocol == "paspa_aligned_output":
        warnings.append(
            "BA uses the canonical resampled/cropped PASPA WAV instead of the "
            "original TalkSHOW source WAV; motion metrics are unaffected, but "
            "BA can differ near duration boundaries."
        )
    return {
        "status": "initialized",
        "protocol": {
            "name": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "status": "reconstructed_from_public_components",
            "sources": {
                "repository": "https://github.com/JeremyCJM/DiffSHEG",
                "paper": "https://arxiv.org/abs/2401.04747",
            },
            "diffsheg_reference_commit": DIFFSHEG_REFERENCE_COMMIT,
            "public_offline_entry_available": False,
            "parameter_order": {
                "input": "PASPA/BEAT2 poses[165] + expressions[100]",
                "gesture": "DiffSHEG ShowDataset.extract_pose, 129-D",
                "expression": "jaw[3] + expression[100], 103-D",
                "holistic": "gesture[129] + expression[103], 232-D",
            },
            "normalization": "DiffSHEG talkshow_mean_std.npy",
            "window_length": WINDOW_LENGTH,
            "overlap_length": WINDOW_LENGTH - args.window_stride,
            "window_stride": args.window_stride,
            "ae_and_diversity_tail_policy": "drop_incomplete_tail",
            "pcm_frame_policy": "all_frames_once",
            "generation_reference_overlap_length": (
                GENERATION_OVERLAP_LENGTH
            ),
            "clip_order": validation.clip_order,
            "frechet_implementation": "DiffSHEG datasets.data_tools.FIDCalculator",
            "pcm": {
                "space": (
                    "standardized 129-D gesture parameters reshaped to "
                    "43 axis-angle joint vectors"
                ),
                "threshold": PCM_THRESHOLD,
                "comparison": "joint_vector_l2_error < threshold",
            },
            "diversity": {
                "space": "standardized generated parameters",
                "group_size": DIVERSITY_BATCH_SIZE,
                "distance": "mean absolute pairwise distance",
            },
            "ba": (
                "TalkSHOW EmbeddingSpaceEvaluator.get_BCscore"
                if not args.skip_ba
                else None
            ),
            "precision": "float32 AE inference; no autocast",
        },
        "inputs": {
            "prediction_dir": str(
                Path(args.pred_dir).expanduser().resolve()
            ),
            "ground_truth_dir": str(
                Path(args.gt_dir or args.pred_dir).expanduser().resolve()
            ),
            "clip_manifest": (
                str(Path(args.clip_manifest).expanduser().resolve())
                if args.clip_manifest
                else None
            ),
            "clip_count": len(validation.clips),
            "frame_count": validation.total_frames,
            "window_count": validation.total_windows,
            "uncovered_tail_frames": validation.uncovered_tail_frames,
            "clip_manifest_sha256": validation.clip_manifest_sha256,
            "stats": {
                "path": str(stats_path),
                "sha256": _sha256_file(stats_path),
            },
            "weights_dir": str(weights_dir),
            "checkpoint_paths": {
                key: str(value) for key, value in checkpoints.items()
            },
            "audio_protocol": audio_protocol,
        },
        "warnings": warnings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ProtocolError(
                f"--batch-size must be positive, got {args.batch_size}"
            )
        stats_path, weights_dir, checkpoints = _resolve_runtime_paths(args)
        stats = load_normalization_stats(stats_path)
        prediction_dir = Path(args.pred_dir).expanduser().resolve()
        ground_truth_dir = Path(
            args.gt_dir or args.pred_dir
        ).expanduser().resolve()
        validation = validate_inputs(
            prediction_dir=prediction_dir,
            ground_truth_dir=ground_truth_dir,
            clip_manifest=(
                Path(args.clip_manifest) if args.clip_manifest else None
            ),
            window_stride=args.window_stride,
            require_betas=not args.skip_ba,
        )

        audio_paths: dict[str, Path] | None = None
        audio_protocol: str | None = None
        talkshow_root: Path | None = None
        smplx_path: Path | None = None
        if not args.skip_ba:
            if not args.talkshow_root:
                raise ProtocolError("BA requires --talkshow-root")
            talkshow_root = Path(args.talkshow_root).expanduser().resolve()
            smplx_path = (
                Path(args.smplx_path).expanduser().resolve()
                if args.smplx_path
                else talkshow_root / "visualise"
            )
            # Resolve the asset before importing any heavy dependency.
            _resolve_smplx_asset(smplx_path)
            audio_paths, audio_protocol = _resolve_audio_paths(
                validation,
                audio_dir=(
                    Path(args.audio_dir) if args.audio_dir else None
                ),
                source_audio_root=(
                    Path(args.source_audio_root)
                    if args.source_audio_root
                    else None
                ),
            )

        report = _base_report(
            args=args,
            validation=validation,
            stats_path=stats_path,
            weights_dir=weights_dir,
            checkpoints=checkpoints,
            audio_protocol=audio_protocol,
        )
        if args.diffsheg_root:
            observed_diffsheg_head = _git_head(
                Path(args.diffsheg_root).expanduser().resolve()
            )
            if (
                observed_diffsheg_head is not None
                and observed_diffsheg_head != DIFFSHEG_REFERENCE_COMMIT
            ):
                report["warnings"].append(
                    "the supplied DiffSHEG checkout is at "
                    f"{observed_diffsheg_head}, while this adapter was audited "
                    f"against {DIFFSHEG_REFERENCE_COMMIT}; embedded metric "
                    "logic is unchanged, but verify the statistics and AE "
                    "asset hashes before comparison."
                )
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else prediction_dir / "diffsheg_show_metrics.json"
        )
        if args.validate_input_only:
            report["status"] = "input_validation_passed"
            _write_json(output, report)
            print(f"[done] validation report: {output}")
            return 0

        try:
            import torch
        except ImportError as exc:
            raise ProtocolError(
                "PyTorch is required for DiffSHEG AE evaluation"
            ) from exc
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ProtocolError(
                    f"requested {device}, but torch.cuda.is_available() is false"
                )
            index = device.index if device.index is not None else 0
            torch.cuda.set_device(index)
            device = torch.device(f"cuda:{index}")

        metrics, diagnostics = compute_parameter_metrics(validation, stats)
        checkpoint_provenance: dict[str, Any] = {}
        for metric_name, component, input_dim, _ in FEATURE_SPECS:
            print(
                f"[{metric_name.upper()}] loading {checkpoints[metric_name]}",
                flush=True,
            )
            model, checkpoint_info = load_embedding_model(
                torch=torch,
                checkpoint_path=checkpoints[metric_name],
                input_dim=input_dim,
                device=device,
            )
            predicted, ground_truth = extract_features(
                torch=torch,
                model=model,
                device=device,
                validation=validation,
                stats=stats,
                component=component,
                batch_size=args.batch_size,
            )
            metrics[metric_name] = frechet_distance(
                predicted, ground_truth
            )
            checkpoint_info["feature_count"] = int(predicted.shape[0])
            checkpoint_provenance[metric_name] = checkpoint_info
            del model, predicted, ground_truth
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ba_provenance: dict[str, Any] | None = None
        if not args.skip_ba:
            if (
                audio_paths is None
                or talkshow_root is None
                or smplx_path is None
            ):
                raise AssertionError("validated BA inputs disappeared")
            metrics["ba"], ba_provenance = compute_ba(
                validation=validation,
                audio_paths=audio_paths,
                talkshow_root=talkshow_root,
                smplx_path=smplx_path,
                device_name=str(device),
            )

        for key, value in metrics.items():
            if not np.isfinite(float(value)):
                raise ProtocolError(f"metric {key} is not finite: {value}")
        report["status"] = "ok"
        report["metrics"] = metrics
        report["diagnostics"] = diagnostics
        evaluator_path = Path(__file__).resolve()
        repository_root = evaluator_path.parents[1]
        report["provenance"] = {
            "evaluator": {
                "path": str(evaluator_path),
                "sha256": _sha256_file(evaluator_path),
                "repository_root": str(repository_root),
                "repository_git_head": _git_head(repository_root),
            },
            "diffsheg_root": (
                {
                    "path": str(
                        Path(args.diffsheg_root).expanduser().resolve()
                    ),
                    "git_head": _git_head(
                        Path(args.diffsheg_root).expanduser().resolve()
                    ),
                }
                if args.diffsheg_root
                else None
            ),
            "autoencoders": checkpoint_provenance,
            "ba": ba_provenance,
            "device": str(device),
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": getattr(torch, "__version__", None),
                "scipy": _distribution_version("scipy"),
                "smplx": _distribution_version("smplx"),
                "librosa": _distribution_version("librosa"),
                "torchaudio": _distribution_version("torchaudio"),
            },
        }
        _write_json(output, report)
        print("[metrics]")
        for key in (
            "fmd",
            "fed",
            "expression_diversity",
            "fgd",
            "ba",
            "pcm",
            "gesture_diversity",
        ):
            if key in metrics:
                print(f"  {key}: {metrics[key]}")
        print(f"[done] DiffSHEG SHOW metrics: {output}")
        return 0
    except ProtocolError as exc:
        parser.exit(2, f"[abort] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
