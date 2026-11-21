"""Pure-NumPy, speaker-level gesture-style metrics for offline evaluation.

The feature definition is shared with ``scripts/fig6_acf_analysis.py``:

* extension: mean elbow/wrist distance from the neck;
* span: mean distance between the wrists;
* height: mean wrist height relative to the neck;
* laterality: right-minus-left arm extension;
* occupancy: log trace of the two-wrist spatial covariance.

The evaluator intentionally uses only NumPy.  It consumes SMPL-X joint
positions that have already been placed in a canonical facing direction by
the caller, removes root translation, and normalizes body scale by shoulder
width.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


STYLE_FEATURE_NAMES = (
    "extension",
    "span",
    "height",
    "laterality",
    "occupancy",
)
UPPER_JOINT_INDICES = (12, 16, 17, 18, 19, 20, 21)
DEFAULT_STYLE_WINDOW = 30
DEFAULT_ACF_LAG_START = 100
DEFAULT_ACF_LAG_END = 200
DEFAULT_MIN_ACF_PAIRS = 30
DEFAULT_SWD_PROJECTIONS = 256
DEFAULT_PROJECTION_SEED = 20260426
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CONFIDENCE = 0.95


def compute_style_features(
    upper_body_positions: np.ndarray,
    W: int = DEFAULT_STYLE_WINDOW,
) -> np.ndarray:
    """Compute the five pre-defined style descriptors on trailing windows.

    Args:
        upper_body_positions: ``(T, 7, 3)`` normalized positions ordered as
            neck, L/R shoulder, L/R elbow, L/R wrist.
        W: Trailing-window width in frames.  The capitalized name preserves
            the existing Fig. 6 helper API.

    Returns:
        ``(T - W + 1, 5)`` style feature matrix.
    """

    q = np.asarray(upper_body_positions, dtype=np.float64)
    if q.ndim != 3 or q.shape[1:] != (7, 3):
        raise ValueError(
            "upper_body_positions must have shape (T, 7, 3), "
            f"got {q.shape}"
        )
    if W <= 1:
        raise ValueError(f"W must be greater than one, got {W}")
    if len(q) < W:
        return np.empty((0, len(STYLE_FEATURE_NAMES)), dtype=np.float32)
    if not np.isfinite(q).all():
        raise ValueError("upper_body_positions contains NaN or Inf")

    def moving_sum(values: np.ndarray) -> np.ndarray:
        prefix_shape = (1, *values.shape[1:])
        prefix = np.concatenate(
            [
                np.zeros(prefix_shape, dtype=np.float64),
                np.cumsum(values, axis=0, dtype=np.float64),
            ],
            axis=0,
        )
        return prefix[W:] - prefix[:-W]

    neck = q[:, 0]
    left_elbow, right_elbow = q[:, 3], q[:, 4]
    left_wrist, right_wrist = q[:, 5], q[:, 6]
    endpoints = np.stack(
        [left_elbow, right_elbow, left_wrist, right_wrist],
        axis=1,
    )

    extension_frame = np.linalg.norm(
        endpoints - neck[:, None, :],
        axis=-1,
    ).mean(axis=1)
    span_frame = np.linalg.norm(left_wrist - right_wrist, axis=-1)
    height_frame = 0.5 * (
        left_wrist[:, 1]
        - neck[:, 1]
        + right_wrist[:, 1]
        - neck[:, 1]
    )
    laterality_frame = (
        np.linalg.norm(right_wrist - neck, axis=-1)
        - np.linalg.norm(left_wrist - neck, axis=-1)
    )

    # np.cov over the 2W wrist points can be written with rolling first and
    # second moments. This is mathematically the same sample-covariance trace
    # as the original Fig. 6 loop, without one covariance call per frame.
    wrist_sum = moving_sum(left_wrist + right_wrist)
    wrist_square_sum = moving_sum(
        np.sum(
            np.square(left_wrist) + np.square(right_wrist),
            axis=-1,
        )
    )
    wrist_point_count = 2 * W
    occupancy_variance = (
        wrist_square_sum
        - np.sum(np.square(wrist_sum), axis=-1) / wrist_point_count
    ) / (wrist_point_count - 1)

    features = np.column_stack(
        [
            moving_sum(extension_frame) / W,
            moving_sum(span_frame) / W,
            moving_sum(height_frame) / W,
            moving_sum(laterality_frame) / W,
            np.log(np.maximum(occupancy_variance, 0.0) + 1e-6),
        ]
    )
    return features.astype(np.float32, copy=False)


def joint_positions_to_style_features(
    joint_positions: np.ndarray,
    window: int = DEFAULT_STYLE_WINDOW,
    *,
    shoulder_width: float | None = None,
) -> np.ndarray:
    """Convert canonical SMPL-X joints into body-scale-normalized features.

    ``shoulder_width`` can be supplied to place a prediction and its reference
    on the same scale.  The evaluator derives this anchor from the reference
    clip so that a generated pose cannot normalize away its own amplitude
    error.
    """

    positions = np.asarray(joint_positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "joint_positions must have shape (T, J, 3), "
            f"got {positions.shape}"
        )
    if positions.shape[1] <= max(UPPER_JOINT_INDICES):
        raise ValueError(
            "joint_positions does not contain all required SMPL-X joints: "
            f"J={positions.shape[1]}"
        )
    if not np.isfinite(positions).all():
        raise ValueError("joint_positions contains NaN or Inf")

    root_local = positions - positions[:, 0:1]
    upper = root_local[:, list(UPPER_JOINT_INDICES)]
    if shoulder_width is None:
        shoulder_width = float(
            np.linalg.norm(upper[:, 2] - upper[:, 1], axis=-1).mean()
        )
    else:
        shoulder_width = float(shoulder_width)
    if not np.isfinite(shoulder_width) or shoulder_width < 1e-6:
        raise ValueError(
            f"invalid mean shoulder width: {shoulder_width!r}"
        )
    return compute_style_features(upper / shoulder_width, W=window)


def _projection_directions(
    dimension: int,
    n_projections: int,
    seed: int,
) -> np.ndarray:
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if n_projections <= 0:
        raise ValueError(
            f"n_projections must be positive, got {n_projections}"
        )
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    return directions


def _wasserstein_2_from_samples(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Empirical one-dimensional W2 using a shared quantile grid."""

    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n_quantiles = min(len(a), len(b))
    probabilities = np.linspace(0.0, 1.0, n_quantiles)
    qa = np.quantile(a, probabilities)
    qb = np.quantile(b, probabilities)
    return float(np.sqrt(np.mean(np.square(qa - qb))))


def sliced_wasserstein_2(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_projections: int = DEFAULT_SWD_PROJECTIONS,
    seed: int = DEFAULT_PROJECTION_SEED,
) -> float:
    """Sliced Wasserstein-2 distance between two empirical distributions."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            "Sliced W2 inputs must be 2-D with the same feature dimension, "
            f"got {a.shape} and {b.shape}"
        )
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Sliced W2 inputs contain NaN or Inf")

    directions = _projection_directions(
        a.shape[1],
        n_projections,
        seed,
    )
    projected_a = a @ directions.T
    projected_b = b @ directions.T
    n_quantiles = min(len(projected_a), len(projected_b))
    probabilities = np.linspace(0.0, 1.0, n_quantiles)
    quantiles_a = np.quantile(projected_a, probabilities, axis=0)
    quantiles_b = np.quantile(projected_b, probabilities, axis=0)
    w2_squared = np.mean(np.square(quantiles_a - quantiles_b), axis=0)
    return float(np.sqrt(np.mean(w2_squared)))


def multivariate_trace_acf(
    features: np.ndarray,
    max_lag: int,
    *,
    min_pairs: int = DEFAULT_MIN_ACF_PAIRS,
) -> np.ndarray:
    """Compute trace-ACF, matching the definition used by the Fig. 6 script."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"features must be 2-D, got {values.shape}")
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative, got {max_lag}")
    if min_pairs <= 0:
        raise ValueError(f"min_pairs must be positive, got {min_pairs}")
    if not np.isfinite(values).all():
        raise ValueError("features contains NaN or Inf")

    result = np.full(max_lag + 1, np.nan, dtype=np.float64)
    if len(values) == 0:
        return result
    centered = values - values.mean(axis=0)
    variance_trace = float(np.sum(np.square(centered)) / len(centered))
    if variance_trace < 1e-12:
        return result
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        pair_count = len(centered) - lag
        if pair_count < min_pairs:
            continue
        covariance_trace = float(
            np.sum(centered[:pair_count] * centered[lag:]) / pair_count
        )
        result[lag] = covariance_trace / variance_trace
    return result


def _aggregate_acf(
    feature_sequences: Sequence[np.ndarray],
    max_lag: int,
    min_pairs: int,
) -> np.ndarray:
    weighted_sum = np.zeros(max_lag + 1, dtype=np.float64)
    weight_sum = np.zeros(max_lag + 1, dtype=np.float64)
    for features in feature_sequences:
        acf = multivariate_trace_acf(
            features,
            max_lag,
            min_pairs=min_pairs,
        )
        for lag, value in enumerate(acf):
            if not np.isfinite(value):
                continue
            weight = max(len(features) - lag, 0)
            if weight <= 0:
                continue
            weighted_sum[lag] += weight * value
            weight_sum[lag] += weight
    result = np.full(max_lag + 1, np.nan, dtype=np.float64)
    valid = weight_sum > 0
    result[valid] = weighted_sum[valid] / weight_sum[valid]
    return result


def _finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _summarize_speakers(
    values_by_speaker: Mapping[int, float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    values = np.asarray(
        [
            value
            for _, value in sorted(values_by_speaker.items())
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )
    if len(values) == 0:
        return {
            "mean": None,
            "speaker_std": None,
            "bootstrap_std": None,
            "ci_low": None,
            "ci_high": None,
            "n_speakers": 0,
        }

    summary: dict[str, Any] = {
        "mean": float(values.mean()),
        "speaker_std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "bootstrap_std": None,
        "ci_low": None,
        "ci_high": None,
        "n_speakers": int(len(values)),
    }
    if len(values) < 2 or bootstrap_samples <= 0:
        return summary

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(values),
        size=(bootstrap_samples, len(values)),
    )
    bootstrap_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    summary.update(
        {
            "bootstrap_std": float(bootstrap_means.std(ddof=1)),
            "ci_low": float(np.quantile(bootstrap_means, alpha)),
            "ci_high": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        }
    )
    return summary


@dataclass(frozen=True)
class _StyleClip:
    clip_id: str
    speaker_id: int
    prediction: np.ndarray
    reference: np.ndarray


class GestureStyleAccumulator:
    """Accumulate clips and compute speaker-macro style metrics."""

    def __init__(
        self,
        *,
        window: int = DEFAULT_STYLE_WINDOW,
        acf_lag_start: int = DEFAULT_ACF_LAG_START,
        acf_lag_end: int = DEFAULT_ACF_LAG_END,
        min_acf_pairs: int = DEFAULT_MIN_ACF_PAIRS,
        n_projections: int = DEFAULT_SWD_PROJECTIONS,
        projection_seed: int = DEFAULT_PROJECTION_SEED,
        bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
        bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        if window <= 1:
            raise ValueError("window must be greater than one")
        if acf_lag_start < 1 or acf_lag_end < acf_lag_start:
            raise ValueError("invalid ACF lag interval")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be strictly between zero and one")
        self.window = int(window)
        self.acf_lag_start = int(acf_lag_start)
        self.acf_lag_end = int(acf_lag_end)
        self.min_acf_pairs = int(min_acf_pairs)
        self.n_projections = int(n_projections)
        self.projection_seed = int(projection_seed)
        self.bootstrap_samples = int(bootstrap_samples)
        self.bootstrap_seed = int(bootstrap_seed)
        self.confidence = float(confidence)
        self._clips: list[_StyleClip] = []
        self._skipped: Counter[str] = Counter()

    def update(
        self,
        prediction_positions: np.ndarray,
        reference_positions: np.ndarray,
        *,
        speaker_id: int,
        clip_id: str,
    ) -> bool:
        """Add one aligned prediction/reference clip; return whether accepted."""

        prediction = np.asarray(prediction_positions)
        reference = np.asarray(reference_positions)
        if prediction.ndim != 3 or reference.ndim != 3:
            self._skipped["invalid_position_shape"] += 1
            return False
        required_joint_count = max(UPPER_JOINT_INDICES) + 1
        if (
            prediction.shape[-1] != 3
            or reference.shape[-1] != 3
            or prediction.shape[1] < required_joint_count
            or reference.shape[1] < required_joint_count
        ):
            self._skipped["invalid_position_shape"] += 1
            return False
        length = min(len(prediction), len(reference))
        if length < self.window:
            self._skipped["shorter_than_style_window"] += 1
            return False
        try:
            reference_root_local = reference[:length] - reference[:length, 0:1]
            reference_upper = reference_root_local[
                :, list(UPPER_JOINT_INDICES)
            ]
            reference_shoulder_width = float(
                np.linalg.norm(
                    reference_upper[:, 2] - reference_upper[:, 1],
                    axis=-1,
                ).mean()
            )
            prediction_features = joint_positions_to_style_features(
                prediction[:length],
                window=self.window,
                shoulder_width=reference_shoulder_width,
            )
            reference_features = joint_positions_to_style_features(
                reference[:length],
                window=self.window,
                shoulder_width=reference_shoulder_width,
            )
        except ValueError:
            self._skipped["invalid_joint_positions"] += 1
            return False
        if len(prediction_features) == 0 or len(reference_features) == 0:
            self._skipped["empty_style_features"] += 1
            return False
        self._clips.append(
            _StyleClip(
                clip_id=str(clip_id),
                speaker_id=int(speaker_id),
                prediction=prediction_features,
                reference=reference_features,
            )
        )
        return True

    def compute(self) -> dict[str, Any]:
        """Return point estimates plus speaker-level bootstrap summaries."""

        empty_result = {
            "style_swd": None,
            "style_acf_error": None,
            "feature_w2": {
                name: None for name in STYLE_FEATURE_NAMES
            },
        }
        settings = {
            "feature_names": list(STYLE_FEATURE_NAMES),
            "window_frames": self.window,
            "acf_lag_start": self.acf_lag_start,
            "acf_lag_end": self.acf_lag_end,
            "min_acf_pairs": self.min_acf_pairs,
            "swd_projections": self.n_projections,
            "projection_seed": self.projection_seed,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence": self.confidence,
            "aggregation": "speaker_macro",
            "body_scale_source": "paired_clip_gt_mean_shoulder_width",
            "swd_scaler_source": "evaluation_gt_global",
            "acf_scaler_source": "per_speaker_gt",
        }
        if not self._clips:
            return {
                **empty_result,
                "details": {
                    "status": "insufficient_data",
                    "settings": settings,
                    "n_valid_clips": 0,
                    "n_valid_speakers": 0,
                    "skipped_clips": dict(sorted(self._skipped.items())),
                    "scaler": None,
                    "summaries": {},
                    "per_speaker": {},
                },
            }

        all_reference = np.concatenate(
            [clip.reference for clip in self._clips],
            axis=0,
        ).astype(np.float64)
        scaler_mean = all_reference.mean(axis=0)
        scaler_std = all_reference.std(axis=0)
        scaler_std[scaler_std < 1e-8] = 1.0

        grouped: dict[int, list[_StyleClip]] = defaultdict(list)
        for clip in self._clips:
            grouped[clip.speaker_id].append(clip)

        swd_by_speaker: dict[int, float] = {}
        acf_by_speaker: dict[int, float] = {}
        feature_by_speaker: dict[str, dict[int, float]] = {
            name: {} for name in STYLE_FEATURE_NAMES
        }
        per_speaker: dict[str, Any] = {}

        for speaker_id in sorted(grouped):
            clips = grouped[speaker_id]
            prediction_sequences = [
                (clip.prediction.astype(np.float64) - scaler_mean) / scaler_std
                for clip in clips
            ]
            reference_sequences = [
                (clip.reference.astype(np.float64) - scaler_mean) / scaler_std
                for clip in clips
            ]
            prediction_all = np.concatenate(prediction_sequences, axis=0)
            reference_all = np.concatenate(reference_sequences, axis=0)

            speaker_swd = sliced_wasserstein_2(
                prediction_all,
                reference_all,
                n_projections=self.n_projections,
                seed=self.projection_seed,
            )
            swd_by_speaker[speaker_id] = speaker_swd

            speaker_features: dict[str, float] = {}
            for feature_index, feature_name in enumerate(STYLE_FEATURE_NAMES):
                distance = _wasserstein_2_from_samples(
                    prediction_all[:, feature_index],
                    reference_all[:, feature_index],
                )
                feature_by_speaker[feature_name][speaker_id] = distance
                speaker_features[feature_name] = distance

            speaker_reference_raw = np.concatenate(
                [clip.reference for clip in clips],
                axis=0,
            ).astype(np.float64)
            speaker_acf_mean = speaker_reference_raw.mean(axis=0)
            speaker_acf_std = speaker_reference_raw.std(axis=0)
            speaker_acf_std[speaker_acf_std < 1e-8] = 1.0
            prediction_acf_sequences = [
                (
                    clip.prediction.astype(np.float64)
                    - speaker_acf_mean
                )
                / speaker_acf_std
                for clip in clips
            ]
            reference_acf_sequences = [
                (
                    clip.reference.astype(np.float64)
                    - speaker_acf_mean
                )
                / speaker_acf_std
                for clip in clips
            ]
            prediction_acf = _aggregate_acf(
                prediction_acf_sequences,
                self.acf_lag_end,
                self.min_acf_pairs,
            )
            reference_acf = _aggregate_acf(
                reference_acf_sequences,
                self.acf_lag_end,
                self.min_acf_pairs,
            )
            lag_slice = slice(self.acf_lag_start, self.acf_lag_end + 1)
            valid_lags = (
                np.isfinite(prediction_acf[lag_slice])
                & np.isfinite(reference_acf[lag_slice])
            )
            acf_error: float | None = None
            if np.any(valid_lags):
                differences = np.abs(
                    prediction_acf[lag_slice][valid_lags]
                    - reference_acf[lag_slice][valid_lags]
                )
                acf_error = float(differences.mean())
                acf_by_speaker[speaker_id] = acf_error

            per_speaker[str(speaker_id)] = {
                "n_clips": len(clips),
                "n_prediction_style_frames": int(len(prediction_all)),
                "n_reference_style_frames": int(len(reference_all)),
                "style_swd": _finite_or_none(speaker_swd),
                "style_acf_error": _finite_or_none(acf_error),
                "acf_valid_lags": int(valid_lags.sum()),
                "feature_w2": {
                    name: _finite_or_none(value)
                    for name, value in speaker_features.items()
                },
            }

        summaries: dict[str, Any] = {}
        summaries["style_swd"] = _summarize_speakers(
            swd_by_speaker,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
            confidence=self.confidence,
        )
        summaries["style_acf_error"] = _summarize_speakers(
            acf_by_speaker,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed + 1,
            confidence=self.confidence,
        )
        feature_results: dict[str, float | None] = {}
        for feature_index, feature_name in enumerate(STYLE_FEATURE_NAMES):
            summary = _summarize_speakers(
                feature_by_speaker[feature_name],
                bootstrap_samples=self.bootstrap_samples,
                bootstrap_seed=self.bootstrap_seed + 2 + feature_index,
                confidence=self.confidence,
            )
            summaries[f"{feature_name}_w2"] = summary
            feature_results[feature_name] = summary["mean"]

        style_swd = summaries["style_swd"]["mean"]
        style_acf_error = summaries["style_acf_error"]["mean"]
        status = (
            "ok"
            if style_swd is not None and style_acf_error is not None
            else "partial"
        )
        return {
            "style_swd": style_swd,
            "style_acf_error": style_acf_error,
            "feature_w2": feature_results,
            "details": {
                "status": status,
                "settings": settings,
                "n_valid_clips": len(self._clips),
                "n_valid_speakers": len(grouped),
                "skipped_clips": dict(sorted(self._skipped.items())),
                "scaler": {
                    "mean": scaler_mean.tolist(),
                    "std": scaler_std.tolist(),
                },
                "summaries": summaries,
                "per_speaker": per_speaker,
            },
        }
