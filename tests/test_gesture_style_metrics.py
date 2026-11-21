from __future__ import annotations

import numpy as np

from scripts.gesture_style_metrics import GestureStyleAccumulator


def _synthetic_upper_body(
    *,
    frames: int = 360,
    frequency: float = 1.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Build a minimal, valid 55-joint SMPL-X position sequence."""

    phase = np.linspace(0.0, frequency * 8.0 * np.pi, frames)
    positions = np.zeros((frames, 55, 3), dtype=np.float32)
    positions[:, 12] = (0.0, 1.45, 0.0)
    positions[:, 16] = (-0.30, 1.35, 0.0)
    positions[:, 17] = (0.30, 1.35, 0.0)
    positions[:, 18, 0] = -0.52 - 0.06 * amplitude * np.sin(phase)
    positions[:, 18, 1] = 1.05 + 0.08 * amplitude * np.cos(phase)
    positions[:, 19, 0] = 0.52 + 0.05 * amplitude * np.cos(phase)
    positions[:, 19, 1] = 1.05 + 0.07 * amplitude * np.sin(phase)
    positions[:, 20, 0] = -0.76 - 0.18 * amplitude * np.sin(phase)
    positions[:, 20, 1] = 0.82 + 0.20 * amplitude * np.cos(phase)
    positions[:, 20, 2] = 0.12 * amplitude * np.sin(phase * 0.5)
    positions[:, 21, 0] = 0.76 + 0.16 * amplitude * np.cos(phase)
    positions[:, 21, 1] = 0.82 + 0.17 * amplitude * np.sin(phase)
    positions[:, 21, 2] = 0.10 * amplitude * np.cos(phase * 0.5)
    return positions


def test_style_metrics_are_zero_for_a_match_and_detect_a_style_change() -> None:
    reference = _synthetic_upper_body()

    matched = GestureStyleAccumulator(
        n_projections=64,
        bootstrap_samples=0,
    )
    assert matched.update(
        reference.copy(),
        reference,
        speaker_id=2,
        clip_id="matched",
    )
    matched_result = matched.compute()
    assert matched_result["details"]["status"] == "ok"
    assert abs(matched_result["style_swd"]) < 1e-12
    assert abs(matched_result["style_acf_error"]) < 1e-12

    changed = GestureStyleAccumulator(
        n_projections=64,
        bootstrap_samples=0,
    )
    assert changed.update(
        _synthetic_upper_body(frequency=2.5, amplitude=1.8),
        reference,
        speaker_id=2,
        clip_id="changed",
    )
    changed_result = changed.compute()
    assert changed_result["style_swd"] > 0.01
    assert changed_result["style_acf_error"] > 0.01
