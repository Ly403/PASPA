# Terminal-window FGD@T evaluator (PASPA E1 long-horizon style preservation).
#
# For each rollout length T in --eval_at_T, computes FGD/BC/L1Div/LVD/MSE on
# the *terminal* window [(T - W) * fps, T * fps] of every clip, where W is
# --terminal_window_sec. Outputs both JSON (full result) and a sibling CSV
# (T_sec,fgd,bc,...) for downstream plotting (FGD@T curves).
#
# Default: a single common subset is used across all T values (filter by
# max(T)*fps), so cross-T FGD differences reflect rollout degradation only —
# not population shift. Use --per_T_subset to filter independently per T.
#
# 用法与 emage_eval.py 平行，输入格式完全一致 (res_<id>.npz / gt_<id>.npz /
# gt_sr16000_<id>.wav)。
#
# 示例:
#   python scripts/emage_eval_temporal.py \
#       -i /path/to/npz -ia /path/to/audio -o /path/to/output.json \
#       --eval_at_T 10 30 60 --terminal_window_sec 10
#
# 注意：替换了原先的 sliding / cumulative 模式（已废弃，不向后兼容）。

import os
import sys
import json
import math
import argparse
import wget
from pathlib import Path
import smplx
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from collections import defaultdict

sys.path.append("..")
from tools.EMAGE.emage_evaltools.mertic import FGD, BC, L1div, LVDFace, MSEFace
import diffusion.utils.rotation_conversions as rc
# scipy.stats is already an indirect dependency (mertic.py uses scipy.linalg).
from scipy.stats import norm as _norm

REPO_ROOT = Path(__file__).resolve().parents[1]

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Terminal-window FGD@T evaluation using standard EMAGE metrics. "
                "For each T in --eval_at_T, compute metrics on the [T-W, T]s window."
)
parser.add_argument("-i", "--input_path", type=str, required=True)
parser.add_argument("-o", "--output_path", type=str, required=True,
                    help="Path to JSON output. A sibling CSV (.csv) is also written.")
parser.add_argument("-ia", "--input_audio_path", type=str, required=True)
parser.add_argument("-dp", "--download_path", type=str,
                    default=os.environ.get(
                        "EMAGE_EVALTOOLS_DIR",
                        str(REPO_ROOT / "tools" / "EMAGE" / "emage_evaltools"),
                    ))
parser.add_argument("-sf", "--split_file", type=str,
                    default=os.environ.get("BEAT2_SPLIT_FILE"))
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-p", "--person", type=int, nargs="+",
                    default=list(range(30)))

# Terminal-window 模式
parser.add_argument("--eval_at_T", type=int, nargs="+", default=[10, 30, 60],
                    help="Rollout lengths T (seconds) at which to evaluate. "
                         "Each T yields one terminal window [T-W, T]s. Default: 10 30 60.")
parser.add_argument("--terminal_window_sec", type=int, default=10,
                    help="Width W (seconds) of the terminal window. Default: 10.")
parser.add_argument("--per_T_subset", action="store_true", default=False,
                    help="When set, filter clips per T (each T uses its own clip "
                         "subset). Default: filter once by max(T)*fps so all T "
                         "values use the same common subset (cross-T comparable).")

# 公共参数
parser.add_argument("--max_drop_ratio", type=float, default=0.8,
                    help="If subset filtering drops more than this fraction of clips, abort.")
parser.add_argument("--min_samples", type=int, default=5,
                    help="Subset must have at least this many clips, else abort/skip.")
parser.add_argument("--pose_fps", type=int, default=30)
parser.add_argument("--only_fgd", action="store_true", default=False)

# Bootstrap CI on the clip dimension. Captures sampling variability over the
# evaluation set; does NOT capture model-stochasticity (sampler seed) variance.
# Cost: a few seconds per window. Off by default.
parser.add_argument("--n_bootstrap", type=int, default=0,
                    help="Number of clip-bootstrap iterations per metric. "
                         "0 = disabled (no CI columns). Typical value: 1000.")
parser.add_argument("--bootstrap_seed", type=int, default=42,
                    help="RNG seed for bootstrap reproducibility.")
parser.add_argument("--ci_level", type=float, default=0.95,
                    help="Two-sided confidence interval level (e.g. 0.95 -> [2.5%%, 97.5%%]).")
parser.add_argument("--ci_method", type=str, default="percentile",
                    choices=["percentile", "bca", "wald"],
                    help="Bootstrap CI method. "
                         "percentile = raw [a/2, 1-a/2] percentiles of bootstrap "
                         "samples (default, fast). "
                         "bca = bias-corrected and accelerated, requires N "
                         "additional jackknife computations per metric per window; "
                         "more rigorous but can fall back to percentile when bias "
                         "is too extreme (z0 = ±inf). "
                         "wald = θ̂ ± z_{α/2}·σ_bootstrap, symmetric around the "
                         "point estimate; cheapest method (no extra cost beyond "
                         "the bootstrap loop), guarantees the line lies inside "
                         "the CI band by construction; assumes the bootstrap "
                         "distribution is approximately Gaussian.")

config = parser.parse_args()
if not config.split_file:
    parser.error("--split_file is required (or set BEAT2_SPLIT_FILE)")
config.download_path = os.path.abspath(
    os.path.expanduser(config.download_path)
).rstrip(os.sep) + os.sep
device = torch.device(config.device)
download_path = config.download_path
pose_fps = config.pose_fps
bootstrap_enabled = config.n_bootstrap > 0
rng = np.random.default_rng(config.bootstrap_seed) if bootstrap_enabled else None
if bootstrap_enabled:
    if not (0 < config.ci_level < 1):
        raise ValueError(f"--ci_level must be in (0, 1), got {config.ci_level}")

# ──────────────────────────────────────────────────────────────
# SMPLX model (reuse emage_eval.py logic)
# ──────────────────────────────────────────────────────────────
smplx_model_dir = os.path.join(download_path, "smplx_models", "smplx")
if not os.path.exists(smplx_model_dir):
    smplx_model_file_path = os.path.join(smplx_model_dir, "SMPLX_NEUTRAL_2020.npz")
    os.makedirs(smplx_model_dir, exist_ok=True)
    if not os.path.exists(smplx_model_file_path):
        print(f"Downloading {smplx_model_file_path}")
        wget.download(
            "https://huggingface.co/spaces/H-Liu1997/EMAGE/resolve/main/EMAGE/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
            smplx_model_file_path,
        )

smplx_model = smplx.create(
    download_path + "smplx_models/",
    model_type="smplx",
    gender="NEUTRAL_2020",
    use_face_contour=False,
    num_betas=300,
    num_expression_coeffs=100,
    ext="npz",
    use_pca=False,
).eval()


def get_motion_rep_numpy(
    poses_np, pose_fps=30, device="cuda", expressions=None,
    expression_only=False, betas=None,
):
    """与 emage_eval.py 中完全一致的 motion representation 计算"""
    global smplx_model
    smplx_model = smplx_model.to(device)
    n = poses_np.shape[0]
    poses_ts = torch.from_numpy(poses_np).float().to(device).unsqueeze(0)
    poses_ts_reshaped = poses_ts.reshape(-1, 165)
    betas_t = (
        torch.zeros(n, 300, device=device)
        if betas is None
        else torch.from_numpy(betas).to(device).unsqueeze(0).repeat(n, 1)
    )

    if expressions is not None and expression_only:
        expressions = torch.from_numpy(expressions).float().to(device)
        output = smplx_model(
            betas=betas_t,
            transl=torch.zeros(n, 3, device=device),
            expression=expressions,
            jaw_pose=poses_ts_reshaped[:, 22 * 3 : 23 * 3],
            global_orient=torch.zeros(n, 3, device=device),
            body_pose=torch.zeros(n, 21 * 3, device=device),
            left_hand_pose=torch.zeros(n, 15 * 3, device=device),
            right_hand_pose=torch.zeros(n, 15 * 3, device=device),
            return_joints=True,
            leye_pose=torch.zeros(n, 3, device=device),
            reye_pose=torch.zeros(n, 3, device=device),
        )
        joints = output["vertices"].detach().cpu().numpy().reshape(n, -1)
        return {"vertices": joints}

    output = smplx_model(
        betas=betas_t,
        transl=torch.zeros(n, 3, device=device),
        expression=torch.zeros(n, 100, device=device),
        jaw_pose=torch.zeros(n, 3, device=device),
        global_orient=torch.zeros(n, 3, device=device),
        body_pose=poses_ts_reshaped[:, 3 : 21 * 3 + 3],
        left_hand_pose=poses_ts_reshaped[:, 25 * 3 : 40 * 3],
        right_hand_pose=poses_ts_reshaped[:, 40 * 3 : 55 * 3],
        return_joints=True,
        leye_pose=torch.zeros(n, 3, device=device),
        reye_pose=torch.zeros(n, 3, device=device),
    )
    joints = output["joints"].detach().cpu().numpy().reshape(n, 127, 3)[:, :55, :]
    dt = 1 / pose_fps
    init_vel = (joints[1:2] - joints[0:1]) / dt
    middle_vel = (joints[2:] - joints[:-2]) / (2 * dt)
    final_vel = (joints[-1:] - joints[-2:-1]) / dt
    vel = np.concatenate([init_vel, middle_vel, final_vel], axis=0)
    position = joints
    return {"position": position, "velocity": vel}


# ──────────────────────────────────────────────────────────────
# Bootstrap helpers (clip-level resampling for CI estimation)
# ──────────────────────────────────────────────────────────────
def _percentile_ci(samples, ci_level):
    """Return (lo, hi, std) from a 1-D numpy array of bootstrap statistics."""
    samples = np.asarray(samples, dtype=np.float64)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - float(ci_level)) / 2.0
    lo = float(np.percentile(finite, 100.0 * alpha))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha)))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return lo, hi, std


def _bca_ci(samples, theta_hat, jackknife_vals, ci_level):
    """Bias-corrected and accelerated bootstrap CI (Efron, 1987).

    Inputs:
      samples         : 1-D array of B bootstrap statistics
      theta_hat       : the original point estimate on the full (un-resampled) data
      jackknife_vals  : 1-D array of N leave-one-clip-out estimates
      ci_level        : two-sided confidence level (e.g. 0.95)

    The CI is `[θ*_{α₁}, θ*_{α₂}]` where the adjusted percentiles are:
        α₁ = Φ(z₀ + (z₀ + z_{α/2})  / (1 - a·(z₀ + z_{α/2})))
        α₂ = Φ(z₀ + (z₀ + z_{1-α/2})/ (1 - a·(z₀ + z_{1-α/2})))
    with bias correction `z₀ = Φ⁻¹(P(θ̂* < θ̂))` and acceleration `a` from
    jackknife skewness. On numerical edge cases (z₀ = ±∞ or
    1 - a(z₀+z) ≤ 0) we fall back to percentile and warn.
    """
    samples = np.asarray(samples, dtype=np.float64)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")

    # Bias correction z₀
    p_lt = float(np.mean(finite < theta_hat))
    if p_lt <= 0.0 or p_lt >= 1.0:
        print(f"  WARN: BCa z0 is infinite (P(θ*<θ̂)={p_lt:.3f}); "
              f"falling back to percentile CI.")
        return _percentile_ci(samples, ci_level)
    z0 = float(_norm.ppf(p_lt))

    # Acceleration a via jackknife skewness
    jack = np.asarray(jackknife_vals, dtype=np.float64)
    jack = jack[np.isfinite(jack)]
    if jack.size < 2:
        print("  WARN: BCa needs ≥2 finite jackknife values; "
              "falling back to percentile CI.")
        return _percentile_ci(samples, ci_level)
    jack_mean = jack.mean()
    diffs = jack_mean - jack
    num = np.sum(diffs ** 3)
    den = 6.0 * (np.sum(diffs ** 2)) ** 1.5
    a = float(num / den) if den > 0 else 0.0

    # Adjusted percentiles
    alpha = (1.0 - float(ci_level)) / 2.0
    z_lo = float(_norm.ppf(alpha))
    z_hi = float(_norm.ppf(1.0 - alpha))

    def _adjust(z):
        denom = 1.0 - a * (z0 + z)
        if denom <= 0:
            return None
        return float(_norm.cdf(z0 + (z0 + z) / denom))

    a_lo = _adjust(z_lo)
    a_hi = _adjust(z_hi)
    if a_lo is None or a_hi is None:
        print(f"  WARN: BCa denominator non-positive (z0={z0:.3f}, a={a:.3f}); "
              f"falling back to percentile CI.")
        return _percentile_ci(samples, ci_level)
    if a_lo > a_hi:
        a_lo, a_hi = a_hi, a_lo  # defensive (shouldn't happen with valid z₀, a)

    lo = float(np.percentile(finite, 100.0 * a_lo))
    hi = float(np.percentile(finite, 100.0 * a_hi))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return lo, hi, std


def _wald_ci(samples, theta_hat, ci_level):
    """Symmetric Wald CI centered on theta_hat using bootstrap std.

    CI = [theta_hat - z·σ, theta_hat + z·σ]
    where z = norm.ppf(1 - (1 - ci_level)/2) (e.g. ci_level=0.95 → z≈1.96).

    By construction theta_hat is at the center of the interval, so the line
    will always be inside the shaded CI band on plots. Trade-off: assumes
    the bootstrap distribution is approximately Gaussian; bias and skewness
    are ignored. For FGD specifically this is a reasonable visual-presentation
    choice — the percentile / BCa CI bounds reflect the true bootstrap
    distribution but may not contain θ̂ when bias is large.

    Returns (ci_lo, ci_hi, std).
    """
    samples = np.asarray(samples, dtype=np.float64)
    finite = samples[np.isfinite(samples)]
    if finite.size < 2:
        return float("nan"), float("nan"), float("nan")
    std = float(np.std(finite, ddof=1))
    if not np.isfinite(std) or std <= 0:
        print(f"  WARN: Wald CI: std is non-positive/non-finite ({std}); "
              f"falling back to percentile CI.")
        return _percentile_ci(samples, ci_level)
    z = float(_norm.ppf(1.0 - (1.0 - float(ci_level)) / 2.0))
    lo = float(theta_hat) - z * std
    hi = float(theta_hat) + z * std
    return lo, hi, std


def _jackknife_fgd(pred_features_list, target_features_list):
    """Leave-one-clip-out FGD per clip; returns (N,) array (or empty if N<2).

    Cost: N FGD computations, each on N-1 clips' features. Dominant cost is
    the cov + matrix sqrt in 240-D, ~1 second per clip → ~3 minutes for N=202.
    """
    n = len(pred_features_list)
    if n < 2:
        return np.array([])
    # Pre-reshape per-clip blocks once.
    pred_blocks = [x.reshape(-1, x.shape[-1]) for x in pred_features_list]
    target_blocks = [x.reshape(-1, x.shape[-1]) for x in target_features_list]
    jack_vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        pred_loo = np.concatenate(pred_blocks[:i] + pred_blocks[i + 1:], axis=0)
        target_loo = np.concatenate(target_blocks[:i] + target_blocks[i + 1:], axis=0)
        try:
            jack_vals[i] = FGD.frechet_distance(pred_loo, target_loo)
        except Exception as exc:
            print(f"  WARN: jackknife FGD failed for clip {i}: {exc!r}; "
                  f"using NaN.")
            jack_vals[i] = float("nan")
    return jack_vals


def _jackknife_metric(sum_incs, counter_incs):
    """Leave-one-clip-out metric per clip via accumulator differences.

    Returns (N,) array. For each clip i the leave-one-out metric is
    (Σs - s_i) / (Σc - c_i). O(N) total.
    """
    sum_arr = np.asarray(sum_incs, dtype=np.float64)
    cnt_arr = np.asarray(counter_incs, dtype=np.float64)
    n = sum_arr.size
    if n < 2:
        return np.array([])
    total_sum = sum_arr.sum()
    total_cnt = cnt_arr.sum()
    jack_vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = total_sum - sum_arr[i]
        c = total_cnt - cnt_arr[i]
        jack_vals[i] = s / c if c > 0 else float("nan")
    return jack_vals


def bootstrap_fgd(pred_features_list, target_features_list,
                  n_bootstrap, ci_level, rng,
                  ci_method="percentile", point_estimate=None):
    """Clip-level bootstrap of the Fréchet distance.

    pred_features_list / target_features_list are the raw lists held by the
    EMAGE FGD evaluator (one ndarray per clip, each shaped (1, T_chunks, D) or
    similar). For each bootstrap iteration we resample N clip indices with
    replacement, concatenate their per-clip feature blocks, and recompute the
    Fréchet distance via the same staticmethod the evaluator uses.

    `ci_method`:
      - "percentile" (default): raw [α/2, 1-α/2] percentiles of bootstrap samples
      - "bca"                 : bias-corrected and accelerated; uses `point_estimate`
                                and a leave-one-clip-out jackknife pass.
    """
    n_clips = len(pred_features_list)
    if n_clips == 0:
        raise ValueError("FGD bootstrap requires non-empty per-clip feature lists.")
    if n_clips != len(target_features_list):
        raise ValueError(
            f"pred/target feature list length mismatch: "
            f"{n_clips} vs {len(target_features_list)}"
        )
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_clips, n_clips)
        sampled_pred = np.concatenate(
            [pred_features_list[i].reshape(-1, pred_features_list[i].shape[-1])
             for i in idx],
            axis=0,
        )
        sampled_target = np.concatenate(
            [target_features_list[i].reshape(-1, target_features_list[i].shape[-1])
             for i in idx],
            axis=0,
        )
        samples[b] = FGD.frechet_distance(sampled_pred, sampled_target)

    if ci_method == "bca":
        if point_estimate is None:
            raise ValueError("BCa requires `point_estimate` (the FGD on full data).")
        jack = _jackknife_fgd(pred_features_list, target_features_list)
        lo, hi, std = _bca_ci(samples, point_estimate, jack, ci_level)
    elif ci_method == "wald":
        if point_estimate is None:
            raise ValueError("Wald CI requires `point_estimate` (the FGD on full data).")
        lo, hi, std = _wald_ci(samples, point_estimate, ci_level)
    else:
        lo, hi, std = _percentile_ci(samples, ci_level)
    return {"ci_lo": lo, "ci_hi": hi, "std": std}


def bootstrap_metric(sum_incs, counter_incs, n_bootstrap, ci_level, rng,
                     ci_method="percentile", point_estimate=None):
    """Clip-level bootstrap of a sum/counter aggregator (BC / L1div / LVD / MSE).

    Resample (sum_inc, counter_inc) pairs with replacement; recompute
    sum(s_inc) / sum(c_inc). This mirrors the evaluator's `.avg()` exactly,
    so weighting by per-clip counter increments is preserved (matters when
    per-clip lengths differ; here windows are equal, but kept for safety).

    `ci_method`:
      - "percentile" (default)
      - "bca"        : uses `point_estimate` and an O(N) leave-one-clip-out
                       jackknife (cheap for sum/counter aggregators).
    """
    sum_arr = np.asarray(sum_incs, dtype=np.float64)
    cnt_arr = np.asarray(counter_incs, dtype=np.float64)
    n = sum_arr.size
    if n == 0:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "std": float("nan")}
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        s = sum_arr[idx].sum()
        c = cnt_arr[idx].sum()
        samples[b] = s / c if c > 0 else float("nan")

    if ci_method == "bca":
        if point_estimate is None:
            raise ValueError("BCa requires `point_estimate` (the metric on full data).")
        jack = _jackknife_metric(sum_arr, cnt_arr)
        lo, hi, std = _bca_ci(samples, point_estimate, jack, ci_level)
    elif ci_method == "wald":
        if point_estimate is None:
            raise ValueError("Wald CI requires `point_estimate` (the metric on full data).")
        lo, hi, std = _wald_ci(samples, point_estimate, ci_level)
    else:
        lo, hi, std = _percentile_ci(samples, ci_level)
    return {"ci_lo": lo, "ci_hi": hi, "std": std}


# ──────────────────────────────────────────────────────────────
# 读取数据
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(config.split_file)
df = df[df["type"] == "test"]
df = df[df["id"].apply(lambda x: str(x).split("_")[0]).isin([str(p) for p in config.person])]
test_files = df["id"].tolist()

all_data = []  # list of dicts
for file in tqdm(test_files, desc="Loading data"):
    npz_path = os.path.join(config.input_path, "res_" + file + ".npz")
    gt_npz_path = os.path.join(config.input_path, "gt_" + file + ".npz")
    audio_path = os.path.join(config.input_audio_path, "gt_sr16000_" + file + ".wav")
    if not os.path.exists(npz_path) or not os.path.exists(gt_npz_path):
        continue
    npz = np.load(npz_path, allow_pickle=True)
    gt_npz = np.load(gt_npz_path, allow_pickle=True)
    audio, _ = librosa.load(audio_path, sr=16000)
    t = min(npz["poses"].shape[0], gt_npz["poses"].shape[0])
    all_data.append({
        "file": file,
        "motion_pred": npz["poses"][:t],
        "motion_gt": gt_npz["poses"][:t],
        "expression_pred": npz["expressions"][:t],
        "expression_gt": gt_npz["expressions"][:t],
        "betas": gt_npz["betas"],
        "audio": audio,
        "num_frames": t,
    })

total_sequences = len(all_data)
if total_sequences == 0:
    raise RuntimeError("No valid data found. Check input paths.")
print(f"Loaded {total_sequences} sequences")

# 统计长度分布
lengths = sorted([d["num_frames"] for d in all_data])
print(f"Sequence length stats: min={lengths[0]}, max={lengths[-1]}, "
      f"median={lengths[len(lengths)//2]}, mean={np.mean(lengths):.0f}")


# ──────────────────────────────────────────────────────────────
# 生成 terminal-window 列表
# ──────────────────────────────────────────────────────────────
def build_terminal_windows(eval_at_T, terminal_window_sec, fps):
    """Return list of (start_frame, end_frame, T_sec).

    Each T in eval_at_T defines a terminal window [(T - W) * fps, T * fps].
    If T < W, the window is clipped at 0 (effectively the whole rollout).
    """
    windows = []
    for T in eval_at_T:
        end = int(T * fps)
        start = max(0, int((T - terminal_window_sec) * fps))
        windows.append((start, end, int(T)))
    return windows


windows = build_terminal_windows(
    config.eval_at_T, config.terminal_window_sec, pose_fps,
)

if len(windows) == 0:
    raise RuntimeError(
        f"No windows generated. eval_at_T={config.eval_at_T}. "
        f"Provide at least one positive T."
    )

# Sort by T ascending so the FGD@T curve is naturally ordered.
windows.sort(key=lambda x: x[2])
W_sec = config.terminal_window_sec
print(f"Mode: terminal, W={W_sec}s, {len(windows)} window(s): "
      + ", ".join(f"T={T}s→[{s/pose_fps:.1f},{e/pose_fps:.1f}]s"
                  for (s, e, T) in windows))


# ──────────────────────────────────────────────────────────────
# Subset filter: common subset (default) vs per-T (--per_T_subset)
# ──────────────────────────────────────────────────────────────
max_T_frames = max(end for (_, end, _) in windows)
min_T_frames = min(end for (_, end, _) in windows)

if not config.per_T_subset:
    # Default: filter once by max(T)*fps so the same clip set is used at all T.
    common_data = [d for d in all_data if d["num_frames"] >= max_T_frames]
    n_common = len(common_data)
    drop_ratio_common = 1 - n_common / total_sequences

    if n_common < config.min_samples:
        raise RuntimeError(
            f"Common subset (clips with >= {max_T_frames} frames "
            f"= {max_T_frames/pose_fps:.1f}s) has only {n_common} clips, "
            f"less than min_samples={config.min_samples}. "
            f"Consider lowering max(eval_at_T), or use --per_T_subset."
        )
    if drop_ratio_common > config.max_drop_ratio:
        raise RuntimeError(
            f"Common subset filter dropped {drop_ratio_common*100:.1f}% > "
            f"max_drop_ratio={config.max_drop_ratio*100:.1f}% of clips. "
            f"Consider lowering max(eval_at_T), or relax --max_drop_ratio, "
            f"or use --per_T_subset."
        )
    print(f"Common subset: {n_common}/{total_sequences} clips "
          f"(>= {max_T_frames} frames = {max_T_frames/pose_fps:.1f}s, "
          f"drop_ratio={drop_ratio_common*100:.1f}%)")
else:
    common_data = None  # per-T filtering applied inside the loop
    n_common = None
    drop_ratio_common = None
    print(f"Per-T subset: each T filters its own clips "
          f"(min_T={min_T_frames}f, max_T={max_T_frames}f)")


# ──────────────────────────────────────────────────────────────
# 逐窗口计算指标
# ──────────────────────────────────────────────────────────────
results = []

for win_idx, (w_start, w_end, T_sec) in enumerate(windows):
    win_len = w_end - w_start
    win_label = f"T={T_sec}s window=[{w_start/pose_fps:.1f}s, {w_end/pose_fps:.1f}s]"

    # Pick the subset for this window.
    if config.per_T_subset:
        # Per-T mode: filter inside the loop.
        valid_data = [d for d in all_data if d["num_frames"] >= w_end]
        n_valid = len(valid_data)
        n_dropped = total_sequences - n_valid
        drop_ratio = n_dropped / total_sequences

        if drop_ratio > config.max_drop_ratio:
            raise RuntimeError(
                f"{win_label}: dropped {n_dropped}/{total_sequences} "
                f"({drop_ratio*100:.1f}%) sequences (insufficient length), "
                f"exceeding max_drop_ratio={config.max_drop_ratio}. "
                f"Lower max(eval_at_T), or relax --max_drop_ratio."
            )
        if n_valid < config.min_samples:
            print(f"  {win_label}: only {n_valid} valid sequences "
                  f"(< min_samples={config.min_samples}), skipping.")
            continue
        print(f"  {win_label}: {n_valid}/{total_sequences} sequences "
              f"(dropped {drop_ratio*100:.1f}%)")
    else:
        # Common subset mode: same set of clips at every T (already validated above).
        valid_data = common_data
        n_valid = len(valid_data)
        drop_ratio = drop_ratio_common
        print(f"  {win_label}: using common subset (n={n_valid})")

    # 创建 evaluators（每个窗口独立实例）
    fgd_eval = FGD(download_path=download_path, device=str(device))
    if bootstrap_enabled:
        # bootstrap_fgd reads fgd_eval.pred_features / .target_features (lists).
        # The offload mode stores features in memmap files instead, which would
        # silently break bootstrap. Guard against it here.
        if getattr(fgd_eval, "feature_offload", False):
            raise RuntimeError(
                "bootstrap_fgd requires FGD(feature_offload=False) (in-memory "
                "feature lists). Re-construct FGD without offload."
            )
    if not config.only_fgd:
        bc_eval = BC(download_path=download_path, sigma=0.3, order=7)
        l1_eval = L1div()
        lvd_eval = LVDFace()
        mse_eval = MSEFace()

    # Per-clip (sum_inc, counter_inc) for BC/L1/LVD/MSE — derived from
    # accumulator diffs so bootstrap_metric can reproduce evaluator.avg().
    per_clip_diffs = {
        name: {"sum": [], "counter": []}
        for name in ("bc", "l1", "lvd", "mse")
    } if (bootstrap_enabled and not config.only_fgd) else None

    for d in valid_data:
        motion_pred = d["motion_pred"][w_start:w_end]
        motion_gt = d["motion_gt"][w_start:w_end]
        expression_pred = d["expression_pred"][w_start:w_end]
        expression_gt = d["expression_gt"][w_start:w_end]
        betas = d["betas"]
        audio = d["audio"]
        t = motion_pred.shape[0]

        if not config.only_fgd:
            # BC: 需要对齐 audio 和 motion 的时间范围
            # audio 采样率 16000, motion 30fps
            # 窗口内不裁剪首尾 2s（原始 emage_eval 裁剪是针对完整序列的，
            # 窗口内序列本身已经是子段，不需要额外裁剪）
            sr_audio = 16000
            audio_start = int(w_start * sr_audio / pose_fps)
            audio_end = int(w_end * sr_audio / pose_fps)
            audio_clip = audio[audio_start:audio_end]

            motion_position_pred = get_motion_rep_numpy(
                motion_pred, pose_fps=pose_fps, device=str(device), betas=betas
            )["position"].reshape(t, -1)

            audio_beat = bc_eval.load_audio(
                audio_clip, t_start=None, t_end=None, without_file=True, sr_audio=sr_audio,
            )
            motion_beat = bc_eval.load_motion(
                motion_position_pred, t_start=0, t_end=t, pose_fps=pose_fps, without_file=True,
            )
            # Snapshot accumulator state before .compute() to derive per-clip
            # increments (used only when --n_bootstrap > 0).
            if per_clip_diffs is not None:
                _bc_s0, _bc_c0 = bc_eval.sum, bc_eval.counter
                _l1_s0, _l1_c0 = l1_eval.sum, l1_eval.counter

            bc_eval.compute(audio_beat, motion_beat, length=t, pose_fps=pose_fps)

            l1_eval.compute(motion_position_pred)

            face_position_pred = get_motion_rep_numpy(
                motion_pred, device=str(device),
                expressions=expression_pred, expression_only=True, betas=betas,
            )["vertices"]
            face_position_gt = get_motion_rep_numpy(
                motion_gt, device=str(device),
                expressions=expression_gt, expression_only=True, betas=betas,
            )["vertices"]
            if per_clip_diffs is not None:
                _lvd_s0, _lvd_c0 = lvd_eval.sum, lvd_eval.counter
                _mse_s0, _mse_c0 = mse_eval.sum, mse_eval.counter

            lvd_eval.compute(face_position_pred, face_position_gt)
            mse_eval.compute(face_position_pred, face_position_gt)

            if per_clip_diffs is not None:
                per_clip_diffs["bc"]["sum"].append(bc_eval.sum - _bc_s0)
                per_clip_diffs["bc"]["counter"].append(bc_eval.counter - _bc_c0)
                per_clip_diffs["l1"]["sum"].append(l1_eval.sum - _l1_s0)
                per_clip_diffs["l1"]["counter"].append(l1_eval.counter - _l1_c0)
                per_clip_diffs["lvd"]["sum"].append(lvd_eval.sum - _lvd_s0)
                per_clip_diffs["lvd"]["counter"].append(lvd_eval.counter - _lvd_c0)
                per_clip_diffs["mse"]["sum"].append(mse_eval.sum - _mse_s0)
                per_clip_diffs["mse"]["counter"].append(mse_eval.counter - _mse_c0)

        # FGD: rotation 6d
        motion_gt_t = torch.from_numpy(motion_gt).to(device).unsqueeze(0)
        motion_pred_t = torch.from_numpy(motion_pred).to(device).unsqueeze(0)
        motion_gt_r6d = rc.axis_angle_to_rotation_6d(
            motion_gt_t.reshape(1, t, 55, 3)
        ).reshape(1, t, 55 * 6)
        motion_pred_r6d = rc.axis_angle_to_rotation_6d(
            motion_pred_t.reshape(1, t, 55, 3)
        ).reshape(1, t, 55 * 6)
        fgd_eval.update(motion_pred_r6d.float(), motion_gt_r6d.float())

    # 汇总该窗口的指标
    win_metrics = {
        "T_sec": T_sec,
        "window_start_frame": w_start,
        "window_end_frame": w_end,
        "window_start_sec": w_start / pose_fps,
        "window_end_sec": w_end / pose_fps,
        "num_valid_sequences": n_valid,
        "drop_ratio": drop_ratio,
        "fgd": float(fgd_eval.compute()),
    }
    if not config.only_fgd:
        win_metrics["bc"] = float(bc_eval.avg())
        win_metrics["l1"] = float(l1_eval.avg())
        win_metrics["lvd"] = float(lvd_eval.avg())
        win_metrics["mse"] = float(mse_eval.avg())

    # Bootstrap CI (clip-level resampling) — captures sampling noise over the
    # evaluation set, not model stochasticity.
    if bootstrap_enabled:
        fgd_ci = bootstrap_fgd(
            fgd_eval.pred_features, fgd_eval.target_features,
            n_bootstrap=config.n_bootstrap, ci_level=config.ci_level, rng=rng,
            ci_method=config.ci_method, point_estimate=win_metrics["fgd"],
        )
        win_metrics["fgd_ci_lo"] = fgd_ci["ci_lo"]
        win_metrics["fgd_ci_hi"] = fgd_ci["ci_hi"]
        win_metrics["fgd_std"] = fgd_ci["std"]
        if not config.only_fgd:
            for name in ("bc", "l1", "lvd", "mse"):
                m_ci = bootstrap_metric(
                    per_clip_diffs[name]["sum"],
                    per_clip_diffs[name]["counter"],
                    n_bootstrap=config.n_bootstrap,
                    ci_level=config.ci_level,
                    rng=rng,
                    ci_method=config.ci_method,
                    point_estimate=win_metrics[name],
                )
                win_metrics[f"{name}_ci_lo"] = m_ci["ci_lo"]
                win_metrics[f"{name}_ci_hi"] = m_ci["ci_hi"]
                win_metrics[f"{name}_std"] = m_ci["std"]

    results.append(win_metrics)
    if bootstrap_enabled:
        fgd_str = (f"FGD={win_metrics['fgd']:.4f} "
                   f"[{win_metrics['fgd_ci_lo']:.4f},{win_metrics['fgd_ci_hi']:.4f}]")
    else:
        fgd_str = f"FGD={win_metrics['fgd']:.4f}"
    extra_str = ""
    if not config.only_fgd:
        extra_str = (f", BC={win_metrics['bc']:.4f}, L1={win_metrics['l1']:.4f}, "
                     f"LVD={win_metrics['lvd']:.6e}, MSE={win_metrics['mse']:.6e}")
    print(f"    → T={T_sec}s {fgd_str}{extra_str}")

# ──────────────────────────────────────────────────────────────
# 输出（JSON + sibling CSV）
# ──────────────────────────────────────────────────────────────
output = {
    "mode": "terminal",
    "eval_at_T": [int(T) for T in config.eval_at_T],
    "terminal_window_sec": int(config.terminal_window_sec),
    "subset_strategy": "per_T" if config.per_T_subset else "common",
    "subset_filter_threshold_frames": (None if config.per_T_subset else int(max_T_frames)),
    "common_subset_size": (None if config.per_T_subset else int(n_common)),
    "pose_fps": pose_fps,
    "total_sequences": total_sequences,
    "person": [int(p) for p in config.person],
    "bootstrap": {
        "enabled": bootstrap_enabled,
        "n_iterations": int(config.n_bootstrap),
        "ci_level": float(config.ci_level),
        "method": str(config.ci_method),
        "seed": int(config.bootstrap_seed),
        "scope": "clip-level resampling (per-test-set sampling noise only; "
                 "does NOT capture sampler-seed or training-seed variability)",
    },
    "sequence_length_stats": {
        "min": int(lengths[0]),
        "max": int(lengths[-1]),
        "median": int(lengths[len(lengths) // 2]),
        "mean": float(np.mean(lengths)),
    },
    "windows": results,
}

os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
with open(config.output_path, "w") as f:
    json.dump(output, f, indent=4)
print(f"\nResults saved to {config.output_path}")

# Sibling CSV: one row per T, columns ready for downstream FGD@T plotting.
# When bootstrap is enabled, ci_lo/ci_hi/std are appended for every metric.
csv_path = os.path.splitext(config.output_path)[0] + ".csv"


def _metric_block(name, fmt):
    """Return (column_names, value-formatter-fn) for a single metric.
    The formatter takes the per-window result dict and returns a list of
    string-formatted values matching the column names."""
    cols = [name]
    if bootstrap_enabled:
        cols += [f"{name}_ci_lo", f"{name}_ci_hi", f"{name}_std"]

    def _fmt(r):
        out = [fmt.format(r[name])]
        if bootstrap_enabled:
            out += [
                fmt.format(r[f"{name}_ci_lo"]),
                fmt.format(r[f"{name}_ci_hi"]),
                fmt.format(r[f"{name}_std"]),
            ]
        return out
    return cols, _fmt


csv_columns = ["T_sec", "window_start_sec", "window_end_sec", "n_valid"]
metric_specs = [("fgd", "{:.6f}")]
if not config.only_fgd:
    metric_specs += [("bc", "{:.6f}"), ("l1", "{:.6f}"),
                     ("lvd", "{:.6e}"), ("mse", "{:.6e}")]
metric_blocks = [_metric_block(name, fmt) for name, fmt in metric_specs]
for cols, _ in metric_blocks:
    csv_columns += cols

with open(csv_path, "w") as f:
    f.write(",".join(csv_columns) + "\n")
    for r in results:
        row = [
            f"{r['T_sec']}",
            f"{r['window_start_sec']:.3f}",
            f"{r['window_end_sec']:.3f}",
            f"{r['num_valid_sequences']}",
        ]
        for _, fmt_fn in metric_blocks:
            row += fmt_fn(r)
        f.write(",".join(row) + "\n")
print(f"CSV    saved to {csv_path}")

# 打印摘要表格（terminal-window）
print(f"\n{'='*80}")
ci_tag = (f", {int(config.ci_level*100)}% bootstrap CI "
          f"({config.n_bootstrap} iters, {config.ci_method})"
          if bootstrap_enabled else "")
print(f"Summary (terminal-window FGD@T, W={config.terminal_window_sec}s, "
      f"{'common' if not config.per_T_subset else 'per-T'} subset{ci_tag})")
print(f"{'='*80}")


def _fmt_with_ci(value, fmt, ci_lo=None, ci_hi=None):
    """Format point estimate, optionally with bracketed CI."""
    base = fmt.format(value)
    if ci_lo is None or ci_hi is None:
        return base
    return f"{base} [{fmt.format(ci_lo)},{fmt.format(ci_hi)}]"


fgd_w = 28 if bootstrap_enabled else 8
bc_w = 28 if bootstrap_enabled else 8
l1_w = 32 if bootstrap_enabled else 10
lvd_w = 40 if bootstrap_enabled else 12
mse_w = 40 if bootstrap_enabled else 12

header = f"  {'T(s)':>5s}  {'Window(s)':>15s}  {'N':>5s}  {'FGD':>{fgd_w}s}"
if not config.only_fgd:
    header += (f"  {'BC':>{bc_w}s}  {'L1':>{l1_w}s}  "
               f"{'LVD':>{lvd_w}s}  {'MSE':>{mse_w}s}")
print(header)
print("  " + "-" * (len(header) - 2))
for r in results:
    win_str = f"[{r['window_start_sec']:.1f},{r['window_end_sec']:.1f}]"
    fgd_str = _fmt_with_ci(
        r["fgd"], "{:.4f}",
        r.get("fgd_ci_lo") if bootstrap_enabled else None,
        r.get("fgd_ci_hi") if bootstrap_enabled else None,
    )
    line = (f"  {r['T_sec']:>5d}"
            f"  {win_str:>15s}"
            f"  {r['num_valid_sequences']:>5d}"
            f"  {fgd_str:>{fgd_w}s}")
    if not config.only_fgd:
        bc_str = _fmt_with_ci(
            r["bc"], "{:.4f}",
            r.get("bc_ci_lo") if bootstrap_enabled else None,
            r.get("bc_ci_hi") if bootstrap_enabled else None,
        )
        l1_str = _fmt_with_ci(
            r["l1"], "{:.4f}",
            r.get("l1_ci_lo") if bootstrap_enabled else None,
            r.get("l1_ci_hi") if bootstrap_enabled else None,
        )
        lvd_str = _fmt_with_ci(
            r["lvd"], "{:.6e}",
            r.get("lvd_ci_lo") if bootstrap_enabled else None,
            r.get("lvd_ci_hi") if bootstrap_enabled else None,
        )
        mse_str = _fmt_with_ci(
            r["mse"], "{:.6e}",
            r.get("mse_ci_lo") if bootstrap_enabled else None,
            r.get("mse_ci_hi") if bootstrap_enabled else None,
        )
        line += (f"  {bc_str:>{bc_w}s}  {l1_str:>{l1_w}s}  "
                 f"{lvd_str:>{lvd_w}s}  {mse_str:>{mse_w}s}")
    print(line)
