"""
SHOW (TalkSHOW SMPL-X) → BEAT2-style LMDB ETL.

Mirrors `scripts/create_lmdb.py` but reads from the SHOW dataset directory layout
(<root>/<speaker>/<vid>/<split>/<seq>/{<seq>.pkl, <seq>.wav}) and emits LMDB
records that exactly match the contract consumed by
`diffusion/data/datasets/beat2/BEATXDataset.py:LMDBDataset`.

This script does NOT touch any existing BEAT2 code path.  Canonical TalkSHOW
face-condition LMDBs use ``--feature_scope full_clip``: WavLM encodes each
source clip once, the features are interpolated to the full motion length, and
train/val windows are sliced afterwards.  ``--feature_scope window`` preserves
the legacy per-window behavior explicitly.

Field contract (must match BEAT2 LMDB exactly):
    betas        : float32, (300,)         — SHOW betas (10) right zero-padded
    poses        : float32, (T, 165)       — axis-angle, BEAT2 joint order
    expressions  : float32, (T, 100)
    trans        : float32, (T, 3)         — SHOW has none → zeros
    audio        : float32, (T*16000//30,) — 16 kHz mono
    huberts      : float32, (1, T, D)      — wavlm/hubert features at motion fps
    face_huberts : float32, (1, Tctx, D)   — source-valid face-head context
    face_hubert_valid_length : int          — valid prefix length in Tctx
    face_hubert_crop_start   : int          — central T-frame crop in valid prefix
    pid          : int                     — SHOW_SPEAKER_ID[name] + pid_offset
    ori_vid      : str                     — "<speaker>__<seq_id>"

Database-level records:
    __len__       : ASCII integer
    __meta__      : UTF-8 JSON provenance/contract metadata (never pickle)

Joint order in `poses` (165 = 55 joints × 3 axis-angle):
    [0:3]    global_orient    [3:66]   body_pose (21 joints)
    [66:69]  jaw_pose         [69:72]  leye_pose       [72:75]  reye_pose
    [75:120] left_hand_pose   [120:165] right_hand_pose
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import uuid
from collections import Counter
from pathlib import Path

import lmdb
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_COMPONENT_PATH = REPO_ROOT / "tools" / "show" / "hand_component.json"
LMDB_META_KEY = b"__meta__"
LMDB_META_SCHEMA_VERSION = 2
OFFICIAL_SPLIT_SHA256 = (
    "2df6e745cdf7473f13ce3ae2ed759c3cceb60c9197e7f3fd65110e7bc20b6f2d"
)
OFFICIAL_HAND_COMPONENT_SHA256 = (
    "a9db3d2f04fe161e09553b17aaf5922791b0ed826593e640f4ddf8ebb8baf423"
)
POSE_FPS = 30
AUDIO_SR = 16000
POSE_ORDER = (
    "global_orient",
    "body_pose_axis",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "left_hand_pose_axis_angle",
    "right_hand_pose_axis_angle",
)
FEATURE_SCOPES = ("full_clip", "window")
DURATION_MISMATCH_POLICIES = ("error", "skip")
FACE_CONTEXT_FRAMES = 6
FACE_CONTEXT_PROTOCOL = "source_clip_valid_slice_v1"
_AUDIO_METADATA_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "added_tokens.json",
)
_AUDIO_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


class ShowClipValidationError(ValueError):
    """A malformed SHOW clip with a stable machine-readable reason."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = str(reason)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value):
    """Match the canonical JSON fingerprint stored by TalkSHOW face checkpoints."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot read JSON file {path}: {e}") from e


def collect_audio_model_provenance(model_id, model_type):
    """Collect portable model metadata and hashes for a local HF snapshot.

    Remote Hugging Face identifiers remain supported for compatibility, but they
    cannot be fingerprinted before ``from_pretrained`` resolves them. In that
    case the hash fields are empty/None and downstream strict verification can
    reject the artifact.
    """
    model_path = Path(os.path.expanduser(str(model_id)))
    is_local = model_path.is_dir()
    resolved_model_id = str(model_path.resolve()) if is_local else str(model_id)
    result = {
        "audio_model_id": resolved_model_id,
        "audio_config_sha256": None,
        "audio_preprocessor_sha256": None,
        "audio_model_files_sha256": {},
        "audio_do_normalize": None,
        "audio_feat_extract_norm": None,
        "audio_add_adapter": None,
        "configured_audio_feature_dim": None,
    }
    if not is_local:
        return result

    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"audio model snapshot has no config.json: {model_path}"
        )
    config = _read_json_file(config_path)
    configured_model_type = config.get("model_type")
    if configured_model_type is not None and configured_model_type != model_type:
        raise RuntimeError(
            "audio model type mismatch: "
            f"--model_type={model_type!r}, config.json={configured_model_type!r}"
        )

    hidden_size = config.get("hidden_size")
    if hidden_size is not None:
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise RuntimeError(
                f"audio config has invalid hidden_size: {hidden_size!r}"
            )
        result["configured_audio_feature_dim"] = hidden_size
    result["audio_config_sha256"] = canonical_json_sha256(config)
    result["audio_feat_extract_norm"] = config.get("feat_extract_norm")
    result["audio_add_adapter"] = bool(config.get("add_adapter", False))

    preprocessor_path = model_path / "preprocessor_config.json"
    if model_type == "wavlm" and not preprocessor_path.is_file():
        raise FileNotFoundError(
            f"WavLM snapshot has no preprocessor_config.json: {model_path}"
        )
    if preprocessor_path.is_file():
        preprocessor = _read_json_file(preprocessor_path)
        result["audio_preprocessor_sha256"] = canonical_json_sha256(
            preprocessor
        )
        do_normalize = preprocessor.get("do_normalize")
        if model_type == "wavlm" and not isinstance(do_normalize, bool):
            raise RuntimeError(
                "WavLM preprocessor_config.json must contain boolean "
                "'do_normalize'"
            )
        result["audio_do_normalize"] = do_normalize

    if model_type == "wavlm":
        if result["audio_add_adapter"]:
            raise RuntimeError("WavLM checkpoints with adapters are unsupported")
        if result["audio_feat_extract_norm"] != "layer":
            raise RuntimeError(
                "TalkSHOW WavLM requires feat_extract_norm='layer'; "
                f"got {result['audio_feat_extract_norm']!r}"
            )

    files_to_hash = set()
    for name in _AUDIO_METADATA_FILES + _AUDIO_WEIGHT_FILES:
        candidate = model_path / name
        if candidate.is_file():
            files_to_hash.add(candidate)

    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = model_path / index_name
        if not index_path.is_file():
            continue
        index = _read_json_file(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(
                f"audio model weight index has no weight_map: {index_path}"
            )
        for relative_name in sorted(set(weight_map.values())):
            shard = model_path / relative_name
            if not shard.is_file():
                raise FileNotFoundError(
                    f"audio model weight shard is missing: {shard}"
                )
            files_to_hash.add(shard)

    has_weights = any(
        (model_path / name).is_file() for name in _AUDIO_WEIGHT_FILES
    )
    if not has_weights:
        raise FileNotFoundError(
            "audio model snapshot contains none of: "
            + ", ".join(_AUDIO_WEIGHT_FILES)
        )

    result["audio_model_files_sha256"] = {
        str(path.relative_to(model_path)): sha256_file(path)
        for path in sorted(files_to_hash)
    }
    return result


# ---------------------------------------------------------------------------
# Audio feature extractor (copy of scripts/create_lmdb.py:AudioFeatureExtractor)
# Kept inline so this script has zero coupling with the BEAT2 ETL.
# ---------------------------------------------------------------------------
class AudioFeatureExtractor:
    def __init__(
        self,
        model_type="wavlm",
        model_id=None,
    ):
        super().__init__()
        if model_id is None or not str(model_id).strip():
            raise ValueError("model_id must identify a local audio model")
        self.model_type = model_type
        self.model_id = str(model_id)
        if model_type == "wavlm":
            from transformers import Wav2Vec2Processor, WavLMModel
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = WavLMModel.from_pretrained(model_id).requires_grad_(False).eval().cuda()
        elif model_type == "hubert":
            try:
                from modelscope import AutoProcessor, HubertModel
            except ImportError:
                from transformers import AutoProcessor, HubertModel
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = HubertModel.from_pretrained(model_id).requires_grad_(False).eval().cuda()
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        self.feature_dim = int(hidden_size) if hidden_size is not None else None

    def __call__(self, raw_audio, seq_size=None):
        inputs = self.processor(
            raw_audio, sampling_rate=AUDIO_SR, return_tensors="pt"
        ).input_values.cuda()
        with torch.no_grad():
            features = self.model(inputs).last_hidden_state
            if seq_size is not None:
                features = F.interpolate(
                    features.permute((0, 2, 1)), seq_size, mode="linear", align_corners=True
                ).permute((0, 2, 1))
        features = features.cpu().numpy()
        observed_dim = int(features.shape[-1])
        if self.feature_dim is None:
            self.feature_dim = observed_dim
        elif observed_dim != self.feature_dim:
            raise RuntimeError(
                "audio feature dimension changed during extraction: "
                f"configured={self.feature_dim}, observed={observed_dim}"
            )
        return features


# ---------------------------------------------------------------------------
# SMPL-X kinematic parents (copy of scripts/create_lmdb.py:SMPLX_PARENTS)
# ---------------------------------------------------------------------------
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]


def LocalToGlobalAA(local_aa, parent_idx):
    """Convert per-joint local axis-angle (N, J*3) into global axis-angle (N, J*3)."""
    N = local_aa.shape[0]
    nJoints = len(parent_idx)
    local_rot = R.from_rotvec(local_aa.reshape(-1, 3)).as_matrix().reshape((N, nJoints, 3, 3))
    global_rot = np.zeros_like(local_rot)
    for i in range(nJoints):
        if parent_idx[i] == -1:
            global_rot[:, i] = local_rot[:, i]
        else:
            global_rot[:, i] = np.einsum("nij,njk->nik", global_rot[:, parent_idx[i]], local_rot[:, i])
    global_aa = R.from_matrix(global_rot.reshape(-1, 3, 3)).as_rotvec().reshape(N, -1)
    return global_aa


# ---------------------------------------------------------------------------
# SHOW-specific constants and helpers
# ---------------------------------------------------------------------------
# Source: TalkSHOW data_utils/consts.py (24 speakers).
# Values 0..23 are picked up directly; pass --pid_offset to shift the whole table
# (e.g. 31 to avoid clashing with BEAT2's 0..30 if you later want to merge).
SHOW_SPEAKER_ID = {
    "Amel_Karboul": 0,
    "Bill_Gates": 1,
    "Christina_Wallace": 2,
    "Dan_Ariely": 3,
    "daniel_susskind": 4,
    "Dena_Simmons": 5,
    "Enric_Sala": 6,
    "FeiFei_Li": 7,
    "GabeBarcia_Colombo": 8,
    "Kelly_Richmond_Pope": 9,
    "Keller_Rinaudo": 10,
    "Laurel_Braitman": 11,
    "Lisa_Feldman_Barrett": 12,
    "molly_winter": 13,
    "Sara_DeWitt": 14,
    "Seema_Bansal": 15,
    "Stacy_Smith": 16,
    "Stanley_McChrystal": 17,
    "Vicki_Arroyo": 18,
    "speeker_oliver": 19,
    "oliver": 20,
    "chemistry": 21,
    "seth": 22,
    "conan": 23,
}
CANONICAL_TALKSHOW_SPEAKERS = frozenset(
    ("oliver", "chemistry", "seth", "conan")
)


def load_hand_pca(path):
    """Load 12-component left/right hand PCA basis. Returns (LH_C, RH_C) of shape (12, 45)."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "TalkSHOW hand PCA file does not exist: "
            f"{path}. Expected the repository asset at "
            f"{DEFAULT_HAND_COMPONENT_PATH} or pass --hand_component_path."
        )
    comp = _read_json_file(path)
    if not isinstance(comp, dict) or "left" not in comp or "right" not in comp:
        raise RuntimeError(
            f"hand PCA file must contain 'left' and 'right' arrays: {path}"
        )
    lh = np.asarray(comp["left"], dtype=np.float32)[:12, :]
    rh = np.asarray(comp["right"], dtype=np.float32)[:12, :]
    if lh.shape != (12, 45):
        raise RuntimeError(f"Expected LH_C (12, 45), got {lh.shape} in {path}")
    if rh.shape != (12, 45):
        raise RuntimeError(f"Expected RH_C (12, 45), got {rh.shape} in {path}")
    if not np.isfinite(lh).all() or not np.isfinite(rh).all():
        raise RuntimeError(f"hand PCA file contains NaN/Inf: {path}")
    return lh, rh


def scan_show_dataset(
    data_root, speakers, split, verbose=True, return_stats=False
):
    """Walk SHOW directory tree and yield clip metadata.

    Layout: <data_root>/<speaker>/<vid>/<split>/<seq>/{<seq>.pkl, <seq>.wav}
    Returns: list of (pkl_path, wav_path, speaker_name, seq_id) tuples.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"SHOW data_root does not exist: {data_root}")
    if verbose:
        try:
            top = sorted(os.listdir(data_root))
        except OSError as e:
            print(f"[scan] cannot list data_root: {e}", file=sys.stderr)
            top = []
        print(f"[scan] data_root={data_root} contains {len(top)} entries: {top[:20]}{' ...' if len(top) > 20 else ''}",
              file=sys.stderr)

    items = []
    stats = {
        "speaker_missing": 0,
        "no_split_dir": 0,
        "sequence_dirs": 0,
        "missing_pkl": 0,
        "missing_wav": 0,
        "usable_clips": 0,
    }
    for sp in speakers:
        sp_root = os.path.join(data_root, sp)
        if not os.path.isdir(sp_root):
            print(f"[skip] speaker dir missing: {sp_root}", file=sys.stderr)
            stats["speaker_missing"] += 1
            continue
        vids = sorted(os.listdir(sp_root))
        sp_split_dirs = 0
        sp_seq_dirs = 0
        sp_ok = 0
        for vid in vids:
            split_dir = os.path.join(sp_root, vid, split)
            if not os.path.isdir(split_dir):
                continue
            sp_split_dirs += 1
            seqs = sorted(os.listdir(split_dir))
            for seq in seqs:
                seq_dir = os.path.join(split_dir, seq)
                if not os.path.isdir(seq_dir):
                    continue
                sp_seq_dirs += 1
                stats["sequence_dirs"] += 1
                pkl = os.path.join(seq_dir, f"{seq}.pkl")
                wav = os.path.join(seq_dir, f"{seq}.wav")
                pkl_ok = os.path.isfile(pkl)
                wav_ok = os.path.isfile(wav)
                if pkl_ok and wav_ok:
                    items.append((pkl, wav, sp, seq))
                    sp_ok += 1
                    stats["usable_clips"] += 1
                else:
                    if not pkl_ok:
                        stats["missing_pkl"] += 1
                    if not wav_ok:
                        stats["missing_wav"] += 1
                    if verbose and (stats["missing_pkl"] + stats["missing_wav"]) <= 5:
                        try:
                            inside = os.listdir(seq_dir)
                        except OSError:
                            inside = []
                        print(f"[skip] {seq_dir} missing files (pkl={pkl_ok}, wav={wav_ok}); "
                              f"directory contains: {inside[:10]}", file=sys.stderr)
        if sp_split_dirs == 0:
            print(f"[skip] speaker '{sp}' has no '{split}' subdirectory under any of its {len(vids)} videos "
                  f"(checked {sp_root}); first vids: {vids[:5]}", file=sys.stderr)
            stats["no_split_dir"] += 1
        elif verbose:
            print(f"[scan] speaker '{sp}': {sp_split_dirs} {split}-dirs, {sp_seq_dirs} seqs, {sp_ok} usable",
                  file=sys.stderr)
    print(f"[scan] summary: {stats}", file=sys.stderr)
    if return_stats:
        return items, stats
    return items


def load_split_view_provenance(
    data_root,
    *,
    split,
    speakers,
    usable_clip_count,
    required,
):
    """Load and validate ``prepare_split_view.py`` provenance when present."""

    summary_path = Path(data_root).expanduser().resolve() / (
        "split_view_summary.json"
    )
    empty = {
        "split_manifest_sha256": None,
        "split_view_summary_sha256": None,
        "split_view_counts": None,
        "split_view_missing_count": None,
        "split_view_limit_per_speaker_split": None,
    }
    if not summary_path.is_file():
        if required:
            raise FileNotFoundError(
                "strict TalkSHOW generation requires "
                f"{summary_path}; build the non-destructive official split "
                "view first"
            )
        return empty

    summary = _read_json_file(summary_path)
    if not isinstance(summary, dict):
        raise RuntimeError(
            f"split view summary must be a JSON object: {summary_path}"
        )
    manifest_sha = summary.get("split_sha256")
    if manifest_sha != OFFICIAL_SPLIT_SHA256:
        raise RuntimeError(
            "split view does not use the pinned official manifest: "
            f"{manifest_sha!r} != {OFFICIAL_SPLIT_SHA256}"
        )
    counts = summary.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != {"train", "val", "test"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in counts.values()
        )
    ):
        raise RuntimeError(
            f"split view summary has invalid counts: {counts!r}"
        )
    missing_count = summary.get("missing_count")
    if (
        isinstance(missing_count, bool)
        or not isinstance(missing_count, int)
        or missing_count < 0
    ):
        raise RuntimeError(
            "split view summary has invalid missing_count: "
            f"{missing_count!r}"
        )
    if set(speakers) == CANONICAL_TALKSHOW_SPEAKERS:
        if counts[split] != usable_clip_count:
            raise RuntimeError(
                "split view summary count disagrees with the scanned clips: "
                f"split={split}, summary={counts[split]}, "
                f"scanned={usable_clip_count}"
            )
    elif required:
        raise RuntimeError(
            "strict split provenance requires all canonical TalkSHOW speakers"
        )
    return {
        "split_manifest_sha256": manifest_sha,
        "split_view_summary_sha256": sha256_file(summary_path),
        "split_view_counts": dict(counts),
        "split_view_missing_count": missing_count,
        "split_view_limit_per_speaker_split": summary.get(
            "limit_per_speaker_split"
        ),
    }


def normalize_show_betas(raw_betas, num_frames):
    """Validate SHOW betas and return one SMPL-X 300-D shape vector.

    Accepted source layouts are ``(D,)``, ``(1, D)``, and ``(T, D)`` with
    ``D`` equal to 10 or 300. Per-frame betas must be constant (up to small
    floating-point noise). Deliberately do not flatten arbitrary arrays: doing
    so can turn a malformed tensor into a plausible-looking 300-D vector.
    """
    betas = np.asarray(raw_betas, dtype=np.float32)
    if betas.ndim == 1:
        vector = betas
    elif betas.ndim == 2 and betas.shape[0] in (1, num_frames):
        if betas.shape[0] == num_frames and num_frames > 1:
            reference = np.broadcast_to(betas[0:1], betas.shape)
            if not np.allclose(betas, reference, rtol=1e-5, atol=1e-6):
                raise ShowClipValidationError(
                    "betas_vary_by_frame",
                    "per-frame SHOW betas are not constant",
                )
        vector = betas[0]
    else:
        raise ShowClipValidationError(
            "betas_shape",
            "SHOW betas must have shape (10,), (300,), (1, D), or (T, D); "
            f"got {betas.shape} for T={num_frames}",
        )

    if vector.shape != (10,) and vector.shape != (300,):
        raise ShowClipValidationError(
            "betas_shape",
            f"SHOW beta dimension must be 10 or 300, got {vector.shape}",
        )
    if not np.isfinite(vector).all():
        raise ShowClipValidationError(
            "betas_nonfinite", "SHOW betas contain NaN/Inf"
        )
    if vector.shape[0] == 10:
        vector = np.pad(vector, (0, 290), mode="constant")
    return np.asarray(vector, dtype=np.float32)


def load_show_clip(pkl_path, lh_c, rh_c):
    """Read a single SHOW pkl and assemble it into BEAT2-shape arrays.

    Returns:
        poses       : float32, (T, 165), axis-angle, BEAT2 joint order
        expressions : float32, (T, 100)
        trans       : float32, (T, 3), all zeros (SHOW has no root translation)
        betas       : float32, (300,), right zero-padded from (10,)
        T           : int, number of frames

    Raises ``ShowClipValidationError`` on validation failure so the caller can
    maintain per-reason statistics.
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise ShowClipValidationError(
            "pkl_load_error", f"cannot load {pkl_path}: {e}"
        ) from e
    if not isinstance(data, dict):
        raise ShowClipValidationError(
            "pkl_schema", f"{pkl_path}: top-level object is not a dictionary"
        )

    required = (
        "betas", "global_orient", "body_pose_axis", "jaw_pose",
        "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose",
        "expression",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ShowClipValidationError(
            "missing_motion_field",
            f"{pkl_path}: missing fields {missing}",
        )

    body_pose = np.asarray(data["body_pose_axis"], dtype=np.float32)
    if body_pose.ndim < 1:
        raise ShowClipValidationError(
            "motion_shape",
            f"{pkl_path}: body_pose_axis has invalid shape {body_pose.shape}",
        )
    T = body_pose.shape[0]
    if T == 0:
        raise ShowClipValidationError(
            "zero_motion_frames", f"{pkl_path}: zero frames"
        )

    try:
        global_orient = np.asarray(data["global_orient"], dtype=np.float32).reshape(T, 3)
        body_pose = body_pose.reshape(T, 63)
        jaw_pose = np.asarray(data["jaw_pose"], dtype=np.float32).reshape(T, 3)
        leye_pose = np.asarray(data["leye_pose"], dtype=np.float32).reshape(T, 3)
        reye_pose = np.asarray(data["reye_pose"], dtype=np.float32).reshape(T, 3)
        lhand_pca = np.asarray(data["left_hand_pose"], dtype=np.float32).reshape(T, 12)
        rhand_pca = np.asarray(data["right_hand_pose"], dtype=np.float32).reshape(T, 12)
        expression = np.asarray(data["expression"], dtype=np.float32).reshape(T, 100)
    except Exception as e:
        raise ShowClipValidationError(
            "motion_shape", f"{pkl_path}: shape error: {e}"
        ) from e

    # Hand PCA -> 45-dim axis-angle (15 joints * 3)
    lhand_aa = (lhand_pca @ lh_c).astype(np.float32)
    rhand_aa = (rhand_pca @ rh_c).astype(np.float32)
    if lhand_aa.shape != (T, 45) or rhand_aa.shape != (T, 45):
        raise ShowClipValidationError(
            "hand_expand_shape",
            f"{pkl_path}: hand expand shape error "
            f"left={lhand_aa.shape}, right={rhand_aa.shape}",
        )

    # Concatenate in BEAT2 / SMPL-X order. See diffusion/model/emage_diffusion.py
    # loss_function for the matching slice indices on the consumer side.
    poses = np.concatenate(
        [global_orient, body_pose, jaw_pose, leye_pose, reye_pose, lhand_aa, rhand_aa],
        axis=1,
    ).astype(np.float32)
    if poses.shape != (T, 165):
        raise ShowClipValidationError(
            "motion_shape",
            f"{pkl_path}: poses shape {poses.shape} != {(T, 165)}",
        )

    betas = normalize_show_betas(data["betas"], T)

    trans = np.zeros((T, 3), dtype=np.float32)

    # NaN/Inf checks
    for name, arr in (
        ("poses", poses),
        ("expressions", expression),
        ("betas", betas),
    ):
        if not np.isfinite(arr).all():
            raise ShowClipValidationError(
                f"{name}_nonfinite",
                f"{pkl_path}: {name} contains NaN/Inf",
            )

    return poses, expression, trans, betas, T


def extract_window_audio_features(
    audio_feature_extractor,
    raw_audio,
    audio_windows,
    window_start_frames,
    *,
    motion_frames,
    seq_size,
    feature_scope,
):
    """Extract central features and exact source-valid face-head contexts.

    The frozen TalkSHOW face head has six layers of kernel-3 temporal
    convolutions (three in ``audio_middle`` and three in the expression
    decoder), hence a six-frame receptive-field radius.  A fixed 268-frame
    carrier keeps DataLoader collation simple, while ``valid_length`` prevents
    artificial zero frames from entering the nonlinear face head at the true
    source-clip boundaries.
    """
    if feature_scope not in FEATURE_SCOPES:
        raise ValueError(
            f"feature_scope must be one of {FEATURE_SCOPES}, got {feature_scope!r}"
        )
    if len(audio_windows) != len(window_start_frames):
        raise ValueError("audio_windows and window_start_frames length mismatch")

    if feature_scope == "full_clip":
        full_features = np.asarray(
            audio_feature_extractor(raw_audio, seq_size=motion_frames),
            dtype=np.float32,
        )
        if (
            full_features.ndim != 3
            or full_features.shape[0] != 1
            or full_features.shape[1] != motion_frames
        ):
            raise RuntimeError(
                "full-clip audio features must have shape "
                f"(1, {motion_frames}, D), got {full_features.shape}"
            )
        windows = [
            full_features[:, start:start + seq_size, :]
            for start in window_start_frames
        ]
        face_contexts = []
        context_capacity = seq_size + 2 * FACE_CONTEXT_FRAMES
        for start in window_start_frames:
            context_start = max(0, start - FACE_CONTEXT_FRAMES)
            context_end = min(
                motion_frames,
                start + seq_size + FACE_CONTEXT_FRAMES,
            )
            valid_context = full_features[:, context_start:context_end, :]
            valid_length = int(valid_context.shape[1])
            crop_start = int(start - context_start)
            carrier = np.zeros(
                (1, context_capacity, full_features.shape[-1]),
                dtype=np.float32,
            )
            carrier[:, :valid_length, :] = valid_context
            if crop_start + seq_size > valid_length:
                raise RuntimeError(
                    "face context cannot contain the central motion window: "
                    f"start={start}, crop_start={crop_start}, "
                    f"valid_length={valid_length}, seq_size={seq_size}"
                )
            face_contexts.append(
                {
                    "features": carrier,
                    "valid_length": valid_length,
                    "crop_start": crop_start,
                }
            )
    else:
        windows = [
            np.asarray(
                audio_feature_extractor(audio_window, seq_size=seq_size),
                dtype=np.float32,
            )
            for audio_window in audio_windows
        ]
        face_contexts = None

    feature_dim = getattr(audio_feature_extractor, "feature_dim", None)
    for index, features in enumerate(windows):
        if (
            features.ndim != 3
            or features.shape[0] != 1
            or features.shape[1] != seq_size
        ):
            raise RuntimeError(
                f"audio feature window {index} must have shape "
                f"(1, {seq_size}, D), got {features.shape}"
            )
        if feature_dim is not None and features.shape[-1] != feature_dim:
            raise RuntimeError(
                f"audio feature window {index} has dimension "
                f"{features.shape[-1]}, expected {feature_dim}"
            )
        if not np.isfinite(features).all():
            raise RuntimeError(
                f"audio feature window {index} contains NaN/Inf"
            )
    return windows, face_contexts


def _json_ready_stats(stats):
    result = {}
    for key, value in stats.items():
        if isinstance(value, Counter):
            result[key] = dict(sorted(value.items()))
        elif isinstance(value, dict):
            result[key] = _json_ready_stats(value)
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def _bounded_console_stats(stats, duration_clip_limit=20):
    """Keep terminal/error summaries bounded while metadata retains all clips."""
    result = _json_ready_stats(stats)
    skipped_clips = result.get("duration_mismatch_skipped_clips")
    if (
        isinstance(skipped_clips, list)
        and len(skipped_clips) > duration_clip_limit
    ):
        result["duration_mismatch_skipped_clips"] = skipped_clips[
            :duration_clip_limit
        ]
        result["duration_mismatch_skipped_clips_omitted"] = (
            len(skipped_clips) - duration_clip_limit
        )
    return result


def build_lmdb_metadata(
    *,
    split,
    speakers,
    pid_offset,
    seq_size,
    stride_size,
    local_rotation,
    model_type,
    feature_scope,
    hand_component_path,
    hand_component_sha256,
    audio_provenance,
    audio_feature_dim,
    split_provenance,
    source_clip_count,
    output_sample_count,
    stats,
):
    speaker_ids = {
        name: SHOW_SPEAKER_ID[name] + int(pid_offset)
        for name in speakers
        if name in SHOW_SPEAKER_ID
    }
    metadata = {
        "schema_version": LMDB_META_SCHEMA_VERSION,
        "dataset": "SHOW",
        "split": str(split),
        "speakers": list(speakers),
        "speaker_ids": speaker_ids,
        "pid_offset": int(pid_offset),
        "seq_size": int(seq_size),
        "stride_size": int(stride_size),
        "local_rotation": bool(local_rotation),
        "rotation_space": "local" if local_rotation else "global",
        "pose_representation": "axis_angle",
        "pose_order": list(POSE_ORDER),
        "model_type": str(model_type),
        "audio_model_id": audio_provenance["audio_model_id"],
        "audio_config_sha256": audio_provenance["audio_config_sha256"],
        "audio_preprocessor_sha256": audio_provenance[
            "audio_preprocessor_sha256"
        ],
        "audio_model_files_sha256": dict(
            audio_provenance["audio_model_files_sha256"]
        ),
        "audio_do_normalize": audio_provenance["audio_do_normalize"],
        "audio_feat_extract_norm": audio_provenance[
            "audio_feat_extract_norm"
        ],
        "audio_add_adapter": audio_provenance["audio_add_adapter"],
        "audio_feature_dim": int(audio_feature_dim),
        "audio_sr": AUDIO_SR,
        "pose_fps": POSE_FPS,
        "feature_scope": str(feature_scope),
        "face_context_frames": (
            FACE_CONTEXT_FRAMES if feature_scope == "full_clip" else 0
        ),
        "face_context_protocol": (
            FACE_CONTEXT_PROTOCOL if feature_scope == "full_clip" else "none"
        ),
        "hand_component_path": str(Path(hand_component_path).resolve()),
        "hand_component_sha256": str(hand_component_sha256),
        **dict(split_provenance),
        "source_clip_count": int(source_clip_count),
        "output_sample_count": int(output_sample_count),
        "stats": _json_ready_stats(stats),
    }
    return metadata


def encode_lmdb_metadata(metadata):
    """Encode LMDB metadata as portable UTF-8 JSON (never pickle)."""
    return json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Main ETL — mirrors scripts/create_lmdb.py:create_lmdb structure.
# ---------------------------------------------------------------------------
def create_lmdb_show(
    data_folder,
    lmdb_path,
    seq_size,
    stride_size,
    local_rotation=True,
    split="train",
    speakers=None,
    model_type="wavlm",
    model_id=None,
    hand_component_path=str(DEFAULT_HAND_COMPONENT_PATH),
    pid_offset=0,
    feature_scope="window",
    max_av_duration_delta_frames=None,
    require_split_summary=False,
    duration_mismatch_policy="error",
):
    if speakers is None or len(speakers) == 0:
        speakers = ["oliver", "seth", "conan", "chemistry"]
    speakers = list(speakers)
    if len(set(speakers)) != len(speakers):
        raise ValueError(f"duplicate SHOW speakers are not allowed: {speakers}")
    unknown_speakers = [
        speaker for speaker in speakers if speaker not in SHOW_SPEAKER_ID
    ]
    if unknown_speakers:
        raise ValueError(
            f"unknown SHOW speakers: {sorted(unknown_speakers)}"
        )
    if split not in ("train", "val", "test"):
        raise ValueError(f"invalid SHOW split: {split!r}")
    if not isinstance(seq_size, int) or seq_size < 1:
        raise ValueError(f"seq_size must be positive, got {seq_size!r}")
    if not isinstance(stride_size, int) or stride_size < 1:
        raise ValueError(f"stride_size must be positive, got {stride_size!r}")
    if feature_scope not in FEATURE_SCOPES:
        raise ValueError(
            f"feature_scope must be one of {FEATURE_SCOPES}, "
            f"got {feature_scope!r}"
        )
    if (
        max_av_duration_delta_frames is not None
        and max_av_duration_delta_frames < 0
    ):
        raise ValueError(
            "max_av_duration_delta_frames must be non-negative, got "
            f"{max_av_duration_delta_frames}"
        )
    if duration_mismatch_policy not in DURATION_MISMATCH_POLICIES:
        raise ValueError(
            "duration_mismatch_policy must be one of "
            f"{DURATION_MISMATCH_POLICIES}, got "
            f"{duration_mismatch_policy!r}"
        )
    if (
        duration_mismatch_policy == "skip"
        and max_av_duration_delta_frames is None
    ):
        raise ValueError(
            "duration_mismatch_policy='skip' requires "
            "max_av_duration_delta_frames"
        )

    lh_c, rh_c = load_hand_pca(hand_component_path)
    hand_component_path = str(Path(hand_component_path).expanduser().resolve())
    hand_component_sha256 = sha256_file(hand_component_path)
    if (
        require_split_summary
        and hand_component_sha256 != OFFICIAL_HAND_COMPONENT_SHA256
    ):
        raise RuntimeError(
            "strict TalkSHOW generation requires the pinned canonical "
            "hand_component.json: "
            f"{hand_component_sha256} != "
            f"{OFFICIAL_HAND_COMPONENT_SHA256}"
        )
    items, scan_stats = scan_show_dataset(
        data_folder, speakers, split, return_stats=True
    )
    print(f"[scan] found {len(items)} clips for split='{split}', speakers={speakers}")
    if len(items) == 0:
        raise RuntimeError(
            "no usable SHOW clips found under "
            f"{data_folder} for split={split!r}; scan_stats={scan_stats}"
        )
    split_provenance = load_split_view_provenance(
        data_folder,
        split=split,
        speakers=speakers,
        usable_clip_count=len(items),
        required=bool(require_split_summary),
    )
    if model_id is None or not str(model_id).strip():
        raise ValueError(
            "model_id is required; pass the exact local WavLM directory used "
            "by TalkSHOW face training"
        )
    model_path = Path(str(model_id)).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(
            "strict TalkSHOW LMDB generation requires a local WavLM "
            f"snapshot directory, got: {model_path}"
        )
    model_id = str(model_path.resolve())

    final_lmdb_path = str(Path(lmdb_path).expanduser().resolve())
    if os.path.exists(final_lmdb_path):
        raise FileExistsError(
            "refusing to overwrite an existing LMDB; choose a new path or "
            f"move the old artifact first: {final_lmdb_path}"
        )
    audio_provenance = collect_audio_model_provenance(model_id, model_type)
    audio_feature_extractor = AudioFeatureExtractor(
        model_type=model_type, model_id=model_id
    )

    building_lmdb_path = (
        f"{final_lmdb_path}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    env = lmdb.open(
        building_lmdb_path,
        map_size=1024 * 1024 * 1024 * 1024,
        subdir=False,
    )
    txn = env.begin(write=True)

    dp_id = 0
    commit_interval = 500
    processing_stats = {
        "clips_seen": 0,
        "clips_validated": 0,
        "clips_written": 0,
        "clips_skipped_by_reason": Counter(),
        "candidate_windows": 0,
        "windows_written": 0,
        "windows_skipped_by_reason": Counter(),
        "short_clips": 0,
        "short_clips_retained": 0,
        "duration_checked_clips": 0,
        "duration_delta_frames_abs_sum": 0.0,
        "duration_delta_frames_abs_max": 0.0,
        "duration_mismatch_skipped_clips": [],
    }
    metadata = None
    build_succeeded = False
    try:
        for pkl_path, audio_file, speaker_name, seq_id in tqdm(
            items, desc=f"SHOW->{split}"
        ):
            processing_stats["clips_seen"] += 1
            pid = SHOW_SPEAKER_ID[speaker_name] + pid_offset

            try:
                raw_audio, _ = librosa.load(audio_file, sr=AUDIO_SR)
            except Exception as e:
                processing_stats["clips_skipped_by_reason"][
                    "audio_load_error"
                ] += 1
                print(
                    f"[skip] audio load error {audio_file}: {e}",
                    file=sys.stderr,
                )
                continue
            raw_audio = np.asarray(raw_audio, dtype=np.float32)
            if raw_audio.ndim != 1 or raw_audio.size == 0:
                processing_stats["clips_skipped_by_reason"][
                    "audio_shape"
                ] += 1
                print(
                    f"[skip] {audio_file}: expected non-empty mono waveform, "
                    f"got {raw_audio.shape}",
                    file=sys.stderr,
                )
                continue
            if not np.isfinite(raw_audio).all():
                processing_stats["clips_skipped_by_reason"][
                    "audio_nonfinite"
                ] += 1
                print(
                    f"[skip] {audio_file}: waveform contains NaN/Inf",
                    file=sys.stderr,
                )
                continue

            try:
                poses, expressions, trans, betas, T_pkl = load_show_clip(
                    pkl_path, lh_c, rh_c
                )
            except ShowClipValidationError as e:
                processing_stats["clips_skipped_by_reason"][e.reason] += 1
                print(f"[skip] {e}", file=sys.stderr)
                continue

            audio_frames = raw_audio.shape[0] * POSE_FPS / AUDIO_SR
            duration_delta_frames = audio_frames - T_pkl
            abs_duration_delta = abs(duration_delta_frames)
            processing_stats["duration_checked_clips"] += 1
            processing_stats["duration_delta_frames_abs_sum"] += (
                abs_duration_delta
            )
            processing_stats["duration_delta_frames_abs_max"] = max(
                processing_stats["duration_delta_frames_abs_max"],
                abs_duration_delta,
            )
            if (
                max_av_duration_delta_frames is not None
                and abs_duration_delta > max_av_duration_delta_frames
            ):
                mismatch_message = (
                    "SHOW audio/motion duration mismatch exceeds threshold: "
                    f"clip={speaker_name}/{seq_id}, motion_frames={T_pkl}, "
                    f"audio_frames={audio_frames:.3f}, "
                    f"delta_frames={duration_delta_frames:+.3f}, "
                    f"threshold={max_av_duration_delta_frames:.3f}"
                )
                if duration_mismatch_policy == "error":
                    raise RuntimeError(mismatch_message)
                processing_stats["clips_skipped_by_reason"][
                    "duration_mismatch"
                ] += 1
                processing_stats["duration_mismatch_skipped_clips"].append(
                    {
                        "clip": f"{speaker_name}/{seq_id}",
                        "motion_frames": int(T_pkl),
                        "audio_samples": int(raw_audio.shape[0]),
                        "audio_frames": float(audio_frames),
                        "delta_frames": float(duration_delta_frames),
                    }
                )
                print(f"[skip] {mismatch_message}", file=sys.stderr)
                continue
            processing_stats["clips_validated"] += 1

            # Match BEAT2 ETL semantics: local_rotation=False stores global AA.
            if not local_rotation:
                poses = LocalToGlobalAA(
                    poses, parent_idx=SMPLX_PARENTS
                ).astype(np.float32)

            file_name = f"{speaker_name}__{seq_id}"
            wrote_clip = False

            if split != "test":
                # Duration compatibility was already checked above.  Motion is
                # authoritative here: a contract-valid T-frame waveform has
                # length ``T * sr // fps``, which can round back to T-1 under
                # ``int(audio_len / sr * fps)`` (e.g. T=256 at 30 FPS).
                # Per-window audio slicing below remains the final exact-length
                # guard when a source waveform is genuinely too short.
                num_frames = T_pkl
                num_windows = (
                    (num_frames - seq_size) // stride_size + 1
                )
                if num_windows <= 0:
                    processing_stats["short_clips"] += 1
                    processing_stats["clips_skipped_by_reason"][
                        "shorter_than_seq_size"
                    ] += 1
                    continue
                processing_stats["candidate_windows"] += num_windows

                starts = [i * stride_size for i in range(num_windows)]
                audio_window_size = seq_size * AUDIO_SR // POSE_FPS
                valid_starts = []
                poses_lst = []
                expressions_lst = []
                trans_lst = []
                audio_lst = []
                for start in starts:
                    audio_start = start * AUDIO_SR // POSE_FPS
                    pose_window = poses[start:start + seq_size]
                    expression_window = expressions[start:start + seq_size]
                    trans_window = trans[start:start + seq_size]
                    audio_window = raw_audio[
                        audio_start:audio_start + audio_window_size
                    ]
                    if len(pose_window) != seq_size:
                        processing_stats["windows_skipped_by_reason"][
                            "motion_length"
                        ] += 1
                        continue
                    if len(audio_window) != audio_window_size:
                        processing_stats["windows_skipped_by_reason"][
                            "audio_length"
                        ] += 1
                        continue
                    valid_starts.append(start)
                    poses_lst.append(pose_window)
                    expressions_lst.append(expression_window)
                    trans_lst.append(trans_window)
                    audio_lst.append(audio_window)

                if not valid_starts:
                    processing_stats["clips_skipped_by_reason"][
                        "no_valid_windows"
                    ] += 1
                    continue
                feature_windows, face_context_windows = (
                    extract_window_audio_features(
                        audio_feature_extractor,
                        raw_audio,
                        audio_lst,
                        valid_starts,
                        motion_frames=T_pkl,
                        seq_size=seq_size,
                        feature_scope=feature_scope,
                    )
                )

                for window_index, (
                    pose_window,
                    expression_window,
                    trans_window,
                    audio_window,
                    huberts,
                ) in enumerate(
                    zip(
                        poses_lst,
                        expressions_lst,
                        trans_lst,
                        audio_lst,
                        feature_windows,
                    )
                ):
                    datum = {
                        "betas": betas,
                        "poses": pose_window.astype(np.float32),
                        "expressions": expression_window.astype(np.float32),
                        "trans": trans_window.astype(np.float32),
                        "pid": pid,
                        "audio": audio_window.astype(np.float32),
                        "huberts": huberts.astype(np.float32),
                        "ori_vid": file_name,
                    }
                    if face_context_windows is not None:
                        face_context = face_context_windows[window_index]
                        datum.update(
                            {
                                "face_huberts": face_context[
                                    "features"
                                ].astype(np.float32),
                                "face_hubert_valid_length": int(
                                    face_context["valid_length"]
                                ),
                                "face_hubert_crop_start": int(
                                    face_context["crop_start"]
                                ),
                            }
                        )
                    key = f"{dp_id:010d}".encode("ascii")
                    txn.put(key, pickle.dumps(datum))
                    dp_id += 1
                    processing_stats["windows_written"] += 1
                    wrote_clip = True
                    if dp_id % commit_interval == 0:
                        txn.commit()
                        txn = env.begin(write=True)
            else:
                # Test remains one full-length record per clip.
                target_audio_len = AUDIO_SR * T_pkl // POSE_FPS
                if raw_audio.shape[0] > target_audio_len:
                    aligned_audio = raw_audio[:target_audio_len]
                elif raw_audio.shape[0] < target_audio_len:
                    aligned_audio = np.pad(
                        raw_audio,
                        (0, target_audio_len - raw_audio.shape[0]),
                        mode="constant",
                    )
                else:
                    aligned_audio = raw_audio

                huberts = np.asarray(
                    audio_feature_extractor(
                        aligned_audio, seq_size=T_pkl
                    ),
                    dtype=np.float32,
                )
                if (
                    huberts.ndim != 3
                    or huberts.shape[0] != 1
                    or huberts.shape[1] != T_pkl
                    or not np.isfinite(huberts).all()
                ):
                    raise RuntimeError(
                        "test audio features must have shape "
                        f"(1, {T_pkl}, D) and be finite; got {huberts.shape}"
                    )

                datum = {
                    "betas": betas,
                    "poses": poses.astype(np.float32),
                    "expressions": expressions.astype(np.float32),
                    "trans": trans.astype(np.float32),
                    "pid": pid,
                    "audio": aligned_audio.astype(np.float32),
                    "huberts": huberts,
                    # Test runs the face head once on the complete source
                    # sequence, so its valid context is exactly the full clip.
                    "face_huberts": huberts.copy(),
                    "face_hubert_valid_length": int(T_pkl),
                    "face_hubert_crop_start": 0,
                    "ori_vid": file_name,
                }
                key = f"{dp_id:010d}".encode("ascii")
                txn.put(key, pickle.dumps(datum))
                dp_id += 1
                processing_stats["windows_written"] += 1
                wrote_clip = True
                if dp_id % commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            if wrote_clip:
                processing_stats["clips_written"] += 1

        validated_clips = processing_stats["clips_validated"]
        written_clips = processing_stats["clips_written"]
        short_clips = processing_stats["short_clips"]
        processing_stats["clip_retention_rate"] = (
            written_clips / validated_clips if validated_clips else 0.0
        )
        processing_stats["short_clip_retention_rate"] = (
            processing_stats["short_clips_retained"] / short_clips
            if short_clips
            else 1.0
        )
        duration_count = processing_stats["duration_checked_clips"]
        processing_stats["duration_delta_frames_abs_mean"] = (
            processing_stats["duration_delta_frames_abs_sum"]
            / duration_count
            if duration_count
            else 0.0
        )
        duration_mismatch_skipped_count = len(
            processing_stats["duration_mismatch_skipped_clips"]
        )
        processing_stats["duration_mismatch_skipped_count"] = (
            duration_mismatch_skipped_count
        )
        processing_stats["duration_mismatch_skip_rate"] = (
            duration_mismatch_skipped_count / duration_count
            if duration_count
            else 0.0
        )

        if dp_id == 0:
            raise RuntimeError(
                "SHOW ETL produced zero LMDB records; "
                f"processing_stats={_bounded_console_stats(processing_stats)}"
            )
        feature_dim = getattr(audio_feature_extractor, "feature_dim", None)
        if feature_dim is None or int(feature_dim) < 1:
            raise RuntimeError(
                f"could not determine audio feature dimension: {feature_dim!r}"
            )
        configured_dim = audio_provenance["configured_audio_feature_dim"]
        if configured_dim is not None and int(feature_dim) != configured_dim:
            raise RuntimeError(
                "audio feature dimension does not match config.json: "
                f"observed={feature_dim}, configured={configured_dim}"
            )

        effective_feature_scope = (
            "full_clip" if split == "test" else feature_scope
        )
        all_stats = {
            "scan": scan_stats,
            "processing": processing_stats,
        }
        metadata = build_lmdb_metadata(
            split=split,
            speakers=speakers,
            pid_offset=pid_offset,
            seq_size=seq_size,
            stride_size=stride_size,
            local_rotation=local_rotation,
            model_type=model_type,
            feature_scope=effective_feature_scope,
            hand_component_path=hand_component_path,
            hand_component_sha256=hand_component_sha256,
            audio_provenance=audio_provenance,
            audio_feature_dim=feature_dim,
            split_provenance=split_provenance,
            source_clip_count=len(items),
            output_sample_count=dp_id,
            stats=all_stats,
        )
        metadata["requested_feature_scope"] = feature_scope
        metadata["max_av_duration_delta_frames"] = (
            None
            if max_av_duration_delta_frames is None
            else float(max_av_duration_delta_frames)
        )
        metadata["duration_mismatch_policy"] = duration_mismatch_policy
        txn.put(b"__len__", str(dp_id).encode("ascii"))
        txn.put(LMDB_META_KEY, encode_lmdb_metadata(metadata))
        txn.commit()
        txn = None
        build_succeeded = True
    finally:
        if txn is not None:
            try:
                txn.abort()
            except Exception:
                pass
        env.close()
        if not build_succeeded:
            for partial_path in (
                building_lmdb_path,
                building_lmdb_path + "-lock",
            ):
                try:
                    os.unlink(partial_path)
                except FileNotFoundError:
                    pass

    # A file-backed LMDB is atomically published only after all records and
    # metadata have committed. Lightweight unit-test environments may provide
    # an in-memory lmdb stub and therefore have no file to rename.
    if os.path.isfile(building_lmdb_path):
        os.replace(building_lmdb_path, final_lmdb_path)
        try:
            os.unlink(building_lmdb_path + "-lock")
        except FileNotFoundError:
            pass

    printable_stats = _bounded_console_stats(processing_stats)
    print(f"[stats] {json.dumps(printable_stats, sort_keys=True)}")
    if max_av_duration_delta_frames is not None:
        mismatch_count = processing_stats["duration_mismatch_skipped_count"]
        checked_count = processing_stats["duration_checked_clips"]
        mismatch_rate = processing_stats["duration_mismatch_skip_rate"]
        print(
            "[duration-mismatch] "
            f"policy={duration_mismatch_policy}, "
            f"threshold_frames={max_av_duration_delta_frames:.3f}, "
            f"skipped_clips={mismatch_count}, "
            f"checked_clips={checked_count}, "
            f"skip_rate={mismatch_rate:.6%}"
        )
    print(
        f"[done] LMDB created at {final_lmdb_path} with {dp_id} samples; "
        f"feature_scope={metadata['feature_scope']}, "
        f"audio_feature_dim={metadata['audio_feature_dim']}."
    )
    return metadata


def create():
    parser = argparse.ArgumentParser(
        description="Create BEAT2-style LMDB from the SHOW dataset"
    )
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Path to SHOW dataset root "
            "(contains <speaker>/<vid>/<split>/<seq>/...)"
        ),
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=["oliver", "seth", "conan", "chemistry"],
        help="Speaker subdirectories to include",
    )
    parser.add_argument(
        "--seq_size",
        type=int,
        default=256,
        help="Sequence size in frames",
    )
    parser.add_argument(
        "--stride_size",
        type=int,
        default=20,
        help="Stride size in frames (train/val only)",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default="",
        help="Path to save the lmdb (auto-named if empty)",
    )
    parser.add_argument(
        "--local_rotation",
        type=int,
        default=0,
        help=(
            "Use local rotation (1) or global (0). "
            "Matches BEAT2 ETL semantics."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wavlm",
        choices=["wavlm", "hubert"],
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Exact local path to the audio feature model",
    )
    parser.add_argument(
        "--hand_component_path",
        type=str,
        default=str(DEFAULT_HAND_COMPONENT_PATH),
        help=(
            "Path to TalkSHOW hand_component.json (12-PCA basis); "
            "defaults to the repository tools/show asset"
        ),
    )
    parser.add_argument(
        "--pid_offset",
        type=int,
        default=0,
        help=(
            "Offset added to SHOW_SPEAKER_ID; set to 31 to avoid clashing "
            "with BEAT2 pids"
        ),
    )
    parser.add_argument(
        "--require_split_summary",
        action="store_true",
        help=(
            "Require split_view_summary.json from prepare_split_view.py and "
            "bind the LMDB to the pinned official split manifest"
        ),
    )
    parser.add_argument(
        "--feature_scope",
        type=str,
        default="window",
        choices=FEATURE_SCOPES,
        help=(
            "Audio extraction scope: canonical full_clip encodes once before "
            "windowing; window preserves legacy per-window extraction"
        ),
    )
    parser.add_argument(
        "--max_av_duration_delta_frames",
        type=float,
        default=None,
        help=(
            "Threshold for absolute audio/motion duration mismatch in "
            "30-FPS frames; omitted preserves the legacy no-limit mode"
        ),
    )
    parser.add_argument(
        "--duration_mismatch_policy",
        type=str,
        default="error",
        choices=DURATION_MISMATCH_POLICIES,
        help=(
            "Action when the duration threshold is exceeded: error preserves "
            "strict fail-closed behavior; skip excludes the complete clip "
            "before WavLM extraction and requires a threshold"
        ),
    )
    args = parser.parse_args()

    if args.lmdb_path == "":
        local_global = "local" if args.local_rotation != 0 else "global"
        args.lmdb_path = (
            f"./show_{args.split}_seq_size_{args.seq_size}"
            f"_stride_size_{args.stride_size}_{local_global}.lmdb"
        )

    parent = os.path.dirname(args.lmdb_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    create_lmdb_show(
        data_folder=args.data_folder,
        lmdb_path=args.lmdb_path,
        seq_size=args.seq_size,
        stride_size=args.stride_size,
        local_rotation=args.local_rotation != 0,
        split=args.split,
        speakers=args.speakers,
        model_type=args.model_type,
        model_id=args.model_id,
        hand_component_path=args.hand_component_path,
        pid_offset=args.pid_offset,
        feature_scope=args.feature_scope,
        max_av_duration_delta_frames=args.max_av_duration_delta_frames,
        require_split_summary=args.require_split_summary,
        duration_mismatch_policy=args.duration_mismatch_policy,
    )


"""
Example usage (mirrors create_lmdb.py example block):

    CUDA_VISIBLE_DEVICES=1 python scripts/create_lmdb_show.py \
        /path/to/SHOW_dataset_v1.0/ExpressiveWholeBodyDatasetReleaseV1.0 \
        --split train --seq_size 256 --stride_size 20 --local_rotation 0 \
        --speakers oliver seth conan chemistry \
        --model_type wavlm \
        --model_id /absolute/path/to/wavlm-libri-clean-100h-large \
        --require_split_summary \
        --feature_scope full_clip --max_av_duration_delta_frames 3 \
        --duration_mismatch_policy skip

    CUDA_VISIBLE_DEVICES=4 python scripts/create_lmdb_show.py \
        /path/to/SHOW_dataset_v1.0/ExpressiveWholeBodyDatasetReleaseV1.0 \
        --split val --seq_size 256 --stride_size 20 --local_rotation 0 \
        --speakers oliver seth conan chemistry \
        --model_type wavlm \
        --model_id /absolute/path/to/wavlm-libri-clean-100h-large \
        --require_split_summary \
        --feature_scope full_clip --max_av_duration_delta_frames 3 \
        --duration_mismatch_policy skip

    CUDA_VISIBLE_DEVICES=5 python scripts/create_lmdb_show.py \
        /path/to/SHOW_dataset_v1.0/ExpressiveWholeBodyDatasetReleaseV1.0 \
        --split test --seq_size 256 --stride_size 20 --local_rotation 0 \
        --speakers oliver seth conan chemistry \
        --model_type wavlm \
        --model_id /absolute/path/to/wavlm-libri-clean-100h-large \
        --require_split_summary \
        --feature_scope full_clip --max_av_duration_delta_frames 3 \
        --duration_mismatch_policy skip
"""
if __name__ == "__main__":
    create()
