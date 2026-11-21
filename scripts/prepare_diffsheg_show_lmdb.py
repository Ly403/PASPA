#!/usr/bin/env python3
"""Build the missing DiffSHEG SHOW test caches from a TalkSHOW split view.

DiffSHEG hard-codes the following two directory-backed LMDB environments but
does not publish the script that creates them::

    <DiffSHEG>/data/SHOW/cached_data/talkshow_test_cache
    <DiffSHEG>/data/SHOW/cached_aud_hubert/test/hubert_large_ls960_ft

This adapter builds both caches without changing the source split view.  Every
test clip remains one complete, variable-length record; it is not sliced to the
64/128/256-frame training profiles used by the main project.

The cache layout follows DiffSHEG ``datasets/show.py``:

* keys are contiguous five-digit ASCII values (``00000``, ``00001``, ...);
* the main value is a legacy PyArrow-serialized 8-tuple
  ``(pose, expression, raw, mfcc, mel, speaker, wav_path, betas)``;
* the HuBERT value under the same key is a serialized ``[Th, 1024]`` array;
* pose is local axis-angle in native TalkSHOW order
  ``jaw, eyes, global, body, left hand, right hand``;
* speaker IDs retain TalkSHOW's native values 20--23 because the public
  DiffSHEG trainer subtracts 20 internally.

The original DiffSHEG cache builder is unavailable.  The 128-bin mel features
therefore use its public custom-audio recipe (18 kHz, power mel) with a
30-FPS-compatible hop of 600 samples.  This choice and all source/output
records are written to ``diffsheg_show_test_cache_summary.json``.

Run this script in the legacy DiffSHEG environment because its reader requires
``pyarrow==3.0.0`` and ``pyarrow.serialize``::

    python scripts/prepare_diffsheg_show_lmdb.py \
      --split-view-root /absolute/path/to/talkshow_split_view \
      --diffsheg-root /absolute/path/to/DiffSHEG \
      --hubert-model /absolute/path/to/facebook-hubert-large-ls960-ft \
      --device cuda:0 \
      --require-split-summary \
      --max-av-duration-delta-frames 3 \
      --invalid-clip-policy skip

Omit ``--hubert-model`` to resolve ``facebook/hubert-large-ls960-ft`` through
Hugging Face.  Existing cache directories are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DIFFSHEG_REFERENCE_COMMIT = "3ebf3058f48cba3da9146afb7623e9ec1ab9e9a5"
POSE_FPS = 30
AUDIO_SR = 16000
MEL_SR = 18000
MEL_HOP_LENGTH = MEL_SR // POSE_FPS
MEL_DIM = 128
MFCC_DIM = 64
HUBERT_DIM = 1024
MAIN_CACHE_RELATIVE = Path("cached_data") / "talkshow_test_cache"
HUBERT_CACHE_RELATIVE = (
    Path("cached_aud_hubert")
    / "test"
    / "hubert_large_ls960_ft"
)
SUMMARY_FILENAME = "diffsheg_show_test_cache_summary.json"
DEFAULT_HUBERT_MODEL = "facebook/hubert-large-ls960-ft"
DEFAULT_SPEAKERS = ("oliver", "seth", "conan", "chemistry")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_COMPONENT_PATH = (
    REPO_ROOT / "tools" / "show" / "hand_component.json"
)


class CacheBuildError(RuntimeError):
    """A cache or source artifact violates the DiffSHEG adapter contract."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _load_source_helpers() -> Any:
    """Load the audited TalkSHOW motion parser from the existing ETL."""

    repository = str(REPO_ROOT)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    try:
        from scripts import create_lmdb_show
    except Exception as exc:
        raise CacheBuildError(
            "cannot import scripts/create_lmdb_show.py; run this command from "
            f"the main codebase with its dependencies installed: {exc}"
        ) from exc
    return create_lmdb_show


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import librosa
        import lmdb
        import numpy as np
        import pyarrow
    except ImportError as exc:
        raise CacheBuildError(
            "missing cache-build dependency. Use the DiffSHEG environment "
            "(notably lmdb and pyarrow==3.0.0): "
            f"{exc}"
        ) from exc
    if not callable(getattr(pyarrow, "serialize", None)):
        raise CacheBuildError(
            "this script requires legacy pyarrow.serialize; DiffSHEG pins "
            "pyarrow==3.0.0. Newer PyArrow releases removed this API."
        )
    if not callable(getattr(pyarrow, "deserialize", None)):
        raise CacheBuildError(
            "this script requires legacy pyarrow.deserialize; use the exact "
            "DiffSHEG data environment."
        )
    return librosa, lmdb, np, pyarrow


def _validate_diffsheg_layout(
    diffsheg_root: Path,
    np: Any,
) -> tuple[Path, Path]:
    root = diffsheg_root.expanduser().resolve()
    if not root.is_dir():
        raise CacheBuildError(f"DiffSHEG root is not a directory: {root}")
    reader = root / "datasets" / "show.py"
    if not reader.is_file():
        raise CacheBuildError(
            f"DiffSHEG datasets/show.py does not exist under {root}"
        )
    show_root = root / "data" / "SHOW"
    stats_path = show_root / "talkshow_mean_std.npy"
    if not stats_path.is_file():
        raise CacheBuildError(
            "DiffSHEG SHOW statistics are missing. Extract "
            f"{root / 'assets' / 'data.tar.gz'} first; expected {stats_path}"
        )
    try:
        loaded = np.load(stats_path, allow_pickle=True)
        payload = loaded.item() if getattr(loaded, "shape", None) == () else loaded
    except Exception as exc:
        raise CacheBuildError(
            f"cannot read DiffSHEG statistics {stats_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CacheBuildError(
            f"DiffSHEG statistics must contain a mapping: {stats_path}"
        )
    expected = {
        "pose_mean": 165,
        "pose_std": 165,
        "expression_mean": 100,
        "expression_std": 100,
    }
    for key, size in expected.items():
        if key not in payload or np.asarray(payload[key]).size != size:
            observed = None if key not in payload else np.asarray(payload[key]).shape
            raise CacheBuildError(
                f"{stats_path}: {key} must contain {size} values, got {observed}"
            )
    return show_root, stats_path


def _align_feature_frames(
    features: Any,
    frame_count: int,
    *,
    label: str,
    np: Any,
) -> Any:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0:
        raise CacheBuildError(f"{label} must be [T,D], got {array.shape}")
    delta = int(array.shape[0]) - frame_count
    if delta >= 0:
        array = array[:frame_count]
    elif delta == -1:
        array = np.pad(array, ((0, 1), (0, 0)), mode="edge")
    else:
        raise CacheBuildError(
            f"{label} produced {array.shape[0]} frames for {frame_count} "
            "motion frames"
        )
    if array.shape[0] != frame_count or not np.isfinite(array).all():
        raise CacheBuildError(
            f"{label} alignment produced invalid shape/data: {array.shape}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def _waveform_at_motion_fps(audio: Any, frame_count: int, np: Any) -> Any:
    if frame_count < 1 or audio.size < 1:
        raise CacheBuildError("cannot frame an empty waveform")
    if frame_count == 1:
        values = np.asarray([float(audio.mean())], dtype=np.float32)
    else:
        positions = np.linspace(
            0.0,
            float(audio.shape[0] - 1),
            num=frame_count,
            dtype=np.float64,
        )
        values = np.interp(
            positions,
            np.arange(audio.shape[0], dtype=np.float64),
            audio,
        ).astype(np.float32)
    return values.reshape(frame_count, 1)


def _extract_classic_audio_features(
    audio_16k: Any,
    frame_count: int,
    *,
    librosa: Any,
    np: Any,
) -> tuple[Any, Any, Any]:
    audio_18k = librosa.resample(
        audio_16k,
        orig_sr=AUDIO_SR,
        target_sr=MEL_SR,
    )
    mel_frequency_first = librosa.feature.melspectrogram(
        y=audio_18k,
        sr=MEL_SR,
        n_fft=2048,
        hop_length=MEL_HOP_LENGTH,
        n_mels=MEL_DIM,
        power=2.0,
    )
    mel = _align_feature_frames(
        mel_frequency_first.T,
        frame_count,
        label="mel",
        np=np,
    )
    mfcc_frequency_first = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_frequency_first, ref=1.0),
        n_mfcc=MFCC_DIM,
    )
    mfcc = _align_feature_frames(
        mfcc_frequency_first.T,
        frame_count,
        label="mfcc",
        np=np,
    )
    raw = _waveform_at_motion_fps(audio_16k, frame_count, np)
    if raw.shape != (frame_count, 1):
        raise CacheBuildError(f"raw audio feature has invalid shape {raw.shape}")
    return raw, mfcc, mel


class HubertExtractor:
    """DiffSHEG's public long-audio HuBERT-Large extraction routine."""

    def __init__(
        self,
        model_id: str,
        device_name: str,
        *,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from transformers import HubertModel, Wav2Vec2Processor
        except ImportError as exc:
            raise CacheBuildError(
                f"HuBERT extraction requires torch and transformers: {exc}"
            ) from exc
        self.torch = torch
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CacheBuildError(
                f"requested {self.device}, but CUDA is unavailable"
            )
        if self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else 0
            torch.cuda.set_device(index)
            self.device = torch.device(f"cuda:{index}")
        kwargs = {"local_files_only": True} if local_files_only else {}
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_id, **kwargs
            )
            self.model = HubertModel.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            raise CacheBuildError(
                f"cannot load HuBERT model/processor {model_id!r}: {exc}"
            ) from exc
        hidden_size = int(getattr(self.model.config, "hidden_size", -1))
        if hidden_size != HUBERT_DIM:
            raise CacheBuildError(
                "DiffSHEG SHOW requires HuBERT Large hidden_size=1024, "
                f"got {hidden_size} from {model_id!r}"
            )
        self.model.requires_grad_(False).eval().to(self.device)
        self.model_id = str(model_id)
        self.resolved_name = str(
            getattr(self.model.config, "_name_or_path", model_id)
        )

    def __call__(self, audio_16k: Any) -> Any:
        torch = self.torch
        try:
            input_values = self.processor(
                audio_16k,
                return_tensors="pt",
                sampling_rate=AUDIO_SR,
            ).input_values
        except Exception as exc:
            raise CacheBuildError(
                f"HuBERT processor rejected the waveform: {exc}"
            ) from exc
        if input_values.ndim != 2 or input_values.shape[0] != 1:
            raise CacheBuildError(
                f"HuBERT processor returned {tuple(input_values.shape)}"
            )

        kernel = 400
        stride = 320
        clip_length = stride * 1000
        sample_count = int(input_values.shape[1])
        if sample_count < kernel:
            raise CacheBuildError(
                f"audio has {sample_count} samples; HuBERT needs at least {kernel}"
            )
        expected_steps = (sample_count - (kernel - stride)) // stride
        num_iter = sample_count // clip_length
        features: list[Any] = []
        with torch.no_grad():
            for index in range(num_iter):
                if index == 0:
                    start = 0
                    stop = clip_length - stride + kernel
                else:
                    start = clip_length * index
                    stop = start + clip_length - stride + kernel
                segment = input_values[:, start:stop].to(self.device)
                output = self.model(segment).last_hidden_state[0]
                features.append(output.detach().cpu())
            remainder_start = clip_length * num_iter if num_iter > 0 else 0
            remainder = input_values[:, remainder_start:]
            if remainder.shape[1] >= kernel:
                output = self.model(
                    remainder.to(self.device)
                ).last_hidden_state[0]
                features.append(output.detach().cpu())
        if not features:
            raise CacheBuildError("HuBERT produced no feature segments")
        result = torch.cat(features, dim=0)
        if result.shape[0] < expected_steps:
            result = torch.nn.functional.pad(
                result,
                (0, 0, 0, expected_steps - result.shape[0]),
            )
        else:
            result = result[:expected_steps]
        array = result.numpy().astype("float32", copy=False)
        if (
            array.ndim != 2
            or array.shape != (expected_steps, HUBERT_DIM)
            or not bool(torch.isfinite(result).all())
        ):
            raise CacheBuildError(
                f"HuBERT produced invalid features {array.shape}"
            )
        return array


def _serialize(pyarrow: Any, value: Any, *, label: str) -> bytes:
    try:
        buffer = pyarrow.serialize(value).to_buffer()
        return buffer.to_pybytes()
    except Exception as exc:
        raise CacheBuildError(f"cannot PyArrow-serialize {label}: {exc}") from exc


def _talkshow_pose_order(beat_pose: Any, np: Any) -> Any:
    if beat_pose.ndim != 2 or beat_pose.shape[1] != 165:
        raise CacheBuildError(
            f"source motion parser returned invalid pose {beat_pose.shape}"
        )
    talkshow_pose = np.concatenate(
        (
            beat_pose[:, 66:75],
            beat_pose[:, 0:66],
            beat_pose[:, 75:165],
        ),
        axis=1,
    )
    if talkshow_pose.shape != beat_pose.shape:
        raise CacheBuildError(
            f"TalkSHOW pose reorder produced {talkshow_pose.shape}"
        )
    return np.ascontiguousarray(talkshow_pose, dtype=np.float32)


def _validate_main_payload(
    payload: Sequence[Any],
    frame_count: int,
    *,
    np: Any,
) -> None:
    if len(payload) != 8:
        raise CacheBuildError(f"main cache payload has {len(payload)} fields")
    pose, expression, raw, mfcc, mel, speaker, wav_path, betas = payload
    expected_shapes = (
        ("pose", pose, (frame_count, 165)),
        ("expression", expression, (frame_count, 100)),
        ("raw", raw, (frame_count, 1)),
        ("mfcc", mfcc, (frame_count, MFCC_DIM)),
        ("mel", mel, (frame_count, MEL_DIM)),
        ("betas", betas, (300,)),
    )
    for label, array, expected in expected_shapes:
        value = np.asarray(array)
        if value.shape != expected or value.dtype != np.float32:
            raise CacheBuildError(
                f"{label} must be float32 {expected}, got "
                f"{value.shape}/{value.dtype}"
            )
        if not np.isfinite(value).all():
            raise CacheBuildError(f"{label} contains NaN/Inf")
    speaker_value = int(np.asarray(speaker).item())
    if speaker_value not in (20, 21, 22, 23):
        raise CacheBuildError(
            f"DiffSHEG SHOW speaker ID must be 20..23, got {speaker_value}"
        )
    if not isinstance(wav_path, str) or not wav_path:
        raise CacheBuildError("main cache wav_path must be a non-empty string")


def _verify_cache_pair(
    main_path: Path,
    hubert_path: Path,
    expected_entries: int,
    *,
    lmdb: Any,
    np: Any,
    pyarrow: Any,
) -> None:
    environments = []
    try:
        for path in (main_path, hubert_path):
            environment = lmdb.open(
                str(path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=8,
            )
            environments.append(environment)
            # python-lmdb < 2.2 requires an explicit database handle for
            # Transaction.stat(db). Environment.stat() has always addressed
            # the unnamed main database used by DiffSHEG, so it works with
            # both DiffSHEG's pinned lmdb==0.96 and current releases.
            entries = int(environment.stat()["entries"])
            with environment.begin(write=False) as transaction:
                if entries != expected_entries:
                    raise CacheBuildError(
                        f"{path}: {entries} entries != {expected_entries}"
                    )
                for index, (key, _) in enumerate(transaction.cursor()):
                    expected_key = f"{index:05d}".encode("ascii")
                    if key != expected_key:
                        raise CacheBuildError(
                            f"{path}: key {key!r} != {expected_key!r}"
                        )
        if expected_entries < 1:
            raise CacheBuildError("cache contains no records")
        check_indices = sorted({0, expected_entries - 1})
        with environments[0].begin(write=False) as main_txn, environments[
            1
        ].begin(write=False) as hubert_txn:
            for index in check_indices:
                key = f"{index:05d}".encode("ascii")
                main_raw = main_txn.get(key)
                hubert_raw = hubert_txn.get(key)
                if main_raw is None or hubert_raw is None:
                    raise CacheBuildError(f"missing paired key {key!r}")
                payload = pyarrow.deserialize(main_raw)
                frame_count = int(np.asarray(payload[0]).shape[0])
                _validate_main_payload(payload, frame_count, np=np)
                hubert = np.asarray(
                    pyarrow.deserialize(hubert_raw), dtype=np.float32
                )
                if (
                    hubert.ndim != 2
                    or hubert.shape[0] < 1
                    or hubert.shape[1] != HUBERT_DIM
                    or not np.isfinite(hubert).all()
                ):
                    raise CacheBuildError(
                        f"key {key!r}: invalid HuBERT {hubert.shape}"
                    )
    finally:
        for environment in environments:
            environment.close()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.building-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_ready(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_caches(args: argparse.Namespace) -> Mapping[str, Any]:
    librosa, lmdb, np, pyarrow = _load_runtime_dependencies()
    helpers = _load_source_helpers()

    split_root = args.split_view_root.expanduser().resolve()
    if not split_root.is_dir():
        raise CacheBuildError(
            f"TalkSHOW split-view root is not a directory: {split_root}"
        )
    show_root, stats_path = _validate_diffsheg_layout(
        args.diffsheg_root,
        np,
    )
    hand_path = args.hand_component_path.expanduser().resolve()
    if not hand_path.is_file():
        raise CacheBuildError(
            f"TalkSHOW hand PCA component file does not exist: {hand_path}"
        )
    if len(set(args.speakers)) != len(args.speakers):
        raise CacheBuildError(f"duplicate speakers: {args.speakers}")
    unsupported = sorted(
        set(args.speakers).difference(helpers.CANONICAL_TALKSHOW_SPEAKERS)
    )
    if unsupported:
        raise CacheBuildError(
            f"DiffSHEG SHOW supports only four canonical speakers: {unsupported}"
        )
    if (
        args.invalid_clip_policy == "skip"
        and args.max_av_duration_delta_frames is None
    ):
        raise CacheBuildError(
            "--invalid-clip-policy skip requires "
            "--max-av-duration-delta-frames"
        )

    items, scan_stats = helpers.scan_show_dataset(
        str(split_root),
        list(args.speakers),
        "test",
        return_stats=True,
    )
    if not items:
        raise CacheBuildError(
            f"no TalkSHOW test clips found under {split_root}: {scan_stats}"
        )
    split_provenance = helpers.load_split_view_provenance(
        str(split_root),
        split="test",
        speakers=list(args.speakers),
        usable_clip_count=len(items),
        required=bool(args.require_split_summary),
    )
    left_hand, right_hand = helpers.load_hand_pca(hand_path)

    main_final = show_root / MAIN_CACHE_RELATIVE
    hubert_final = show_root / HUBERT_CACHE_RELATIVE
    summary_path = show_root / SUMMARY_FILENAME
    occupied = [
        path
        for path in (main_final, hubert_final, summary_path)
        if path.exists()
    ]
    if occupied:
        raise CacheBuildError(
            "refusing to overwrite existing DiffSHEG cache artifacts: "
            + ", ".join(str(path) for path in occupied)
        )
    main_final.parent.mkdir(parents=True, exist_ok=True)
    hubert_final.parent.mkdir(parents=True, exist_ok=True)

    build_token = f"{os.getpid()}-{uuid.uuid4().hex}"
    main_staging = main_final.with_name(
        f".{main_final.name}.building-{build_token}"
    )
    hubert_staging = hubert_final.with_name(
        f".{hubert_final.name}.building-{build_token}"
    )
    map_size = int(args.map_size_gb * 1024**3)
    if map_size < 1024**3:
        raise CacheBuildError("--map-size-gb must be at least 1")

    hubert_extractor = HubertExtractor(
        args.hubert_model,
        args.device,
        local_files_only=bool(args.local_files_only),
    )
    main_environment = None
    hubert_environment = None
    main_transaction = None
    hubert_transaction = None
    published: list[Path] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    skipped_by_reason: Counter[str] = Counter()
    duration_abs_max = 0.0
    duration_abs_sum = 0.0
    duration_checked = 0

    try:
        main_environment = lmdb.open(
            str(main_staging),
            map_size=map_size,
            subdir=True,
            create=True,
            lock=True,
            max_readers=32,
        )
        hubert_environment = lmdb.open(
            str(hubert_staging),
            map_size=map_size,
            subdir=True,
            create=True,
            lock=True,
            max_readers=32,
        )
        main_transaction = main_environment.begin(write=True)
        hubert_transaction = hubert_environment.begin(write=True)

        try:
            from tqdm import tqdm

            iterator = tqdm(items, desc="TalkSHOW test -> DiffSHEG LMDB")
        except ImportError:
            iterator = items

        for source_index, (
            pkl_path_raw,
            wav_path_raw,
            speaker,
            sequence,
        ) in enumerate(iterator):
            pkl_path = Path(pkl_path_raw)
            wav_path = Path(wav_path_raw)
            relative = pkl_path.relative_to(split_root)
            video = relative.parts[1]
            clip_name = f"{speaker}/{video}/{sequence}"
            try:
                beat_pose, expression, _, betas, frame_count = (
                    helpers.load_show_clip(
                        str(pkl_path),
                        left_hand,
                        right_hand,
                    )
                )
                talkshow_pose = _talkshow_pose_order(beat_pose, np)
                audio_16k, observed_sr = librosa.load(
                    str(wav_path),
                    sr=AUDIO_SR,
                    mono=True,
                    dtype=np.float32,
                )
                if observed_sr != AUDIO_SR:
                    raise CacheBuildError(
                        f"librosa returned sample rate {observed_sr}"
                    )
                audio_16k = np.asarray(audio_16k, dtype=np.float32)
                if (
                    audio_16k.ndim != 1
                    or audio_16k.size == 0
                    or not np.isfinite(audio_16k).all()
                ):
                    raise CacheBuildError(
                        f"invalid mono waveform {audio_16k.shape}"
                    )
            except helpers.ShowClipValidationError as exc:
                reason = str(exc.reason)
                if args.invalid_clip_policy == "error":
                    raise CacheBuildError(
                        f"{clip_name}: {reason}: {exc}"
                    ) from exc
                skipped_by_reason[reason] += 1
                skipped.append(
                    {
                        "source_index": source_index,
                        "clip": clip_name,
                        "reason": reason,
                        "detail": str(exc),
                    }
                )
                print(
                    f"[skip] {clip_name}: {reason}: {exc}",
                    file=sys.stderr,
                )
                continue
            except (OSError, ValueError, CacheBuildError) as exc:
                reason = "source_audio_or_schema"
                if args.invalid_clip_policy == "error":
                    raise CacheBuildError(
                        f"{clip_name}: {reason}: {exc}"
                    ) from exc
                skipped_by_reason[reason] += 1
                skipped.append(
                    {
                        "source_index": source_index,
                        "clip": clip_name,
                        "reason": reason,
                        "detail": str(exc),
                    }
                )
                print(
                    f"[skip] {clip_name}: {reason}: {exc}",
                    file=sys.stderr,
                )
                continue

            audio_frames = audio_16k.shape[0] * POSE_FPS / AUDIO_SR
            duration_delta = float(audio_frames - frame_count)
            duration_abs = abs(duration_delta)
            duration_checked += 1
            duration_abs_sum += duration_abs
            duration_abs_max = max(duration_abs_max, duration_abs)
            threshold = args.max_av_duration_delta_frames
            if threshold is not None and duration_abs > threshold:
                detail = (
                    f"motion_frames={frame_count}, "
                    f"audio_frames={audio_frames:.3f}, "
                    f"delta_frames={duration_delta:+.3f}, "
                    f"threshold={threshold:.3f}"
                )
                if args.invalid_clip_policy == "error":
                    raise CacheBuildError(
                        f"{clip_name}: duration_mismatch: {detail}"
                    )
                skipped_by_reason["duration_mismatch"] += 1
                skipped.append(
                    {
                        "source_index": source_index,
                        "clip": clip_name,
                        "reason": "duration_mismatch",
                        "detail": detail,
                    }
                )
                print(
                    f"[skip] {clip_name}: duration_mismatch: {detail}",
                    file=sys.stderr,
                )
                continue

            target_samples = frame_count * AUDIO_SR // POSE_FPS
            if audio_16k.shape[0] >= target_samples:
                aligned_audio = audio_16k[:target_samples]
            else:
                aligned_audio = np.pad(
                    audio_16k,
                    (0, target_samples - audio_16k.shape[0]),
                    mode="constant",
                )
            aligned_audio = np.ascontiguousarray(
                aligned_audio, dtype=np.float32
            )
            raw, mfcc, mel = _extract_classic_audio_features(
                aligned_audio,
                frame_count,
                librosa=librosa,
                np=np,
            )
            hubert = np.ascontiguousarray(
                hubert_extractor(aligned_audio),
                dtype=np.float32,
            )
            if (
                hubert.ndim != 2
                or hubert.shape[0] < 1
                or hubert.shape[1] != HUBERT_DIM
                or not np.isfinite(hubert).all()
            ):
                raise CacheBuildError(
                    f"{clip_name}: invalid HuBERT shape/data {hubert.shape}"
                )

            speaker_id = int(helpers.SHOW_SPEAKER_ID[speaker])
            if speaker_id not in (20, 21, 22, 23):
                raise CacheBuildError(
                    f"{clip_name}: invalid native speaker ID {speaker_id}"
                )
            speaker_array = np.asarray(speaker_id, dtype=np.int64)
            payload = (
                talkshow_pose,
                np.ascontiguousarray(expression, dtype=np.float32),
                raw,
                mfcc,
                mel,
                speaker_array,
                str(wav_path.resolve()),
                np.ascontiguousarray(betas, dtype=np.float32),
            )
            _validate_main_payload(payload, frame_count, np=np)
            output_index = len(records)
            if output_index > 99999:
                raise CacheBuildError(
                    "DiffSHEG five-digit key space supports at most 100000 clips"
                )
            key = f"{output_index:05d}".encode("ascii")
            if not main_transaction.put(
                key,
                _serialize(
                    pyarrow,
                    payload,
                    label=f"main record {clip_name}",
                ),
                overwrite=False,
            ):
                raise CacheBuildError(f"duplicate main cache key {key!r}")
            if not hubert_transaction.put(
                key,
                _serialize(
                    pyarrow,
                    hubert,
                    label=f"HuBERT record {clip_name}",
                ),
                overwrite=False,
            ):
                raise CacheBuildError(f"duplicate HuBERT cache key {key!r}")
            records.append(
                {
                    "key": key.decode("ascii"),
                    "source_index": source_index,
                    "clip": clip_name,
                    "speaker": speaker,
                    "speaker_id": speaker_id,
                    "sequence": sequence,
                    "frame_count": frame_count,
                    "audio_samples_16k": int(aligned_audio.shape[0]),
                    "hubert_frames": int(hubert.shape[0]),
                    "duration_delta_frames": duration_delta,
                    "pkl_path": str(pkl_path.resolve()),
                    "wav_path": str(wav_path.resolve()),
                }
            )

            if len(records) % args.commit_interval == 0:
                main_transaction.commit()
                hubert_transaction.commit()
                main_transaction = main_environment.begin(write=True)
                hubert_transaction = hubert_environment.begin(write=True)

        if not records:
            raise CacheBuildError(
                "all TalkSHOW test clips were rejected; no cache was published"
            )
        main_transaction.commit()
        main_transaction = None
        hubert_transaction.commit()
        hubert_transaction = None
        main_environment.sync()
        hubert_environment.sync()
        main_environment.close()
        main_environment = None
        hubert_environment.close()
        hubert_environment = None

        _verify_cache_pair(
            main_staging,
            hubert_staging,
            len(records),
            lmdb=lmdb,
            np=np,
            pyarrow=pyarrow,
        )

        # Publish HuBERT first: the public runner cannot observe a usable main
        # cache until its required parallel feature cache is already complete.
        os.rename(hubert_staging, hubert_final)
        published.append(hubert_final)
        os.rename(main_staging, main_final)
        published.append(main_final)

        model_path = Path(args.hubert_model).expanduser()
        model_files: dict[str, str] = {}
        if model_path.is_dir():
            for filename in ("config.json", "preprocessor_config.json"):
                candidate = model_path / filename
                if candidate.is_file():
                    model_files[filename] = _sha256_file(candidate)
        summary = {
            "schema_version": 1,
            "protocol": "diffsheg_show_public_cache_reconstruction_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "diffsheg_reference_commit": DIFFSHEG_REFERENCE_COMMIT,
            "warning": (
                "DiffSHEG did not publish its SHOW cache builder. The cache "
                "schema and HuBERT extraction match public code; the 30-FPS "
                "mel timing is an explicit reconstruction."
            ),
            "source": {
                "split_view_root": str(split_root),
                "speakers_in_scan_order": list(args.speakers),
                "scan_stats": scan_stats,
                "split_provenance": split_provenance,
                "source_clip_count": len(items),
                "accepted_clip_count": len(records),
                "skipped_clip_count": len(skipped),
                "skipped_by_reason": dict(skipped_by_reason),
                "skipped": skipped,
            },
            "outputs": {
                "main_cache": str(main_final),
                "hubert_cache": str(hubert_final),
                "summary": str(summary_path),
                "main_entries": len(records),
                "hubert_entries": len(records),
            },
            "motion": {
                "fps": POSE_FPS,
                "pose_dim": 165,
                "expression_dim": 100,
                "pose_representation": "local_axis_angle",
                "pose_order": (
                    "jaw,leye,reye,global_orient,body,left_hand,right_hand"
                ),
                "test_record_policy": "one_complete_clip_per_record",
                "speaker_ids": {
                    speaker: int(helpers.SHOW_SPEAKER_ID[speaker])
                    for speaker in args.speakers
                },
                "hand_component_path": str(hand_path),
                "hand_component_sha256": _sha256_file(hand_path),
            },
            "audio": {
                "aligned_sample_rate": AUDIO_SR,
                "mel_sample_rate": MEL_SR,
                "mel_hop_length": MEL_HOP_LENGTH,
                "mel_dim": MEL_DIM,
                "mfcc_dim": MFCC_DIM,
                "hubert_dim": HUBERT_DIM,
                "hubert_model_requested": args.hubert_model,
                "hubert_model_resolved": hubert_extractor.resolved_name,
                "hubert_model_metadata_sha256": model_files,
                "max_av_duration_delta_frames": (
                    args.max_av_duration_delta_frames
                ),
                "invalid_clip_policy": args.invalid_clip_policy,
                "duration_checked_clips": duration_checked,
                "duration_delta_frames_abs_mean": (
                    duration_abs_sum / duration_checked
                    if duration_checked
                    else None
                ),
                "duration_delta_frames_abs_max": duration_abs_max,
            },
            "diffsheg_statistics": {
                "path": str(stats_path),
                "sha256": _sha256_file(stats_path),
            },
            "cache_contract": {
                "backend": "directory-backed LMDB",
                "key_format": "five-digit contiguous ASCII",
                "serialization": "pyarrow.serialize / pyarrow.deserialize",
                "main_tuple_fields": [
                    "pose",
                    "expression",
                    "aud_raw",
                    "mfcc",
                    "mel",
                    "speaker",
                    "aud_file",
                    "betas",
                ],
                "parallel_hubert_key_policy": "exactly identical to main",
            },
            "records": records,
        }
        _write_json_atomic(summary_path, summary)
        return summary
    except BaseException:
        if main_transaction is not None:
            main_transaction.abort()
        if hubert_transaction is not None:
            hubert_transaction.abort()
        if main_environment is not None:
            main_environment.close()
        if hubert_environment is not None:
            hubert_environment.close()
        for staging in (main_staging, hubert_staging):
            if staging.exists():
                shutil.rmtree(staging)
        # Roll back only directories published by this invocation. They were
        # proven absent before the build and carry this invocation's data.
        for path in reversed(published):
            if path.exists():
                shutil.rmtree(path)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build DiffSHEG's missing SHOW test and HuBERT LMDB caches "
            "from a non-destructive TalkSHOW split view."
        )
    )
    parser.add_argument(
        "--split-view-root",
        type=Path,
        required=True,
        help=(
            "TalkSHOW split view containing "
            "<speaker>/<video>/test/<clip>/{clip.pkl,clip.wav}"
        ),
    )
    parser.add_argument(
        "--diffsheg-root",
        type=Path,
        required=True,
        help="DiffSHEG repository root; outputs go under data/SHOW",
    )
    parser.add_argument(
        "--hubert-model",
        default=DEFAULT_HUBERT_MODEL,
        help=(
            "local snapshot or Hugging Face ID for "
            "facebook/hubert-large-ls960-ft"
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="forbid Hugging Face network access",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="HuBERT extraction device, for example cuda:0 or cpu",
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=list(DEFAULT_SPEAKERS),
        help=(
            "canonical speakers in cache scan order; default follows the "
            "TalkSHOW public command"
        ),
    )
    parser.add_argument(
        "--hand-component-path",
        type=Path,
        default=DEFAULT_HAND_COMPONENT_PATH,
        help="TalkSHOW 12-component hand PCA basis JSON",
    )
    parser.add_argument(
        "--require-split-summary",
        action="store_true",
        help=(
            "require split_view_summary.json and the pinned official split "
            "manifest provenance"
        ),
    )
    parser.add_argument(
        "--max-av-duration-delta-frames",
        type=float,
        help=(
            "optional maximum absolute audio/motion duration mismatch in "
            "30-FPS frames"
        ),
    )
    parser.add_argument(
        "--invalid-clip-policy",
        choices=("error", "skip"),
        default="error",
        help=(
            "fail on malformed/mismatched clips, or skip and record them; "
            "skip requires a duration threshold"
        ),
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=1024,
        help="sparse LMDB map size for each cache (default: 1024 GiB)",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=50,
        help="commit both paired LMDB transactions every N accepted clips",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.commit_interval < 1:
        parser.error("--commit-interval must be positive")
    if args.map_size_gb < 1:
        parser.error("--map-size-gb must be positive")
    if (
        args.max_av_duration_delta_frames is not None
        and args.max_av_duration_delta_frames < 0
    ):
        parser.error("--max-av-duration-delta-frames must be non-negative")
    try:
        summary = build_caches(args)
    except CacheBuildError as exc:
        parser.exit(2, f"[abort] {exc}\n")
    print(
        "[done] DiffSHEG SHOW caches created: "
        f"{summary['outputs']['main_entries']} clips"
    )
    print(f"  main:   {summary['outputs']['main_cache']}")
    print(f"  hubert: {summary['outputs']['hubert_cache']}")
    print(f"  summary: {summary['outputs']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
