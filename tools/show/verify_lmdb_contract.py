#!/usr/bin/env python3
"""Fail-closed validation for SHOW LMDBs used by the TalkSHOW face adapter.

The default mode scans every record.  A strict LMDB must contain a JSON
``__meta__`` record that identifies its split and audio-feature provenance.
Older LMDBs can only be inspected with ``--allow-legacy``; that mode emits a
warning and still enforces every record-level invariant that can be proven.

Examples:

    python tools/show/verify_lmdb_contract.py \
        --lmdb-path /data/show_train.lmdb \
        --expected-split train \
        --expected-max-av-duration-delta-frames 3 \
        --rest-joint-path /data/rest_joint_position_show.mat

    python tools/show/verify_lmdb_contract.py \
        --lmdb-path /data/show_test_legacy.lmdb \
        --expected-split test \
        --rest-joint-path /data/rest_joint_position_show.mat \
        --allow-legacy

Optionally pass ``--face-checkpoint`` and/or ``--wavlm-dir`` to compare the
LMDB audio hashes with the face checkpoint's ``resume_contract`` and the local
WavLM files used for runtime validation/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_EXPECTED_PIDS = (21, 22, 23, 24)
DEFAULT_EXPECTED_PID_OFFSET = 1
OFFICIAL_SPLIT_SHA256 = (
    "2df6e745cdf7473f13ce3ae2ed759c3cceb60c9197e7f3fd65110e7bc20b6f2d"
)
OFFICIAL_HAND_COMPONENT_SHA256 = (
    "a9db3d2f04fe161e09553b17aaf5922791b0ed826593e640f4ddf8ebb8baf423"
)
SHOW_SPEAKER_ID = {
    "oliver": 20,
    "chemistry": 21,
    "seth": 22,
    "conan": 23,
}
EXPECTED_POSE_ORDER = (
    "global_orient",
    "body_pose_axis",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "left_hand_pose_axis_angle",
    "right_hand_pose_axis_angle",
)
BASE_RECORD_KEYS = {
    "betas",
    "poses",
    "expressions",
    "trans",
    "audio",
    "huberts",
    "pid",
    "ori_vid",
}
FACE_CONTEXT_RECORD_KEYS = {
    "face_huberts",
    "face_hubert_valid_length",
    "face_hubert_crop_start",
}
EXPECTED_RECORD_KEYS = BASE_RECORD_KEYS | FACE_CONTEXT_RECORD_KEYS
FACE_CONTEXT_FRAMES = 6
FACE_CONTEXT_PROTOCOL = "source_clip_valid_slice_v1"
REQUIRED_META_FIELDS = {
    "schema_version",
    "dataset",
    "split",
    "speakers",
    "speaker_ids",
    "pid_offset",
    "seq_size",
    "stride_size",
    "local_rotation",
    "pose_order",
    "rotation_space",
    "feature_scope",
    "face_context_frames",
    "face_context_protocol",
    "hand_component_sha256",
    "model_type",
    "audio_model_id",
    "audio_config_sha256",
    "audio_preprocessor_sha256",
    "audio_model_files_sha256",
    "audio_do_normalize",
    "audio_feat_extract_norm",
    "audio_add_adapter",
    "audio_feature_dim",
    "audio_sr",
    "pose_fps",
    "split_manifest_sha256",
    "split_view_summary_sha256",
    "split_view_counts",
    "split_view_missing_count",
    "split_view_limit_per_speaker_split",
    "source_clip_count",
    "output_sample_count",
    "stats",
}
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
WAVLM_METADATA_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "added_tokens.json",
)
WAVLM_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


class ContractError(RuntimeError):
    """Raised when a SHOW artifact violates the declared training contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ContractError(f"{name} must be an integer, got {value!r}")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ContractError(
            f"{name} must be >= {minimum}, got {normalized}"
        )
    return normalized


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ContractError(
            f"{name} must be a 64-character SHA-256 hex digest"
        )
    return value.lower()


def _legacy_warning(message: str) -> None:
    warnings.warn(
        "LEGACY SHOW CONTRACT: " + message,
        RuntimeWarning,
        stacklevel=2,
    )


def canonical_json_sha256(value: object) -> str:
    """Hash JSON exactly like reproductions/talkshow/configure_face.py."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def decode_metadata(raw: bytes) -> dict[str, Any]:
    """Decode the canonical UTF-8 JSON metadata record.

    Pickled metadata is deliberately not accepted in strict mode: JSON keeps
    this provenance record inspectable without executing pickle payloads.
    """

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError(
            "LMDB __meta__ must be a UTF-8 JSON object"
        ) from error
    if not isinstance(value, dict):
        raise ContractError("LMDB __meta__ must decode to a JSON object")
    return value


def _normalize_speaker_ids(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ContractError("metadata speaker_ids must be a non-empty mapping")
    normalized: dict[str, int] = {}
    for speaker, pid in value.items():
        if not isinstance(speaker, str) or not speaker:
            raise ContractError(
                "metadata speaker_ids keys must be non-empty strings"
            )
        normalized[speaker] = _require_int(
            pid,
            f"metadata speaker_ids[{speaker!r}]",
        )
    return normalized


def _normalize_model_file_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ContractError(
            "metadata audio_model_files_sha256 must be a non-empty mapping"
        )
    normalized: dict[str, str] = {}
    for relative_name, digest in value.items():
        if not isinstance(relative_name, str) or not relative_name:
            raise ContractError(
                "audio_model_files_sha256 keys must be relative filenames"
            )
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContractError(
                "audio_model_files_sha256 contains an unsafe filename: "
                f"{relative_name!r}"
            )
        normalized[relative_path.as_posix()] = _require_sha256(
            digest,
            f"metadata audio_model_files_sha256[{relative_name!r}]",
        )
    return normalized


def validate_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_split: str | None,
    expected_pids: Sequence[int],
    expected_seq_size: int,
    expected_feature_dim: int,
    expected_pid_offset: int,
    audio_sr: int,
    pose_fps: int,
    lmdb_length: int,
    expected_max_av_duration_delta_frames: float | None = None,
) -> dict[str, Any]:
    """Validate and normalize the canonical SHOW ``__meta__`` object."""

    missing = sorted(REQUIRED_META_FIELDS.difference(metadata))
    if missing:
        raise ContractError(
            "LMDB __meta__ is incomplete; missing fields: "
            + ", ".join(missing)
        )

    schema_version = _require_int(
        metadata["schema_version"],
        "metadata schema_version",
        minimum=1,
    )
    _require(
        schema_version == 2,
        f"unsupported metadata schema_version={schema_version}",
    )
    _require(
        metadata["dataset"] == "SHOW",
        f"metadata dataset must be 'SHOW', got {metadata['dataset']!r}",
    )

    split = metadata["split"]
    _require(
        split in {"train", "val", "test"},
        f"metadata split must be train/val/test, got {split!r}",
    )
    if expected_split is not None:
        _require(
            split == expected_split,
            f"metadata split={split!r}, expected {expected_split!r}",
        )
    split_manifest_sha256 = _require_sha256(
        metadata["split_manifest_sha256"],
        "metadata split_manifest_sha256",
    )
    _require(
        split_manifest_sha256 == OFFICIAL_SPLIT_SHA256,
        "metadata split_manifest_sha256 does not match the pinned official "
        f"manifest: {split_manifest_sha256} != {OFFICIAL_SPLIT_SHA256}",
    )
    split_view_summary_sha256 = _require_sha256(
        metadata["split_view_summary_sha256"],
        "metadata split_view_summary_sha256",
    )
    raw_split_view_counts = metadata["split_view_counts"]
    _require(
        isinstance(raw_split_view_counts, Mapping)
        and set(raw_split_view_counts) == {"train", "val", "test"},
        "metadata split_view_counts must contain exactly train/val/test",
    )
    split_view_counts = {
        split_name: _require_int(
            raw_split_view_counts[split_name],
            f"metadata split_view_counts[{split_name!r}]",
            minimum=1,
        )
        for split_name in ("train", "val", "test")
    }
    split_view_missing_count = _require_int(
        metadata["split_view_missing_count"],
        "metadata split_view_missing_count",
        minimum=0,
    )
    raw_split_view_limit = metadata[
        "split_view_limit_per_speaker_split"
    ]
    split_view_limit = (
        None
        if raw_split_view_limit is None
        else _require_int(
            raw_split_view_limit,
            "metadata split_view_limit_per_speaker_split",
            minimum=1,
        )
    )

    speakers = metadata["speakers"]
    _require(
        isinstance(speakers, list)
        and bool(speakers)
        and all(isinstance(name, str) and name for name in speakers),
        "metadata speakers must be a non-empty list of names",
    )
    _require(
        len(set(speakers)) == len(speakers),
        "metadata speakers contains duplicates",
    )
    speaker_ids = _normalize_speaker_ids(metadata["speaker_ids"])
    _require(
        set(speaker_ids) == set(speakers),
        "metadata speakers and speaker_ids keys differ",
    )
    expected_pid_set = set(int(pid) for pid in expected_pids)
    _require(
        set(speaker_ids.values()) == expected_pid_set,
        "metadata speaker IDs "
        f"{sorted(set(speaker_ids.values()))} != expected "
        f"{sorted(expected_pid_set)}",
    )

    pid_offset = _require_int(metadata["pid_offset"], "metadata pid_offset")
    _require(
        pid_offset == expected_pid_offset,
        f"metadata pid_offset={pid_offset}, expected {expected_pid_offset}",
    )
    canonical_speaker_ids = {
        speaker: source_pid + pid_offset
        for speaker, source_pid in SHOW_SPEAKER_ID.items()
    }
    _require(
        speaker_ids == canonical_speaker_ids,
        "metadata speaker_ids does not match TalkSHOW's canonical identity "
        f"mapping {canonical_speaker_ids}; got {speaker_ids}",
    )
    _require(
        expected_pid_set == set(canonical_speaker_ids.values()),
        "--expected-pids disagrees with the canonical TalkSHOW IDs plus "
        f"pid_offset={pid_offset}",
    )
    seq_size = _require_int(
        metadata["seq_size"], "metadata seq_size", minimum=1
    )
    _require_int(
        metadata["stride_size"], "metadata stride_size", minimum=1
    )
    _require(
        seq_size == expected_seq_size,
        f"metadata seq_size={seq_size}, expected {expected_seq_size}",
    )

    local_rotation = metadata["local_rotation"]
    _require(
        isinstance(local_rotation, bool),
        "metadata local_rotation must be a boolean",
    )
    pose_order = metadata["pose_order"]
    _require(
        isinstance(pose_order, list)
        and tuple(pose_order) == EXPECTED_POSE_ORDER,
        "metadata pose_order must exactly match the canonical 55-joint "
        f"layout {list(EXPECTED_POSE_ORDER)}",
    )
    rotation_space = metadata["rotation_space"]
    _require(
        rotation_space in {"local", "global"},
        "metadata rotation_space must be 'local' or 'global'",
    )
    _require(
        (rotation_space == "local") == local_rotation,
        "metadata rotation_space disagrees with local_rotation",
    )
    _require(
        metadata["feature_scope"] == "full_clip",
        "metadata feature_scope must be 'full_clip' for the TalkSHOW "
        "face-condition contract",
    )
    face_context_frames = _require_int(
        metadata["face_context_frames"],
        "metadata face_context_frames",
        minimum=0,
    )
    _require(
        face_context_frames == FACE_CONTEXT_FRAMES,
        "metadata face_context_frames="
        f"{face_context_frames}, expected {FACE_CONTEXT_FRAMES}",
    )
    _require(
        metadata["face_context_protocol"] == FACE_CONTEXT_PROTOCOL,
        "metadata face_context_protocol must be "
        f"{FACE_CONTEXT_PROTOCOL!r}",
    )
    hand_component_sha256 = _require_sha256(
        metadata["hand_component_sha256"],
        "metadata hand_component_sha256",
    )
    _require(
        hand_component_sha256 == OFFICIAL_HAND_COMPONENT_SHA256,
        "metadata hand_component_sha256 does not match the canonical "
        f"TalkSHOW hand basis: {hand_component_sha256} != "
        f"{OFFICIAL_HAND_COMPONENT_SHA256}",
    )

    _require(
        metadata["model_type"] == "wavlm",
        f"metadata model_type must be 'wavlm', got {metadata['model_type']!r}",
    )
    _require(
        isinstance(metadata["audio_model_id"], str)
        and bool(metadata["audio_model_id"]),
        "metadata audio_model_id must be a non-empty string",
    )
    audio_config_sha256 = _require_sha256(
        metadata["audio_config_sha256"],
        "metadata audio_config_sha256",
    )
    audio_preprocessor_sha256 = _require_sha256(
        metadata["audio_preprocessor_sha256"],
        "metadata audio_preprocessor_sha256",
    )
    audio_model_files_sha256 = _normalize_model_file_hashes(
        metadata["audio_model_files_sha256"]
    )
    _require(
        {"config.json", "preprocessor_config.json"}.issubset(
            audio_model_files_sha256
        ),
        "metadata audio_model_files_sha256 must contain config.json and "
        "preprocessor_config.json",
    )
    _require(
        any(name in audio_model_files_sha256 for name in WAVLM_WEIGHT_FILES),
        "metadata audio_model_files_sha256 contains no recognized WavLM "
        "weight file or weight index",
    )
    _require(
        isinstance(metadata["audio_do_normalize"], bool),
        "metadata audio_do_normalize must be a boolean",
    )
    _require(
        metadata["audio_feat_extract_norm"] == "layer",
        "metadata audio_feat_extract_norm must be 'layer'",
    )
    _require(
        metadata["audio_add_adapter"] is False,
        "metadata audio_add_adapter must be false",
    )

    feature_dim = _require_int(
        metadata["audio_feature_dim"],
        "metadata audio_feature_dim",
        minimum=1,
    )
    _require(
        feature_dim == expected_feature_dim,
        "metadata audio_feature_dim="
        f"{feature_dim}, expected {expected_feature_dim}",
    )
    metadata_audio_sr = _require_int(
        metadata["audio_sr"], "metadata audio_sr", minimum=1
    )
    _require(
        metadata_audio_sr == audio_sr,
        f"metadata audio_sr={metadata_audio_sr}, expected {audio_sr}",
    )
    metadata_pose_fps = _require_int(
        metadata["pose_fps"], "metadata pose_fps", minimum=1
    )
    _require(
        metadata_pose_fps == pose_fps,
        f"metadata pose_fps={metadata_pose_fps}, expected {pose_fps}",
    )

    source_clip_count = _require_int(
        metadata["source_clip_count"],
        "metadata source_clip_count",
        minimum=1,
    )
    _require(
        split_view_counts[split] == source_clip_count,
        "metadata split_view_counts disagrees with source_clip_count for "
        f"split={split!r}: {split_view_counts[split]} != "
        f"{source_clip_count}",
    )
    output_sample_count = _require_int(
        metadata["output_sample_count"],
        "metadata output_sample_count",
        minimum=1,
    )
    _require(
        output_sample_count == lmdb_length,
        "metadata output_sample_count="
        f"{output_sample_count}, LMDB __len__={lmdb_length}",
    )
    stats = metadata["stats"]
    _require(
        isinstance(stats, Mapping),
        "metadata stats must be a mapping",
    )
    raw_duration_threshold = metadata.get(
        "max_av_duration_delta_frames"
    )
    duration_threshold = None
    if raw_duration_threshold is not None:
        _require(
            not isinstance(raw_duration_threshold, bool)
            and isinstance(raw_duration_threshold, (int, float))
            and np.isfinite(raw_duration_threshold)
            and raw_duration_threshold >= 0,
            "metadata max_av_duration_delta_frames must be a finite "
            "non-negative number or null",
        )
        duration_threshold = float(raw_duration_threshold)

    duration_policy = metadata.get("duration_mismatch_policy")
    _require(
        duration_policy in {None, "error", "skip"},
        "metadata duration_mismatch_policy must be 'error', 'skip', "
        "or absent for a legacy schema-2 LMDB",
    )
    if duration_policy == "skip":
        _require(
            duration_threshold is not None,
            "metadata duration_mismatch_policy='skip' requires "
            "max_av_duration_delta_frames",
        )
        processing_stats = stats.get("processing")
        _require(
            isinstance(processing_stats, Mapping),
            "metadata skip policy requires stats.processing",
        )
        skipped_count = _require_int(
            processing_stats.get("duration_mismatch_skipped_count"),
            "metadata duration_mismatch_skipped_count",
            minimum=0,
        )
        skipped_clips = processing_stats.get(
            "duration_mismatch_skipped_clips"
        )
        _require(
            isinstance(skipped_clips, list)
            and len(skipped_clips) == skipped_count,
            "metadata duration_mismatch_skipped_clips length must equal "
            "duration_mismatch_skipped_count",
        )

    if expected_max_av_duration_delta_frames is not None:
        expected_threshold = expected_max_av_duration_delta_frames
        _require(
            not isinstance(expected_threshold, bool)
            and isinstance(expected_threshold, (int, float))
            and np.isfinite(expected_threshold)
            and expected_threshold >= 0,
            "expected_max_av_duration_delta_frames must be a finite "
            "non-negative number",
        )
        _require(
            duration_threshold is not None
            and np.isclose(
                duration_threshold,
                float(expected_threshold),
                rtol=0.0,
                atol=1e-9,
            ),
            "metadata max_av_duration_delta_frames="
            f"{duration_threshold!r}, expected "
            f"{float(expected_threshold)!r}",
        )

    return {
        **dict(metadata),
        "split": split,
        "speaker_ids": speaker_ids,
        "audio_config_sha256": audio_config_sha256,
        "audio_preprocessor_sha256": audio_preprocessor_sha256,
        "audio_model_files_sha256": audio_model_files_sha256,
        "hand_component_sha256": hand_component_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "split_view_summary_sha256": split_view_summary_sha256,
        "split_view_counts": split_view_counts,
        "split_view_missing_count": split_view_missing_count,
        "split_view_limit_per_speaker_split": split_view_limit,
        "max_av_duration_delta_frames": duration_threshold,
        "duration_mismatch_policy": duration_policy,
    }


def _require_float32_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ContractError(f"{name} must be a numpy.ndarray")
    if value.shape != shape:
        raise ContractError(f"{name} shape={value.shape}, expected {shape}")
    if value.dtype != np.float32:
        raise ContractError(f"{name} dtype={value.dtype}, expected float32")
    if not np.isfinite(value).all():
        raise ContractError(f"{name} contains NaN or Inf")
    return value


def validate_record(
    record: Any,
    *,
    index: int,
    split: str,
    expected_pids: Sequence[int],
    expected_seq_size: int,
    expected_feature_dim: int,
    audio_sr: int,
    pose_fps: int,
    speaker_ids: Mapping[str, int] | None,
    require_face_context: bool = False,
) -> tuple[int, str, int]:
    """Validate one unpickled SHOW LMDB record."""

    prefix = f"record {index}"
    if not isinstance(record, Mapping):
        raise ContractError(f"{prefix} must be a mapping")
    expected_record_keys = (
        EXPECTED_RECORD_KEYS if require_face_context else BASE_RECORD_KEYS
    )
    keys = set(record)
    if keys != expected_record_keys:
        missing = sorted(expected_record_keys - keys)
        extra = sorted(keys - expected_record_keys)
        raise ContractError(
            f"{prefix} field set differs; missing={missing}, extra={extra}"
        )

    _require_float32_array(
        record["betas"], name=f"{prefix}.betas", shape=(300,)
    )
    poses = record["poses"]
    if not isinstance(poses, np.ndarray) or poses.ndim != 2:
        raise ContractError(f"{prefix}.poses must have shape [T,165]")
    frame_count = poses.shape[0]
    _require(frame_count > 0, f"{prefix} has no motion frames")
    _require_float32_array(
        poses,
        name=f"{prefix}.poses",
        shape=(frame_count, 165),
    )
    _require_float32_array(
        record["expressions"],
        name=f"{prefix}.expressions",
        shape=(frame_count, 100),
    )
    trans = _require_float32_array(
        record["trans"],
        name=f"{prefix}.trans",
        shape=(frame_count, 3),
    )
    _require(
        not np.any(trans),
        f"{prefix}.trans must be all-zero for SHOW",
    )

    audio_length = frame_count * audio_sr // pose_fps
    _require_float32_array(
        record["audio"],
        name=f"{prefix}.audio",
        shape=(audio_length,),
    )
    huberts = _require_float32_array(
        record["huberts"],
        name=f"{prefix}.huberts",
        shape=(1, frame_count, expected_feature_dim),
    )
    if require_face_context:
        expected_context_length = (
            frame_count
            if split == "test"
            else expected_seq_size + 2 * FACE_CONTEXT_FRAMES
        )
        face_huberts = _require_float32_array(
            record["face_huberts"],
            name=f"{prefix}.face_huberts",
            shape=(1, expected_context_length, expected_feature_dim),
        )
        valid_length = _require_int(
            record["face_hubert_valid_length"],
            f"{prefix}.face_hubert_valid_length",
            minimum=1,
        )
        crop_start = _require_int(
            record["face_hubert_crop_start"],
            f"{prefix}.face_hubert_crop_start",
            minimum=0,
        )
        _require(
            valid_length <= expected_context_length,
            f"{prefix}.face_hubert_valid_length exceeds carrier length",
        )
        _require(
            crop_start <= FACE_CONTEXT_FRAMES,
            f"{prefix}.face_hubert_crop_start exceeds "
            f"{FACE_CONTEXT_FRAMES}",
        )
        _require(
            crop_start + frame_count <= valid_length,
            f"{prefix} face context does not contain the central window",
        )
        _require(
            valid_length - crop_start - frame_count
            <= FACE_CONTEXT_FRAMES,
            f"{prefix} right face context exceeds {FACE_CONTEXT_FRAMES}",
        )
        _require(
            np.array_equal(
                face_huberts[
                    :, crop_start:crop_start + frame_count, :
                ],
                huberts,
            ),
            f"{prefix} face context central crop differs from huberts",
        )
        _require(
            not np.any(face_huberts[:, valid_length:, :]),
            f"{prefix} face context carrier tail must be zero",
        )

    pid = _require_int(record["pid"], f"{prefix}.pid")
    expected_pid_set = set(int(value) for value in expected_pids)
    _require(
        pid in expected_pid_set,
        f"{prefix}.pid={pid} is outside expected IDs "
        f"{sorted(expected_pid_set)}",
    )
    ori_vid = record["ori_vid"]
    ori_parts = ori_vid.split("__", 1) if isinstance(ori_vid, str) else ()
    _require(
        len(ori_parts) == 2 and bool(ori_parts[0]) and bool(ori_parts[1]),
        f"{prefix}.ori_vid must be '<speaker>__<clip>'",
    )
    speaker = ori_parts[0]
    if speaker_ids is not None:
        _require(
            speaker in speaker_ids,
            f"{prefix}.ori_vid speaker {speaker!r} is absent from metadata",
        )
        _require(
            speaker_ids[speaker] == pid,
            f"{prefix} speaker/PID mismatch: {speaker!r} maps to "
            f"{speaker_ids[speaker]}, record has {pid}",
        )

    if split in {"train", "val"}:
        _require(
            frame_count == expected_seq_size,
            f"{prefix} T={frame_count}, expected fixed T={expected_seq_size} "
            f"for split={split}",
        )
    else:
        _require(frame_count > 0, f"{prefix} test sequence is empty")
    return pid, ori_vid, frame_count


def validate_rest_joint_mapping(
    mat_dict: Mapping[str, Any],
    *,
    expected_pids: Sequence[int],
) -> dict[int, np.ndarray]:
    """Validate a loaded ``rest_joint_position_show.mat`` dictionary."""

    expected_pid_set = set(int(pid) for pid in expected_pids)
    numeric_keys: dict[int, Any] = {
        int(key): value
        for key, value in mat_dict.items()
        if isinstance(key, str) and key.isdigit()
    }
    if set(numeric_keys) != expected_pid_set:
        raise ContractError(
            "rest-joint PID keys "
            f"{sorted(numeric_keys)} != expected {sorted(expected_pid_set)}"
        )
    normalized: dict[int, np.ndarray] = {}
    for pid in sorted(expected_pid_set):
        value = numeric_keys[pid]
        if not isinstance(value, np.ndarray):
            raise ContractError(f"rest joints for PID {pid} must be an ndarray")
        if value.shape != (55, 3):
            raise ContractError(
                f"rest joints for PID {pid} have shape {value.shape}, "
                "expected (55, 3)"
            )
        if value.dtype != np.float32:
            raise ContractError(
                f"rest joints for PID {pid} dtype={value.dtype}, "
                "expected float32"
            )
        if not np.isfinite(value).all():
            raise ContractError(
                f"rest joints for PID {pid} contain NaN or Inf"
            )
        normalized[pid] = value
    return normalized


def load_rest_joint_mapping(
    path: Path,
    *,
    expected_pids: Sequence[int],
) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise ContractError(f"rest-joint file does not exist: {path}")
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise ContractError(
            "scipy is required to validate the rest-joint .mat file"
        ) from error
    try:
        mat_dict = loadmat(path)
    except Exception as error:
        raise ContractError(f"could not load rest-joint file {path}") from error
    return validate_rest_joint_mapping(
        mat_dict,
        expected_pids=expected_pids,
    )


def load_wavlm_identity(
    wavlm_dir: Path,
    *,
    expected_files: Mapping[str, str] | None,
    expected_feature_dim: int,
) -> dict[str, Any]:
    """Hash and validate the runtime WavLM configuration directory."""

    root = wavlm_dir.expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"WavLM directory does not exist: {root}")
    parsed_json: dict[str, Any] = {}
    for filename in ("config.json", "preprocessor_config.json"):
        path = root / filename
        if not path.is_file():
            raise ContractError(f"WavLM directory is missing {filename}: {root}")
        try:
            parsed_json[filename] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(f"invalid WavLM JSON file: {path}") from error
        if not isinstance(parsed_json[filename], dict):
            raise ContractError(f"WavLM {filename} must contain a JSON object")

    config = parsed_json["config.json"]
    preprocessor = parsed_json["preprocessor_config.json"]
    _require(
        config.get("model_type") == "wavlm",
        "runtime WavLM config model_type must be 'wavlm'",
    )
    _require(
        config.get("hidden_size") == expected_feature_dim,
        "runtime WavLM hidden_size="
        f"{config.get('hidden_size')!r}, expected {expected_feature_dim}",
    )
    _require(
        config.get("feat_extract_norm") == "layer",
        "runtime WavLM feat_extract_norm must be 'layer'",
    )
    _require(
        config.get("add_adapter", False) is False,
        "runtime WavLM add_adapter must be false",
    )
    _require(
        isinstance(preprocessor.get("do_normalize"), bool),
        "runtime preprocessor do_normalize must be a boolean",
    )

    discovered_files = {
        name
        for name in WAVLM_METADATA_FILES + WAVLM_WEIGHT_FILES
        if (root / name).is_file()
    }
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(
                f"invalid WavLM weight index: {index_path}"
            ) from error
        weight_map = (
            index_payload.get("weight_map")
            if isinstance(index_payload, Mapping)
            else None
        )
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ContractError(
                f"WavLM weight index has no non-empty weight_map: {index_path}"
            )
        for shard_name in set(weight_map.values()):
            if not isinstance(shard_name, str) or not shard_name:
                raise ContractError(
                    f"WavLM weight index contains an invalid shard name: "
                    f"{shard_name!r}"
                )
            shard_path = Path(shard_name)
            if shard_path.is_absolute() or ".." in shard_path.parts:
                raise ContractError(
                    f"WavLM weight index contains an unsafe shard name: "
                    f"{shard_name!r}"
                )
            discovered_files.add(shard_path.as_posix())

    _require(
        any(name in discovered_files for name in WAVLM_WEIGHT_FILES),
        f"runtime WavLM directory contains no recognized weights: {root}",
    )
    if expected_files is not None:
        expected_names = set(expected_files)
        _require(
            discovered_files == expected_names,
            "runtime WavLM recognized file set differs from LMDB metadata; "
            f"missing_from_metadata={sorted(discovered_files - expected_names)}, "
            f"missing_at_runtime={sorted(expected_names - discovered_files)}",
        )

    file_hashes: dict[str, str] = {}
    for relative_name in sorted(discovered_files):
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContractError(
                f"unsafe WavLM metadata filename: {relative_name!r}"
            )
        # Do not resolve the candidate: Hugging Face snapshots commonly use
        # safe relative symlinks whose targets live in the cache's blob store.
        candidate = root / relative_path
        if not candidate.is_file():
            raise ContractError(
                f"runtime WavLM is missing declared/discovered file "
                f"{relative_name!r}"
            )
        try:
            file_hashes[relative_name] = file_sha256(candidate)
        except OSError as error:
            raise ContractError(
                f"could not hash runtime WavLM file {candidate}"
            ) from error

    return {
        "audio_config_sha256": canonical_json_sha256(config),
        "audio_preprocessor_sha256": canonical_json_sha256(preprocessor),
        "audio_model_files_sha256": file_hashes,
        "audio_do_normalize": preprocessor["do_normalize"],
        "audio_feat_extract_norm": config["feat_extract_norm"],
        "audio_add_adapter": bool(config.get("add_adapter", False)),
    }


def load_face_audio_contract(
    checkpoint: Path,
    *,
    allow_legacy: bool,
    expected_feature_dim: int,
    required_face_loss: str,
) -> Mapping[str, Any] | None:
    """Read WavLM provenance from a TalkSHOW face checkpoint."""

    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"face checkpoint does not exist: {path}")
    try:
        import torch
    except ImportError as error:
        raise ContractError(
            "torch is required to inspect a TalkSHOW face checkpoint"
        ) from error
    try:
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise ContractError(f"could not load face checkpoint {path}") from error
    if not isinstance(payload, Mapping):
        raise ContractError("face checkpoint payload must be a mapping")
    contract = payload.get("resume_contract")
    if contract is None:
        message = (
            f"face checkpoint {path} has no resume_contract; its WavLM "
            "snapshot cannot be compared"
        )
        if allow_legacy:
            _legacy_warning(message)
            return None
        raise ContractError(message + " (pass --allow-legacy to inspect anyway)")
    if not isinstance(contract, Mapping):
        raise ContractError("face checkpoint resume_contract must be a mapping")
    required_fields = {
        "schema_version",
        "trainer",
        "speakers",
        "audio_encoder",
        "audio_hidden_size",
        "audio_config_sha256",
        "audio_preprocessor_sha256",
        "audio_do_normalize",
        "freeze_audio_encoder",
        "num_classes",
        "face_loss",
    }
    missing = sorted(required_fields.difference(contract))
    if missing:
        raise ContractError(
            "face checkpoint resume_contract is incomplete; missing fields: "
            + ", ".join(missing)
        )
    schema_version = _require_int(
        contract["schema_version"],
        "face checkpoint schema_version",
        minimum=1,
    )
    _require(
        schema_version in {1, 2},
        "face checkpoint resume_contract schema_version must be 1 or 2",
    )
    _require(
        contract["trainer"] == "talkshow_face_large_batch",
        "face checkpoint resume_contract trainer must be "
        "'talkshow_face_large_batch'",
    )
    _require(
        contract.get("audio_encoder") == "wavlm",
        "face checkpoint resume_contract audio_encoder must be 'wavlm'",
    )
    _require(
        _require_int(
            contract["audio_hidden_size"],
            "face checkpoint audio_hidden_size",
            minimum=1,
        )
        == expected_feature_dim,
        "face checkpoint audio_hidden_size="
        f"{contract['audio_hidden_size']!r}, expected {expected_feature_dim}",
    )
    audio_config_sha256 = _require_sha256(
        contract.get("audio_config_sha256"),
        "face checkpoint audio_config_sha256",
    )
    audio_preprocessor_sha256 = _require_sha256(
        contract.get("audio_preprocessor_sha256"),
        "face checkpoint audio_preprocessor_sha256",
    )
    speakers = contract["speakers"]
    _require(
        isinstance(speakers, list)
        and len(speakers) == len(SHOW_SPEAKER_ID)
        and set(speakers) == set(SHOW_SPEAKER_ID),
        "face checkpoint speakers must contain exactly "
        f"{sorted(SHOW_SPEAKER_ID)}",
    )
    _require(
        isinstance(contract["audio_do_normalize"], bool),
        "face checkpoint audio_do_normalize must be a boolean",
    )
    _require(
        contract["freeze_audio_encoder"] is True,
        "face checkpoint freeze_audio_encoder must be true",
    )
    _require(
        _require_int(
            contract["num_classes"],
            "face checkpoint num_classes",
            minimum=1,
        )
        == len(SHOW_SPEAKER_ID),
        f"face checkpoint num_classes must be {len(SHOW_SPEAKER_ID)}",
    )
    _require(
        contract["face_loss"] == required_face_loss,
        f"face checkpoint face_loss={contract['face_loss']!r}, "
        f"required {required_face_loss!r}",
    )
    normalized = dict(contract)
    normalized["audio_config_sha256"] = audio_config_sha256
    normalized["audio_preprocessor_sha256"] = audio_preprocessor_sha256
    if schema_version == 2:
        _require(
            "audio_model_files_sha256" in contract,
            "face checkpoint schema 2 contract is missing "
            "audio_model_files_sha256",
        )
    if "audio_model_files_sha256" in contract:
        normalized["audio_model_files_sha256"] = (
            _normalize_model_file_hashes(
                contract["audio_model_files_sha256"]
            )
        )
    return normalized


def validate_audio_identity(
    metadata: Mapping[str, Any],
    *,
    face_contract: Mapping[str, Any] | None,
    wavlm_identity: Mapping[str, Any] | None,
) -> None:
    """Compare LMDB, face-checkpoint, and runtime WavLM provenance."""

    hash_fields = (
        "audio_config_sha256",
        "audio_preprocessor_sha256",
    )
    if face_contract is not None:
        for field in hash_fields:
            _require(
                face_contract.get(field) == metadata[field],
                f"face checkpoint {field} differs from LMDB metadata",
            )
        if "audio_model_files_sha256" in face_contract:
            _require(
                _normalize_model_file_hashes(
                    face_contract["audio_model_files_sha256"]
                )
                == metadata["audio_model_files_sha256"],
                "face checkpoint audio model file hashes differ from LMDB",
            )
        _require(
            face_contract.get("audio_hidden_size")
            == metadata["audio_feature_dim"],
            "face checkpoint audio_hidden_size differs from LMDB metadata",
        )
        _require(
            face_contract.get("audio_do_normalize")
            == metadata["audio_do_normalize"],
            "face checkpoint audio_do_normalize differs from LMDB metadata",
        )
        _require(
            set(face_contract.get("speakers", ()))
            == set(metadata["speaker_ids"]),
            "face checkpoint speakers differ from LMDB metadata",
        )

    if wavlm_identity is not None:
        for field in hash_fields:
            _require(
                wavlm_identity.get(field) == metadata[field],
                f"runtime WavLM {field} differs from LMDB metadata",
            )
        _require(
            wavlm_identity.get("audio_model_files_sha256")
            == metadata["audio_model_files_sha256"],
            "runtime WavLM file hashes differ from LMDB metadata",
        )
        for field in (
            "audio_do_normalize",
            "audio_feat_extract_norm",
            "audio_add_adapter",
        ):
            _require(
                wavlm_identity.get(field) == metadata[field],
                f"runtime WavLM {field} differs from LMDB metadata",
            )


def validate_face_runtime_identity(
    face_contract: Mapping[str, Any],
    wavlm_identity: Mapping[str, Any],
) -> None:
    """Compare the provable subset when a legacy LMDB has no metadata."""

    for field in (
        "audio_config_sha256",
        "audio_preprocessor_sha256",
        "audio_do_normalize",
    ):
        _require(
            face_contract.get(field) == wavlm_identity.get(field),
            f"runtime WavLM {field} differs from face checkpoint",
        )
    if "audio_model_files_sha256" in face_contract:
        _require(
            _normalize_model_file_hashes(
                face_contract["audio_model_files_sha256"]
            )
            == wavlm_identity.get("audio_model_files_sha256"),
            "runtime WavLM file hashes differ from face checkpoint",
        )


def _numeric_record_keys(transaction: Any) -> set[bytes]:
    try:
        cursor = transaction.cursor()
    except Exception as error:
        raise ContractError("could not open an LMDB cursor") from error
    try:
        # Cursor iteration defaults to values=True. SHOW values are megabyte-
        # scale records, so a key-only contract check must not fault them in.
        keys = cursor.iternext(keys=True, values=False)
    except (AttributeError, TypeError) as error:
        raise ContractError(
            "LMDB cursor does not support keys-only iteration"
        ) from error
    return {
        bytes(key)
        for key in keys
        if isinstance(key, (bytes, bytearray))
        and len(key) == 10
        and bytes(key).isdigit()
    }


def verify_transaction(
    transaction: Any,
    *,
    expected_split: str | None,
    expected_pids: Sequence[int],
    expected_seq_size: int,
    expected_feature_dim: int,
    expected_pid_offset: int,
    audio_sr: int,
    pose_fps: int,
    allow_legacy: bool,
    index: int | None,
    progress_every: int,
    expected_max_av_duration_delta_frames: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    stage_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Validate a transaction; split out for lightweight unit testing."""

    length_bytes = transaction.get(b"__len__")
    if length_bytes is None:
        raise ContractError("LMDB key '__len__' is missing")
    try:
        length = int(length_bytes.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ContractError("LMDB __len__ is not an ASCII integer") from error
    _require(length > 0, f"LMDB is empty (__len__={length})")

    if stage_callback is not None:
        stage_callback("validating metadata")
    metadata_raw = transaction.get(b"__meta__")
    metadata: dict[str, Any] | None
    if metadata_raw is None:
        message = "LMDB has no __meta__ provenance record"
        if not allow_legacy:
            raise ContractError(
                message + " (strict mode; use --allow-legacy explicitly)"
            )
        _legacy_warning(
            message
            + "; split and WavLM snapshot identity cannot be proven"
        )
        if expected_split is None:
            raise ContractError(
                "--expected-split is required when inspecting a legacy LMDB"
            )
        if expected_max_av_duration_delta_frames is not None:
            raise ContractError(
                "cannot verify max A/V duration delta for a legacy LMDB "
                "without __meta__"
            )
        metadata = None
        split = expected_split
        speaker_ids = None
    else:
        metadata = validate_metadata(
            decode_metadata(metadata_raw),
            expected_split=expected_split,
            expected_pids=expected_pids,
            expected_seq_size=expected_seq_size,
            expected_feature_dim=expected_feature_dim,
            expected_pid_offset=expected_pid_offset,
            audio_sr=audio_sr,
            pose_fps=pose_fps,
            lmdb_length=length,
            expected_max_av_duration_delta_frames=(
                expected_max_av_duration_delta_frames
            ),
        )
        split = metadata["split"]
        speaker_ids = metadata["speaker_ids"]
    if stage_callback is not None:
        metadata_mode = "legacy" if metadata is None else "strict-json"
        stage_callback(
            f"metadata accepted: split={split} records={length} "
            f"mode={metadata_mode}"
        )

    if stage_callback is not None:
        stage_callback(
            f"checking numeric key continuity for {length} records"
        )
    expected_keys = {
        f"{record_index:010d}".encode("ascii")
        for record_index in range(length)
    }
    numeric_keys = _numeric_record_keys(transaction)
    if numeric_keys != expected_keys:
        missing = sorted(expected_keys - numeric_keys)[:5]
        extra = sorted(numeric_keys - expected_keys)[:5]
        raise ContractError(
            "LMDB numeric keys are not exactly contiguous over "
            f"[0,{length}); missing={missing}, extra={extra}"
        )
    if stage_callback is not None:
        stage_callback(
            f"numeric key continuity passed: {length} records"
        )

    if index is None:
        indices: Sequence[int] = range(length)
        full_scan = True
    else:
        _require(
            0 <= index < length,
            f"requested index {index} is outside [0,{length})",
        )
        indices = (index,)
        full_scan = False
        warnings.warn(
            "PARTIAL SHOW CONTRACT: --idx validates one record only; "
            "dataset-wide PID coverage is not proven",
            RuntimeWarning,
            stacklevel=2,
        )
    if stage_callback is not None:
        stage_callback(f"scanning {len(indices)} record payloads")

    observed_pids: set[int] = set()
    observed_speakers: set[str] = set()
    min_frames: int | None = None
    max_frames: int | None = None
    checked = 0
    for record_index in indices:
        key = f"{record_index:010d}".encode("ascii")
        raw = transaction.get(key)
        if raw is None:
            raise ContractError(f"LMDB record key {key!r} is missing")
        try:
            record = pickle.loads(raw)
        except Exception as error:
            raise ContractError(
                f"could not unpickle LMDB record {record_index}"
            ) from error
        pid, ori_vid, frame_count = validate_record(
            record,
            index=record_index,
            split=split,
            expected_pids=expected_pids,
            expected_seq_size=expected_seq_size,
            expected_feature_dim=expected_feature_dim,
            audio_sr=audio_sr,
            pose_fps=pose_fps,
            speaker_ids=speaker_ids,
            require_face_context=metadata is not None,
        )
        observed_pids.add(pid)
        observed_speakers.add(ori_vid.split("__", 1)[0])
        min_frames = (
            frame_count if min_frames is None else min(min_frames, frame_count)
        )
        max_frames = (
            frame_count if max_frames is None else max(max_frames, frame_count)
        )
        checked += 1
        if (
            progress_callback is not None
            and progress_every > 0
            and (checked % progress_every == 0 or checked == len(indices))
        ):
            progress_callback(checked, len(indices))

    if full_scan:
        expected_pid_set = set(int(pid) for pid in expected_pids)
        _require(
            observed_pids == expected_pid_set,
            f"LMDB observed PID set {sorted(observed_pids)} != expected "
            f"{sorted(expected_pid_set)}",
        )
        if speaker_ids is not None:
            _require(
                observed_speakers == set(speaker_ids),
                "LMDB observed speakers "
                f"{sorted(observed_speakers)} != metadata speakers "
                f"{sorted(speaker_ids)}",
            )

    return {
        "length": length,
        "checked_records": checked,
        "full_scan": full_scan,
        "split": split,
        "observed_pids": sorted(observed_pids),
        "observed_speakers": sorted(observed_speakers),
        "min_frames": min_frames,
        "max_frames": max_frames,
        "metadata": metadata,
        "legacy": metadata is None,
    }


def verify_lmdb_contract(
    lmdb_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    path = lmdb_path.expanduser().resolve()
    if not path.exists():
        raise ContractError(f"LMDB path does not exist: {path}")
    try:
        import lmdb
    except ImportError as error:
        raise ContractError("python-lmdb is required for LMDB validation") from error
    try:
        environment = lmdb.open(
            str(path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=8,
        )
    except Exception as error:
        raise ContractError(f"could not open LMDB {path}") from error
    try:
        with environment.begin() as transaction:
            return verify_transaction(transaction, **kwargs)
    finally:
        environment.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmdb-path",
        "--lmdb_path",
        dest="lmdb_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-split",
        choices=("train", "val", "test"),
        help="Cross-check __meta__.split; required for --allow-legacy LMDBs",
    )
    parser.add_argument(
        "--expected-pids",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXPECTED_PIDS),
    )
    parser.add_argument(
        "--expected-pid-offset",
        type=int,
        default=DEFAULT_EXPECTED_PID_OFFSET,
    )
    parser.add_argument(
        "--expected-seq-size",
        "--expected_seq_size",
        dest="expected_seq_size",
        type=int,
        default=256,
    )
    parser.add_argument("--expected-feature-dim", type=int, default=1024)
    parser.add_argument(
        "--expected-max-av-duration-delta-frames",
        type=float,
        help=(
            "Require __meta__.max_av_duration_delta_frames to equal this "
            "30-FPS-frame threshold"
        ),
    )
    parser.add_argument(
        "--audio-sr",
        "--audio_sr",
        dest="audio_sr",
        type=int,
        default=16000,
    )
    parser.add_argument(
        "--pose-fps",
        "--pose_fps",
        dest="pose_fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Allow an LMDB or face checkpoint without provenance metadata",
    )
    parser.add_argument(
        "--idx",
        type=int,
        help="Explicit partial mode: validate one record instead of all records",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N records; 0 disables progress output",
    )
    rest_group = parser.add_mutually_exclusive_group()
    rest_group.add_argument("--rest-joint-path", type=Path)
    rest_group.add_argument(
        "--skip-rest-check",
        action="store_true",
        help="Explicitly skip the SHOW rest-joint contract",
    )
    parser.add_argument(
        "--face-checkpoint",
        type=Path,
        help="Optional TalkSHOW face ckpt for WavLM provenance comparison",
    )
    parser.add_argument(
        "--wavlm-dir",
        "--wavlm-model-path",
        dest="wavlm_dir",
        type=Path,
        help="Optional runtime WavLM directory for provenance comparison",
    )
    parser.add_argument(
        "--required-face-loss",
        choices=("corrected", "legacy"),
        default="corrected",
        help="Required face-loss semantics in a modern face checkpoint",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.expected_seq_size < 1:
        parser.error("--expected-seq-size must be positive")
    if args.expected_feature_dim < 1:
        parser.error("--expected-feature-dim must be positive")
    if args.audio_sr < 1 or args.pose_fps < 1:
        parser.error("--audio-sr and --pose-fps must be positive")
    if (
        args.expected_max_av_duration_delta_frames is not None
        and (
            not np.isfinite(args.expected_max_av_duration_delta_frames)
            or args.expected_max_av_duration_delta_frames < 0
        )
    ):
        parser.error(
            "--expected-max-av-duration-delta-frames must be finite and "
            "non-negative"
        )
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if len(set(args.expected_pids)) != len(args.expected_pids):
        parser.error("--expected-pids cannot contain duplicates")
    if not args.expected_pids:
        parser.error("--expected-pids cannot be empty")
    if args.rest_joint_path is None and not args.skip_rest_check:
        parser.error(
            "--rest-joint-path is required for the full contract; "
            "pass --skip-rest-check explicitly to omit it"
        )

    def show_progress(checked: int, total: int) -> None:
        print(
            f"[scan] checked {checked}/{total} LMDB records",
            file=sys.stderr,
            flush=True,
        )

    def show_stage(message: str) -> None:
        print(f"[preflight] {message}", file=sys.stderr, flush=True)

    try:
        summary = verify_lmdb_contract(
            args.lmdb_path,
            expected_split=args.expected_split,
            expected_pids=args.expected_pids,
            expected_seq_size=args.expected_seq_size,
            expected_feature_dim=args.expected_feature_dim,
            expected_pid_offset=args.expected_pid_offset,
            audio_sr=args.audio_sr,
            pose_fps=args.pose_fps,
            allow_legacy=args.allow_legacy,
            index=args.idx,
            progress_every=args.progress_every,
            expected_max_av_duration_delta_frames=(
                args.expected_max_av_duration_delta_frames
            ),
            progress_callback=show_progress,
            stage_callback=show_stage,
        )

        rest_checked = False
        if args.rest_joint_path is not None:
            show_stage("validating rest-joint table")
            load_rest_joint_mapping(
                args.rest_joint_path.expanduser().resolve(),
                expected_pids=args.expected_pids,
            )
            rest_checked = True
            show_stage("rest-joint table passed")

        metadata = summary["metadata"]
        audio_checks: list[str] = []
        face_contract = None
        if args.face_checkpoint is not None:
            show_stage("validating face-checkpoint audio identity")
            face_contract = load_face_audio_contract(
                args.face_checkpoint,
                allow_legacy=args.allow_legacy,
                expected_feature_dim=args.expected_feature_dim,
                required_face_loss=args.required_face_loss,
            )
            audio_checks.append("face_checkpoint")
            show_stage("face-checkpoint audio identity passed")

        wavlm_identity = None
        if args.wavlm_dir is not None:
            if metadata is None:
                _legacy_warning(
                    "LMDB has no audio hashes; runtime WavLM cannot be "
                    "compared to the LMDB (face/runtime consistency is still "
                    "checked when the face checkpoint has metadata)"
                )
            show_stage("hashing and validating runtime WavLM files")
            wavlm_identity = load_wavlm_identity(
                args.wavlm_dir,
                expected_files=(
                    metadata["audio_model_files_sha256"]
                    if metadata is not None
                    else None
                ),
                expected_feature_dim=args.expected_feature_dim,
            )
            audio_checks.append("wavlm_dir")
            show_stage("runtime WavLM files passed")

        if metadata is not None:
            if audio_checks:
                show_stage("comparing LMDB and runtime audio identities")
            validate_audio_identity(
                metadata,
                face_contract=face_contract,
                wavlm_identity=wavlm_identity,
            )
            if audio_checks:
                show_stage("LMDB and runtime audio identities passed")
        elif face_contract is not None:
            _legacy_warning(
                "face checkpoint has metadata, but the legacy LMDB does not; "
                "three-way WavLM consistency cannot be proven"
            )
            if wavlm_identity is not None:
                validate_face_runtime_identity(
                    face_contract,
                    wavlm_identity,
                )

    except ContractError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    mode = "full" if summary["full_scan"] else f"record {args.idx}"
    print(f"PASS: SHOW LMDB contract ({mode} scan)")
    print(f"lmdb: {args.lmdb_path.expanduser().resolve()}")
    print(
        f"split={summary['split']} records={summary['length']} "
        f"checked={summary['checked_records']} "
        f"T=[{summary['min_frames']},{summary['max_frames']}]"
    )
    print(
        f"pids={summary['observed_pids']} "
        f"speakers={summary['observed_speakers']}"
    )
    print(
        f"metadata={'legacy-warning' if summary['legacy'] else 'strict-json'} "
        f"rest_joints={'checked' if rest_checked else 'skipped'}"
    )
    if audio_checks:
        print("audio_identity_checked=" + ",".join(audio_checks))


if __name__ == "__main__":
    main()
