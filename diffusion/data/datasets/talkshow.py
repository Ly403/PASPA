"""Strict TalkSHOW LMDB adapter.

The underlying sample layout intentionally stays identical to
``BEATXDataset.LMDBDataset`` so the diffusion trainer does not need a second
data path.  This opt-in subclass only adds the metadata and identity checks
required by the four-person TalkSHOW face checkpoint.  Existing BEAT configs
continue to instantiate the original class unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import lmdb
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from diffusion.data.datasets.beat2.BEATXDataset import LMDBDataset


OFFICIAL_SPLIT_SHA256 = (
    "2df6e745cdf7473f13ce3ae2ed759c3cceb60c9197e7f3fd65110e7bc20b6f2d"
)
OFFICIAL_HAND_COMPONENT_SHA256 = (
    "a9db3d2f04fe161e09553b17aaf5922791b0ed826593e640f4ddf8ebb8baf423"
)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
SPLIT_NAMES = ("train", "val", "test")
FACE_CONTEXT_FRAMES = 6
FACE_CONTEXT_PROTOCOL = "source_clip_valid_slice_v1"
EXPECTED_POSE_ORDER = (
    "global_orient",
    "body_pose_axis",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "left_hand_pose_axis_angle",
    "right_hand_pose_axis_angle",
)
WAVLM_WEIGHT_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
TALKSHOW_SOURCE_SPEAKER_IDS = {
    "oliver": 20,
    "chemistry": 21,
    "seth": 22,
    "conan": 23,
}
TALKSHOW_RAW_PID_OFFSET = 1
TALKSHOW_RAW_SPEAKER_IDS = {
    speaker: source_pid + TALKSHOW_RAW_PID_OFFSET
    for speaker, source_pid in TALKSHOW_SOURCE_SPEAKER_IDS.items()
}
TALKSHOW_RAW_PIDS = tuple(sorted(TALKSHOW_RAW_SPEAKER_IDS.values()))
TALKSHOW_MODEL_PID_OFFSET = min(TALKSHOW_RAW_PIDS)
TALKSHOW_MODEL_PIDS = tuple(range(len(TALKSHOW_RAW_PIDS)))


class TalkSHOWLMDBDataset(LMDBDataset):
    """BEAT-compatible LMDB reader with a fail-closed TalkSHOW contract."""

    def __init__(
        self,
        lmdb_path: str,
        *,
        expected_split: str,
        expected_pids: Sequence[int] = TALKSHOW_RAW_PIDS,
        expected_pid_offset: int = TALKSHOW_RAW_PID_OFFSET,
        model_pid_offset: int = TALKSHOW_MODEL_PID_OFFSET,
        expected_feature_dim: int = 1024,
        expected_feature_scope: str = "full_clip",
        expected_seq_size: int = 256,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("use_symmetry_aug", False):
            raise ValueError(
                "TalkSHOW identity remapping requires symmetry augmentation "
                "to remain disabled; "
                "use_symmetry_aug must remain false"
            )
        if kwargs.get("rotation_method", "axisangle") != "axisangle":
            raise ValueError(
                "TalkSHOW main-model records must be read as axisangle"
            )
        self.expected_split = str(expected_split)
        if self.expected_split not in {"train", "val", "validation", "test"}:
            raise ValueError(
                f"unsupported TalkSHOW split: {self.expected_split!r}"
            )
        self.expected_pids = tuple(int(pid) for pid in expected_pids)
        if not self.expected_pids:
            raise ValueError("expected_pids cannot be empty")
        if len(set(self.expected_pids)) != len(self.expected_pids):
            raise ValueError("expected_pids cannot contain duplicates")
        if (
            isinstance(expected_pid_offset, bool)
            or not isinstance(expected_pid_offset, int)
        ):
            raise TypeError("expected_pid_offset must be an integer")
        if isinstance(model_pid_offset, bool) or not isinstance(
            model_pid_offset,
            int,
        ):
            raise TypeError("model_pid_offset must be an integer")
        self.expected_pid_offset = int(expected_pid_offset)
        self.model_pid_offset = int(model_pid_offset)
        self.expected_model_pids = tuple(
            pid - self.model_pid_offset for pid in self.expected_pids
        )
        if sorted(self.expected_model_pids) != list(
            range(len(self.expected_pids))
        ):
            raise ValueError(
                "TalkSHOW raw PIDs must map contiguously to model IDs "
                f"0..{len(self.expected_pids) - 1}; raw={self.expected_pids}, "
                f"model_pid_offset={self.model_pid_offset}"
            )
        self.expected_feature_dim = int(expected_feature_dim)
        self.expected_feature_scope = str(expected_feature_scope)
        self.expected_seq_size = int(expected_seq_size)

        super().__init__(lmdb_path, **kwargs)
        if self.num <= 0:
            raise ValueError(f"TalkSHOW LMDB is empty: {self.lmdb_path}")
        self.lmdb_metadata = self._load_and_validate_metadata()

        missing_rest = sorted(
            set(self.expected_pids).difference(self.rest_joint_position)
        )
        if missing_rest:
            raise ValueError(
                "TalkSHOW rest-joint table is missing PIDs: "
                f"{missing_rest}"
            )
        for pid in self.expected_pids:
            joints = self.rest_joint_position[pid]
            if joints.shape != (55, 3) or not torch.isfinite(joints).all():
                raise ValueError(
                    f"TalkSHOW rest joints for PID {pid} must be finite "
                    f"float32 [55,3], got shape={tuple(joints.shape)}, "
                    f"dtype={joints.dtype}"
                )

    def _load_and_validate_metadata(self) -> dict[str, Any]:
        env = lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=16,
        )
        try:
            with env.begin() as txn:
                raw = txn.get(b"__meta__")
        finally:
            env.close()
        if raw is None:
            raise ValueError(
                f"TalkSHOW LMDB has no strict __meta__ record: {self.lmdb_path}"
            )
        try:
            metadata = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"TalkSHOW LMDB __meta__ is not UTF-8 JSON: {self.lmdb_path}"
            ) from error
        if not isinstance(metadata, dict):
            raise ValueError("TalkSHOW LMDB __meta__ must be a JSON object")

        required = {
            "schema_version",
            "dataset",
            "split",
            "speakers",
            "speaker_ids",
            "pid_offset",
            "seq_size",
            "stride_size",
            "local_rotation",
            "rotation_space",
            "pose_representation",
            "pose_order",
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
            "feature_scope",
            "face_context_frames",
            "face_context_protocol",
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
        missing = sorted(required.difference(metadata))
        if missing:
            raise ValueError(
                f"TalkSHOW LMDB metadata is incomplete; missing: {missing}"
            )
        if metadata["schema_version"] != 2:
            raise ValueError(
                "unsupported TalkSHOW LMDB schema_version: "
                f"{metadata['schema_version']!r}; rebuild the strict "
                "TalkSHOW LMDB with the face-context-aware ETL"
            )
        if metadata["dataset"] != "SHOW":
            raise ValueError(
                f"expected SHOW dataset metadata, got {metadata['dataset']!r}"
            )
        if metadata["pid_offset"] != self.expected_pid_offset:
            raise ValueError(
                "TalkSHOW LMDB pid_offset mismatch: "
                f"{metadata['pid_offset']!r} != "
                f"{self.expected_pid_offset!r}"
            )
        declared_split = str(metadata["split"])
        allowed_split_names = {self.expected_split}
        if self.expected_split == "validation":
            allowed_split_names.add("val")
        elif self.expected_split == "val":
            allowed_split_names.add("validation")
        if declared_split not in allowed_split_names:
            raise ValueError(
                f"LMDB split is {declared_split!r}, expected "
                f"{self.expected_split!r}"
            )
        if metadata["model_type"] != "wavlm":
            raise ValueError("TalkSHOW LMDB audio model must be wavlm")
        if metadata["feature_scope"] != self.expected_feature_scope:
            raise ValueError(
                "TalkSHOW LMDB feature scope mismatch: "
                f"{metadata['feature_scope']!r} != "
                f"{self.expected_feature_scope!r}"
            )
        if metadata["face_context_frames"] != FACE_CONTEXT_FRAMES:
            raise ValueError(
                "TalkSHOW LMDB face context radius mismatch: "
                f"{metadata['face_context_frames']!r} != "
                f"{FACE_CONTEXT_FRAMES}"
            )
        if metadata["face_context_protocol"] != FACE_CONTEXT_PROTOCOL:
            raise ValueError(
                "TalkSHOW LMDB face context protocol mismatch: "
                f"{metadata['face_context_protocol']!r} != "
                f"{FACE_CONTEXT_PROTOCOL!r}"
            )
        if metadata["audio_feature_dim"] != self.expected_feature_dim:
            raise ValueError(
                "TalkSHOW LMDB feature dimension mismatch: "
                f"{metadata['audio_feature_dim']!r} != "
                f"{self.expected_feature_dim}"
            )
        if metadata["seq_size"] != self.expected_seq_size:
            raise ValueError(
                "TalkSHOW LMDB seq_size mismatch: "
                f"{metadata['seq_size']!r} != {self.expected_seq_size}"
            )
        if metadata["audio_sr"] != 16000 or metadata["pose_fps"] != 30:
            raise ValueError(
                "TalkSHOW integration requires 16 kHz audio and 30 FPS motion"
            )
        if metadata["local_rotation"] is not False:
            raise ValueError(
                "TalkSHOW main-model LMDB must store global rotations"
            )
        if metadata["rotation_space"] != "global":
            raise ValueError(
                "TalkSHOW main-model rotation_space must be 'global'"
            )
        if metadata["pose_representation"] != "axis_angle":
            raise ValueError(
                "TalkSHOW main-model pose representation must be axis_angle"
            )
        if (
            not isinstance(metadata["pose_order"], list)
            or tuple(metadata["pose_order"]) != EXPECTED_POSE_ORDER
        ):
            raise ValueError(
                "TalkSHOW pose_order does not match the canonical 55-joint "
                "layout"
            )
        stride_size = metadata["stride_size"]
        if (
            isinstance(stride_size, bool)
            or not isinstance(stride_size, int)
            or stride_size < 1
        ):
            raise ValueError(
                "TalkSHOW stride_size must be a positive integer"
            )
        hand_component_sha256 = metadata["hand_component_sha256"]
        if (
            not isinstance(hand_component_sha256, str)
            or SHA256_PATTERN.fullmatch(hand_component_sha256) is None
            or hand_component_sha256.lower()
            != OFFICIAL_HAND_COMPONENT_SHA256
        ):
            raise ValueError(
                "TalkSHOW LMDB hand component does not match the canonical "
                f"basis: {hand_component_sha256!r} != "
                f"{OFFICIAL_HAND_COMPONENT_SHA256}"
            )
        split_manifest_sha256 = metadata["split_manifest_sha256"]
        if (
            not isinstance(split_manifest_sha256, str)
            or SHA256_PATTERN.fullmatch(split_manifest_sha256) is None
            or split_manifest_sha256.lower() != OFFICIAL_SPLIT_SHA256
        ):
            raise ValueError(
                "TalkSHOW LMDB split manifest does not match the pinned "
                f"official manifest: {split_manifest_sha256!r} != "
                f"{OFFICIAL_SPLIT_SHA256}"
            )
        split_view_summary_sha256 = metadata[
            "split_view_summary_sha256"
        ]
        if (
            not isinstance(split_view_summary_sha256, str)
            or SHA256_PATTERN.fullmatch(split_view_summary_sha256) is None
        ):
            raise ValueError(
                "TalkSHOW split_view_summary_sha256 must be a "
                "64-character SHA-256 hex digest"
            )
        split_view_counts = metadata["split_view_counts"]
        if (
            not isinstance(split_view_counts, Mapping)
            or set(split_view_counts) != set(SPLIT_NAMES)
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                for count in split_view_counts.values()
            )
        ):
            raise ValueError(
                "TalkSHOW split_view_counts must contain exactly positive "
                f"integer train/val/test counts, got {split_view_counts!r}"
            )
        split_view_missing_count = metadata["split_view_missing_count"]
        if (
            isinstance(split_view_missing_count, bool)
            or not isinstance(split_view_missing_count, int)
            or split_view_missing_count < 0
        ):
            raise ValueError(
                "TalkSHOW split_view_missing_count must be a non-negative "
                f"integer, got {split_view_missing_count!r}"
            )
        split_view_limit = metadata[
            "split_view_limit_per_speaker_split"
        ]
        if (
            split_view_limit is not None
            and (
                isinstance(split_view_limit, bool)
                or not isinstance(split_view_limit, int)
                or split_view_limit < 1
            )
        ):
            raise ValueError(
                "TalkSHOW split_view_limit_per_speaker_split must be null "
                f"or a positive integer, got {split_view_limit!r}"
            )
        source_clip_count = metadata["source_clip_count"]
        if (
            isinstance(source_clip_count, bool)
            or not isinstance(source_clip_count, int)
            or source_clip_count < 1
        ):
            raise ValueError(
                "TalkSHOW source_clip_count must be a positive integer, "
                f"got {source_clip_count!r}"
            )
        split_count_key = (
            "val" if declared_split == "validation" else declared_split
        )
        if (
            split_count_key not in split_view_counts
            or split_view_counts[split_count_key] != source_clip_count
        ):
            raise ValueError(
                "TalkSHOW split view count disagrees with source_clip_count: "
                f"split={split_count_key!r}, "
                f"count={split_view_counts.get(split_count_key)!r}, "
                f"source_clip_count={source_clip_count}"
            )
        output_sample_count = metadata["output_sample_count"]
        if (
            isinstance(output_sample_count, bool)
            or not isinstance(output_sample_count, int)
            or output_sample_count != self.num
        ):
            raise ValueError(
                "TalkSHOW LMDB metadata sample count does not match __len__: "
                f"{output_sample_count!r} != {self.num}"
            )

        speaker_ids = metadata["speaker_ids"]
        if not isinstance(speaker_ids, Mapping):
            raise ValueError("TalkSHOW speaker_ids metadata must be an object")
        observed_pids = sorted(int(pid) for pid in speaker_ids.values())
        if observed_pids != sorted(self.expected_pids):
            raise ValueError(
                f"TalkSHOW PID mapping is {observed_pids}, expected "
                f"{sorted(self.expected_pids)}"
            )
        canonical_speaker_ids = {
            speaker: source_pid + self.expected_pid_offset
            for speaker, source_pid in TALKSHOW_SOURCE_SPEAKER_IDS.items()
        }
        if dict(speaker_ids) != canonical_speaker_ids:
            raise ValueError(
                "TalkSHOW speaker/PID metadata does not match the canonical "
                f"mapping {canonical_speaker_ids}"
            )
        speakers = metadata["speakers"]
        if not isinstance(speakers, list) or set(speakers) != set(speaker_ids):
            raise ValueError(
                "TalkSHOW speakers and speaker_ids metadata disagree"
            )
        digests = metadata["audio_model_files_sha256"]
        if not isinstance(digests, Mapping) or not digests:
            raise ValueError(
                "TalkSHOW audio_model_files_sha256 must be a non-empty object"
            )
        normalized_digest_names = set()
        for relative_name, digest in digests.items():
            if not isinstance(relative_name, str) or not relative_name:
                raise ValueError(
                    "TalkSHOW WavLM digest names must be non-empty strings"
                )
            relative_path = PurePosixPath(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    "TalkSHOW WavLM digest contains an unsafe path: "
                    f"{relative_name!r}"
                )
            if (
                not isinstance(digest, str)
                or SHA256_PATTERN.fullmatch(digest) is None
            ):
                raise ValueError(
                    "TalkSHOW WavLM digest must be a SHA-256 hex string: "
                    f"{relative_name!r}"
                )
            normalized_digest_names.add(relative_path.as_posix())
        if not {"config.json", "preprocessor_config.json"}.issubset(
            normalized_digest_names
        ):
            raise ValueError(
                "TalkSHOW WavLM digests must include config.json and "
                "preprocessor_config.json"
            )
        if not WAVLM_WEIGHT_FILES.intersection(normalized_digest_names):
            raise ValueError(
                "TalkSHOW WavLM digests contain no recognized model weights"
            )
        for field_name in (
            "audio_config_sha256",
            "audio_preprocessor_sha256",
        ):
            value = metadata[field_name]
            if (
                not isinstance(value, str)
                or SHA256_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(
                    f"TalkSHOW {field_name} must be a SHA-256 hex string"
                )
        if (
            not isinstance(metadata["audio_model_id"], str)
            or not metadata["audio_model_id"]
        ):
            raise ValueError("TalkSHOW audio_model_id must be non-empty")
        if not isinstance(metadata["audio_do_normalize"], bool):
            raise ValueError(
                "TalkSHOW audio_do_normalize must be a boolean"
            )
        if metadata["audio_feat_extract_norm"] != "layer":
            raise ValueError(
                "TalkSHOW WavLM feat_extract_norm must be 'layer'"
            )
        if metadata["audio_add_adapter"] is not False:
            raise ValueError(
                "TalkSHOW WavLM add_adapter must be false"
            )
        if not isinstance(metadata["stats"], Mapping):
            raise ValueError("TalkSHOW stats metadata must be a mapping")
        return metadata

    def _augment_sample_from_record(
        self,
        sample: dict[str, Any],
        data_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attach the schema-2 face context only on the TalkSHOW code path."""

        face_context_keys = {
            "face_huberts",
            "face_hubert_valid_length",
            "face_hubert_crop_start",
        }
        present_keys = face_context_keys.intersection(data_dict)
        if present_keys != face_context_keys:
            missing = sorted(face_context_keys - present_keys)
            raise ValueError(
                "TalkSHOW LMDB record has incomplete face audio context; "
                f"missing={missing}"
            )
        face_huberts = data_dict["face_huberts"]
        if not isinstance(face_huberts, np.ndarray):
            raise TypeError("TalkSHOW face_huberts must be a numpy array")
        sample.update(
            {
                "face_audio_feat": torch.from_numpy(face_huberts)[0],
                "face_audio_valid_length": int(
                    data_dict["face_hubert_valid_length"]
                ),
                "face_audio_crop_start": int(
                    data_dict["face_hubert_crop_start"]
                ),
            }
        )
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        is_padding_sample = False
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise TypeError(
                    "TalkSHOW sampler index must be (record_index, is_padding)"
                )
            idx, is_padding_sample = idx
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError(f"TalkSHOW record index must be int, got {idx!r}")
        if not isinstance(is_padding_sample, bool):
            raise TypeError("TalkSHOW padding marker must be bool")

        sample = super().__getitem__(idx)
        sample["is_padding_sample"] = is_padding_sample
        raw_pid = int(sample["id"])
        if raw_pid not in self.expected_pids:
            raise ValueError(
                f"TalkSHOW record {idx} has unsupported raw PID {raw_pid}; "
                f"expected {self.expected_pids}"
            )
        features = sample["audio_feat"]
        if (
            features.ndim != 2
            or features.shape[-1] != self.expected_feature_dim
        ):
            raise ValueError(
                f"TalkSHOW record {idx} audio_feat must have shape "
                f"[T,{self.expected_feature_dim}], got "
                f"{tuple(features.shape)}"
            )
        if features.dtype != torch.float32:
            raise TypeError(
                f"TalkSHOW record {idx} audio_feat must be float32, "
                f"got {features.dtype}"
            )
        motion_length = sample["pose"].shape[0]
        if (
            features.shape[0] != motion_length
            or sample["facial"].shape[0] != motion_length
        ):
            raise ValueError(
                f"TalkSHOW record {idx} has inconsistent frame lengths"
            )
        if self.expected_split in {"train", "val", "validation"}:
            if motion_length != self.expected_seq_size:
                raise ValueError(
                    f"TalkSHOW {self.expected_split} record {idx} has "
                    f"{motion_length} frames, expected "
                    f"{self.expected_seq_size}"
                )
        elif motion_length <= 0:
            raise ValueError(f"TalkSHOW test record {idx} is empty")

        required_context_keys = {
            "face_audio_feat",
            "face_audio_valid_length",
            "face_audio_crop_start",
        }
        missing_context = sorted(required_context_keys.difference(sample))
        if missing_context:
            raise ValueError(
                f"TalkSHOW record {idx} has no strict face-head context; "
                f"missing={missing_context}. Rebuild this LMDB."
            )
        face_features = sample["face_audio_feat"]
        if (
            not isinstance(face_features, torch.Tensor)
            or face_features.ndim != 2
            or face_features.shape[-1] != self.expected_feature_dim
            or face_features.dtype != torch.float32
            or not torch.isfinite(face_features).all()
        ):
            raise ValueError(
                f"TalkSHOW record {idx} face_audio_feat must be finite "
                f"float32 [Tctx,{self.expected_feature_dim}], got "
                f"{getattr(face_features, 'shape', None)} / "
                f"{getattr(face_features, 'dtype', None)}"
            )
        valid_length = sample["face_audio_valid_length"]
        crop_start = sample["face_audio_crop_start"]
        if (
            isinstance(valid_length, bool)
            or not isinstance(valid_length, int)
            or isinstance(crop_start, bool)
            or not isinstance(crop_start, int)
        ):
            raise TypeError(
                f"TalkSHOW record {idx} face context indices must be integers"
            )
        expected_carrier_length = (
            motion_length
            if self.expected_split == "test"
            else self.expected_seq_size + 2 * FACE_CONTEXT_FRAMES
        )
        if face_features.shape[0] != expected_carrier_length:
            raise ValueError(
                f"TalkSHOW record {idx} face context carrier has "
                f"{face_features.shape[0]} frames, expected "
                f"{expected_carrier_length}"
            )
        if not (
            0 < valid_length <= face_features.shape[0]
            and 0 <= crop_start <= FACE_CONTEXT_FRAMES
            and crop_start + motion_length <= valid_length
            and valid_length - crop_start - motion_length
            <= FACE_CONTEXT_FRAMES
        ):
            raise ValueError(
                f"TalkSHOW record {idx} invalid face context bounds: "
                f"valid_length={valid_length}, crop_start={crop_start}, "
                f"motion_length={motion_length}, "
                f"carrier_length={face_features.shape[0]}"
            )
        central_features = face_features[
            crop_start:crop_start + motion_length
        ]
        if not torch.equal(central_features, features):
            raise ValueError(
                f"TalkSHOW record {idx} face context central crop does not "
                "exactly equal audio_feat"
            )
        if (
            valid_length < face_features.shape[0]
            and torch.count_nonzero(face_features[valid_length:]).item() != 0
        ):
            raise ValueError(
                f"TalkSHOW record {idx} face context carrier tail must be zero"
            )
        model_pid = raw_pid - self.model_pid_offset
        if model_pid not in self.expected_model_pids:
            raise ValueError(
                f"TalkSHOW raw PID {raw_pid} maps to unsupported model PID "
                f"{model_pid}; expected {self.expected_model_pids}"
            )
        # ``LMDBDataset.__getitem__`` has already used ``raw_pid`` to load the
        # matching rest-joint entry.  Only the TalkSHOW subclass remaps the ID
        # exposed to the face condition and diffusion backbone.
        sample["raw_id"] = raw_pid
        sample["id"] = model_pid
        return sample


class TalkSHOWDistributedEvalSampler(DistributedSampler):
    """Equal-length distributed eval sampler with explicit padding markers.

    PyTorch's ``DistributedSampler(drop_last=False)`` repeats initial samples
    until every rank has the same number of steps.  Equal step counts are
    required by this project's evaluation barriers, but the repeated samples
    must not be written or counted twice.  The dataset propagates this marker
    and the trainer filters those results after synchronized inference.
    """

    def __init__(self, dataset, **kwargs: Any) -> None:
        if kwargs.get("drop_last", False):
            raise ValueError(
                "TalkSHOWDistributedEvalSampler requires drop_last=false"
            )
        super().__init__(dataset, **kwargs)

    def __iter__(self):
        for local_position, record_index in enumerate(super().__iter__()):
            global_position = (
                self.rank + local_position * self.num_replicas
            )
            yield (
                int(record_index),
                bool(global_position >= len(self.dataset)),
            )


__all__ = [
    "OFFICIAL_HAND_COMPONENT_SHA256",
    "OFFICIAL_SPLIT_SHA256",
    "FACE_CONTEXT_FRAMES",
    "FACE_CONTEXT_PROTOCOL",
    "TALKSHOW_MODEL_PID_OFFSET",
    "TALKSHOW_MODEL_PIDS",
    "TALKSHOW_RAW_PID_OFFSET",
    "TALKSHOW_RAW_PIDS",
    "TALKSHOW_RAW_SPEAKER_IDS",
    "TALKSHOW_SOURCE_SPEAKER_IDS",
    "TalkSHOWDistributedEvalSampler",
    "TalkSHOWLMDBDataset",
]
