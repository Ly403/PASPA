from __future__ import annotations

import importlib.util
import json
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str) -> Any:
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_module(
    "show_lmdb_contract_under_test",
    "tools/show/verify_lmdb_contract.py",
)
REST = _load_module(
    "show_rest_builder_under_test",
    "scripts/build_rest_joint_position_show.py",
)


class FakeCursor:
    def __init__(self, values: dict[bytes, bytes]) -> None:
        self.values = values
        self.iternext_calls: list[tuple[bool, bool]] = []

    def iternext(self, *, keys: bool = True, values: bool = True):
        self.iternext_calls.append((keys, values))
        if not keys or values:
            raise AssertionError(
                "contract key scan must request keys=True, values=False"
            )
        return iter(sorted(self.values))


class FakeTransaction:
    def __init__(self, values: dict[bytes, bytes]) -> None:
        self.values = values
        self.cursors: list[FakeCursor] = []

    def get(self, key: bytes) -> bytes | None:
        return self.values.get(key)

    def cursor(self) -> FakeCursor:
        cursor = FakeCursor(self.values)
        self.cursors.append(cursor)
        return cursor


SPEAKER_IDS = {
    "oliver": 21,
    "chemistry": 22,
    "seth": 23,
    "conan": 24,
}
EXPECTED_PIDS = tuple(sorted(SPEAKER_IDS.values()))
EXPECTED_PID_OFFSET = 1


def _metadata(length: int, *, split: str = "train") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset": "SHOW",
        "split": split,
        # Preserve the common input traversal order deliberately: identity
        # semantics come from speaker_ids, not this list's order.
        "speakers": ["oliver", "seth", "conan", "chemistry"],
        "speaker_ids": dict(SPEAKER_IDS),
        "pid_offset": EXPECTED_PID_OFFSET,
        "seq_size": 2,
        "stride_size": 1,
        "local_rotation": False,
        "pose_representation": "axis_angle",
        "pose_order": list(CONTRACT.EXPECTED_POSE_ORDER),
        "rotation_space": "global",
        "feature_scope": "full_clip",
        "face_context_frames": CONTRACT.FACE_CONTEXT_FRAMES,
        "face_context_protocol": CONTRACT.FACE_CONTEXT_PROTOCOL,
        "hand_component_sha256": (
            CONTRACT.OFFICIAL_HAND_COMPONENT_SHA256
        ),
        "model_type": "wavlm",
        "audio_model_id": "/models/wavlm-large",
        "audio_config_sha256": "1" * 64,
        "audio_preprocessor_sha256": "2" * 64,
        "audio_model_files_sha256": {
            "config.json": "3" * 64,
            "preprocessor_config.json": "4" * 64,
            "model.safetensors": "5" * 64,
        },
        "audio_do_normalize": True,
        "audio_feat_extract_norm": "layer",
        "audio_add_adapter": False,
        "audio_feature_dim": 4,
        "audio_sr": 6,
        "pose_fps": 3,
        "split_manifest_sha256": CONTRACT.OFFICIAL_SPLIT_SHA256,
        "split_view_summary_sha256": "6" * 64,
        "split_view_counts": {"train": 4, "val": 4, "test": 4},
        "split_view_missing_count": 0,
        "split_view_limit_per_speaker_split": None,
        "source_clip_count": 4,
        "output_sample_count": length,
        "stats": {"written": length},
        "max_av_duration_delta_frames": 3.0,
        "duration_mismatch_policy": "error",
    }


def _record(
    speaker: str,
    *,
    frames: int = 2,
    feature_dim: int = 4,
    include_face_context: bool = False,
) -> dict[str, Any]:
    record = {
        "betas": np.zeros((300,), dtype=np.float32),
        "poses": np.zeros((frames, 165), dtype=np.float32),
        "expressions": np.zeros((frames, 100), dtype=np.float32),
        "trans": np.zeros((frames, 3), dtype=np.float32),
        "audio": np.zeros((frames * 2,), dtype=np.float32),
        "huberts": np.zeros(
            (1, frames, feature_dim),
            dtype=np.float32,
        ),
        "pid": SPEAKER_IDS[speaker],
        "ori_vid": f"{speaker}__clip",
    }
    if include_face_context:
        carrier_length = frames + 2 * CONTRACT.FACE_CONTEXT_FRAMES
        face_huberts = np.zeros(
            (1, carrier_length, feature_dim),
            dtype=np.float32,
        )
        face_huberts[:, :frames, :] = record["huberts"]
        record.update(
            {
                "face_huberts": face_huberts,
                "face_hubert_valid_length": frames,
                "face_hubert_crop_start": 0,
            }
        )
    return record


def _transaction(
    *,
    include_metadata: bool = True,
    split: str = "train",
) -> FakeTransaction:
    records = [
        _record("oliver", include_face_context=include_metadata),
        _record("chemistry", include_face_context=include_metadata),
        _record("seth", include_face_context=include_metadata),
        _record("conan", include_face_context=include_metadata),
    ]
    values = {
        b"__len__": str(len(records)).encode("ascii"),
        **{
            f"{index:010d}".encode("ascii"): pickle.dumps(record)
            for index, record in enumerate(records)
        },
    }
    if include_metadata:
        values[b"__meta__"] = json.dumps(
            _metadata(len(records), split=split)
        ).encode("utf-8")
    return FakeTransaction(values)


def _verify(transaction: FakeTransaction, **overrides: Any) -> dict[str, Any]:
    options = {
        "expected_split": "train",
        "expected_pids": EXPECTED_PIDS,
        "expected_seq_size": 2,
        "expected_feature_dim": 4,
        "expected_pid_offset": EXPECTED_PID_OFFSET,
        "audio_sr": 6,
        "pose_fps": 3,
        "allow_legacy": False,
        "index": None,
        "progress_every": 0,
        "expected_max_av_duration_delta_frames": 3.0,
        "progress_callback": None,
    }
    options.update(overrides)
    return CONTRACT.verify_transaction(transaction, **options)


class ShowLmdbContractTest(unittest.TestCase):
    def test_strict_full_scan_accepts_canonical_contract(self) -> None:
        transaction = _transaction()
        summary = _verify(transaction)
        self.assertTrue(summary["full_scan"])
        self.assertFalse(summary["legacy"])
        self.assertEqual(summary["checked_records"], 4)
        self.assertEqual(summary["observed_pids"], list(EXPECTED_PIDS))
        self.assertEqual(summary["min_frames"], 2)
        self.assertEqual(summary["max_frames"], 2)
        metadata = summary["metadata"]
        self.assertEqual(
            metadata["split_manifest_sha256"],
            CONTRACT.OFFICIAL_SPLIT_SHA256,
        )
        self.assertEqual(
            metadata["split_view_counts"],
            {"train": 4, "val": 4, "test": 4},
        )
        self.assertEqual(len(transaction.cursors), 1)
        self.assertEqual(
            transaction.cursors[0].iternext_calls,
            [(True, False)],
        )

    def test_metadata_is_validated_before_numeric_key_scan(self) -> None:
        transaction = _transaction()
        metadata = json.loads(
            transaction.values[b"__meta__"].decode("utf-8")
        )
        metadata["seq_size"] = 128
        transaction.values[b"__meta__"] = json.dumps(metadata).encode("utf-8")

        with self.assertRaisesRegex(CONTRACT.ContractError, "seq_size"):
            _verify(transaction)

        self.assertEqual(transaction.cursors, [])

    def test_split_provenance_fields_are_required_in_strict_mode(self) -> None:
        fields = (
            "split_manifest_sha256",
            "split_view_summary_sha256",
            "split_view_counts",
            "split_view_missing_count",
            "split_view_limit_per_speaker_split",
        )
        for field in fields:
            with self.subTest(field=field):
                transaction = _transaction()
                metadata = json.loads(
                    transaction.values[b"__meta__"].decode("utf-8")
                )
                del metadata[field]
                transaction.values[b"__meta__"] = json.dumps(metadata).encode(
                    "utf-8"
                )
                with self.assertRaisesRegex(
                    CONTRACT.ContractError,
                    field,
                ):
                    _verify(transaction)

    def test_expected_duration_threshold_rejects_missing_or_drifted_value(
        self,
    ) -> None:
        for value in (None, 2.0):
            with self.subTest(value=value):
                transaction = _transaction()
                metadata = json.loads(
                    transaction.values[b"__meta__"].decode("utf-8")
                )
                metadata["max_av_duration_delta_frames"] = value
                transaction.values[b"__meta__"] = json.dumps(metadata).encode(
                    "utf-8"
                )
                with self.assertRaisesRegex(
                    CONTRACT.ContractError,
                    "max_av_duration_delta_frames",
                ):
                    _verify(transaction)

    def test_legacy_schema_two_error_policy_can_omit_policy_field(self) -> None:
        transaction = _transaction()
        metadata = json.loads(
            transaction.values[b"__meta__"].decode("utf-8")
        )
        del metadata["duration_mismatch_policy"]
        transaction.values[b"__meta__"] = json.dumps(metadata).encode("utf-8")

        summary = _verify(transaction)

        self.assertIsNone(summary["metadata"]["duration_mismatch_policy"])
        self.assertEqual(
            summary["metadata"]["max_av_duration_delta_frames"],
            3.0,
        )

    def test_skip_policy_requires_duration_skip_statistics(self) -> None:
        transaction = _transaction()
        metadata = json.loads(
            transaction.values[b"__meta__"].decode("utf-8")
        )
        metadata["duration_mismatch_policy"] = "skip"
        transaction.values[b"__meta__"] = json.dumps(metadata).encode("utf-8")

        with self.assertRaisesRegex(
            CONTRACT.ContractError,
            "stats.processing",
        ):
            _verify(transaction)

    def test_split_and_hand_provenance_reject_digest_drift(self) -> None:
        mutations = (
            (
                "split_manifest_sha256",
                "f" * 64,
                "pinned official manifest",
            ),
            (
                "split_view_summary_sha256",
                "not-a-sha256",
                "split_view_summary_sha256",
            ),
            (
                "hand_component_sha256",
                "f" * 64,
                "canonical TalkSHOW hand basis",
            ),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                transaction = _transaction()
                metadata = json.loads(
                    transaction.values[b"__meta__"].decode("utf-8")
                )
                metadata[field] = value
                transaction.values[b"__meta__"] = json.dumps(metadata).encode(
                    "utf-8"
                )
                with self.assertRaisesRegex(
                    CONTRACT.ContractError,
                    message,
                ):
                    _verify(transaction)

    def test_split_provenance_rejects_invalid_counts_and_limits(self) -> None:
        mutations = (
            (
                lambda metadata: metadata["split_view_counts"].pop("test"),
                "exactly train/val/test",
            ),
            (
                lambda metadata: metadata["split_view_counts"].update(val=0),
                r"split_view_counts\['val'\].*>= 1",
            ),
            (
                lambda metadata: metadata["split_view_counts"].update(
                    train=3
                ),
                "disagrees with source_clip_count",
            ),
            (
                lambda metadata: metadata.update(
                    split_view_missing_count=-1
                ),
                "split_view_missing_count.*>= 0",
            ),
            (
                lambda metadata: metadata.update(
                    split_view_limit_per_speaker_split=0
                ),
                "split_view_limit_per_speaker_split.*>= 1",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                transaction = _transaction()
                metadata = json.loads(
                    transaction.values[b"__meta__"].decode("utf-8")
                )
                mutation(metadata)
                transaction.values[b"__meta__"] = json.dumps(metadata).encode(
                    "utf-8"
                )
                with self.assertRaisesRegex(
                    CONTRACT.ContractError,
                    message,
                ):
                    _verify(transaction)

    def test_numeric_keys_must_be_exactly_contiguous(self) -> None:
        transaction = _transaction()
        del transaction.values[b"0000000002"]
        with self.assertRaisesRegex(CONTRACT.ContractError, "contiguous"):
            _verify(transaction)

        transaction = _transaction()
        transaction.values[b"0000000004"] = pickle.dumps(_record("oliver"))
        with self.assertRaisesRegex(CONTRACT.ContractError, "contiguous"):
            _verify(transaction)

    def test_empty_lmdb_is_rejected(self) -> None:
        transaction = FakeTransaction({b"__len__": b"0"})
        with self.assertRaisesRegex(CONTRACT.ContractError, "empty"):
            _verify(transaction)

    def test_missing_metadata_fails_strict_and_warns_in_legacy_mode(self) -> None:
        transaction = _transaction(include_metadata=False)
        with self.assertRaisesRegex(CONTRACT.ContractError, "strict mode"):
            _verify(transaction)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = _verify(
                transaction,
                allow_legacy=True,
                expected_max_av_duration_delta_frames=None,
            )
        self.assertTrue(summary["legacy"])
        self.assertTrue(
            any("LEGACY SHOW CONTRACT" in str(item.message) for item in caught)
        )

    def test_partial_mode_is_explicit_and_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = _verify(_transaction(), index=2)
        self.assertFalse(summary["full_scan"])
        self.assertEqual(summary["checked_records"], 1)
        self.assertTrue(
            any("PARTIAL SHOW CONTRACT" in str(item.message) for item in caught)
        )

    def test_metadata_rejects_identity_and_feature_scope_drift(self) -> None:
        for mutation, message in (
            (
                lambda meta: meta["speaker_ids"].update(
                    {"chemistry": 23, "seth": 22}
                ),
                "canonical identity mapping",
            ),
            (
                lambda meta: meta.update(feature_scope="window"),
                "full_clip",
            ),
            (
                lambda meta: meta.update(
                    pose_order=list(reversed(meta["pose_order"]))
                ),
                "pose_order",
            ),
        ):
            with self.subTest(message=message):
                transaction = _transaction()
                meta = json.loads(
                    transaction.values[b"__meta__"].decode("utf-8")
                )
                mutation(meta)
                transaction.values[b"__meta__"] = json.dumps(meta).encode(
                    "utf-8"
                )
                with self.assertRaisesRegex(CONTRACT.ContractError, message):
                    _verify(transaction)

    def test_record_array_contract_is_fail_closed(self) -> None:
        base = _record("oliver")
        mutations = (
            ("dtype", lambda item: item.update(
                poses=item["poses"].astype(np.float64)
            )),
            ("shape", lambda item: item.update(
                huberts=np.zeros((2, 2, 4), dtype=np.float32)
            )),
            ("NaN or Inf", lambda item: item["expressions"].__setitem__(
                (0, 0), np.nan
            )),
            ("audio", lambda item: item.update(
                audio=np.zeros((3,), dtype=np.float32)
            )),
            ("outside expected", lambda item: item.update(pid=25)),
        )
        for message, mutation in mutations:
            with self.subTest(message=message):
                record = {
                    key: value.copy()
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in base.items()
                }
                mutation(record)
                with self.assertRaisesRegex(CONTRACT.ContractError, message):
                    CONTRACT.validate_record(
                        record,
                        index=0,
                        split="train",
                        expected_pids=EXPECTED_PIDS,
                        expected_seq_size=2,
                        expected_feature_dim=4,
                        audio_sr=6,
                        pose_fps=3,
                        speaker_ids=SPEAKER_IDS,
                    )

    def test_train_is_fixed_length_but_test_can_be_nonempty(self) -> None:
        record = _record("oliver", frames=3)
        with self.assertRaisesRegex(CONTRACT.ContractError, "fixed T=2"):
            CONTRACT.validate_record(
                record,
                index=0,
                split="train",
                expected_pids=EXPECTED_PIDS,
                expected_seq_size=2,
                expected_feature_dim=4,
                audio_sr=6,
                pose_fps=3,
                speaker_ids=SPEAKER_IDS,
            )
        _, _, frames = CONTRACT.validate_record(
            record,
            index=0,
            split="test",
            expected_pids=EXPECTED_PIDS,
            expected_seq_size=2,
            expected_feature_dim=4,
            audio_sr=6,
            pose_fps=3,
            speaker_ids=SPEAKER_IDS,
        )
        self.assertEqual(frames, 3)

    def test_rest_joint_mapping_requires_all_finite_55_joint_tables(self) -> None:
        valid = {
            str(pid): np.zeros((55, 3), dtype=np.float32)
            for pid in SPEAKER_IDS.values()
        }
        normalized = CONTRACT.validate_rest_joint_mapping(
            valid,
            expected_pids=EXPECTED_PIDS,
        )
        self.assertEqual(set(normalized), set(EXPECTED_PIDS))

        missing = dict(valid)
        del missing["24"]
        with self.assertRaisesRegex(CONTRACT.ContractError, "PID keys"):
            CONTRACT.validate_rest_joint_mapping(
                missing,
                expected_pids=EXPECTED_PIDS,
            )

        wrong_dtype = dict(valid)
        wrong_dtype["24"] = valid["24"].astype(np.float64)
        with self.assertRaisesRegex(CONTRACT.ContractError, "expected float32"):
            CONTRACT.validate_rest_joint_mapping(
                wrong_dtype,
                expected_pids=EXPECTED_PIDS,
            )

        nonfinite = dict(valid)
        nonfinite["24"] = valid["24"].copy()
        nonfinite["24"][0, 0] = np.inf
        with self.assertRaisesRegex(CONTRACT.ContractError, "NaN or Inf"):
            CONTRACT.validate_rest_joint_mapping(
                nonfinite,
                expected_pids=EXPECTED_PIDS,
            )


class ShowAudioIdentityTest(unittest.TestCase):
    def test_canonical_json_hash_ignores_key_order_and_whitespace(self) -> None:
        first = {"z": 1, "nested": {"b": False, "a": "中文"}}
        second = {"nested": {"a": "中文", "b": False}, "z": 1}
        self.assertEqual(
            CONTRACT.canonical_json_sha256(first),
            CONTRACT.canonical_json_sha256(second),
        )

    def test_runtime_wavlm_hashes_every_metadata_declared_file(self) -> None:
        config = {
            "model_type": "wavlm",
            "hidden_size": 4,
            "feat_extract_norm": "layer",
            "add_adapter": False,
        }
        preprocessor = {"do_normalize": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(config, indent=2),
                encoding="utf-8",
            )
            (root / "preprocessor_config.json").write_text(
                json.dumps(preprocessor),
                encoding="utf-8",
            )
            (root / "model.safetensors").write_bytes(b"test weights")
            expected_files = {
                name: CONTRACT.file_sha256(root / name)
                for name in (
                    "config.json",
                    "preprocessor_config.json",
                    "model.safetensors",
                )
            }
            identity = CONTRACT.load_wavlm_identity(
                root,
                expected_files=expected_files,
                expected_feature_dim=4,
            )
            (root / "tokenizer_config.json").write_text(
                "{}",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CONTRACT.ContractError,
                "recognized file set",
            ):
                CONTRACT.load_wavlm_identity(
                    root,
                    expected_files=expected_files,
                    expected_feature_dim=4,
                )

        self.assertEqual(
            identity["audio_config_sha256"],
            CONTRACT.canonical_json_sha256(config),
        )
        self.assertEqual(
            identity["audio_preprocessor_sha256"],
            CONTRACT.canonical_json_sha256(preprocessor),
        )
        self.assertEqual(
            identity["audio_model_files_sha256"],
            expected_files,
        )

    def test_runtime_wavlm_hashes_every_index_referenced_shard(self) -> None:
        config = {
            "model_type": "wavlm",
            "hidden_size": 4,
            "feat_extract_norm": "layer",
            "add_adapter": False,
        }
        preprocessor = {"do_normalize": False}
        index = {
            "weight_map": {
                "encoder.layer.0.weight": "weights/model-00001.bin",
                "encoder.layer.1.weight": "weights/model-00002.bin",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights").mkdir()
            (root / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            (root / "preprocessor_config.json").write_text(
                json.dumps(preprocessor),
                encoding="utf-8",
            )
            (root / "pytorch_model.bin.index.json").write_text(
                json.dumps(index),
                encoding="utf-8",
            )
            (root / "weights/model-00001.bin").write_bytes(b"first")
            (root / "weights/model-00002.bin").write_bytes(b"second")
            filenames = (
                "config.json",
                "preprocessor_config.json",
                "pytorch_model.bin.index.json",
                "weights/model-00001.bin",
                "weights/model-00002.bin",
            )
            expected_files = {
                name: CONTRACT.file_sha256(root / name)
                for name in filenames
            }
            identity = CONTRACT.load_wavlm_identity(
                root,
                expected_files=expected_files,
                expected_feature_dim=4,
            )
        self.assertEqual(
            identity["audio_model_files_sha256"],
            expected_files,
        )

    def test_face_lmdb_runtime_comparison_rejects_any_drift(self) -> None:
        metadata = _metadata(4)
        face = {
            "speakers": list(SPEAKER_IDS),
            "audio_hidden_size": 4,
            "audio_do_normalize": True,
            "audio_config_sha256": metadata["audio_config_sha256"],
            "audio_preprocessor_sha256": metadata[
                "audio_preprocessor_sha256"
            ],
        }
        runtime = {
            "audio_config_sha256": metadata["audio_config_sha256"],
            "audio_preprocessor_sha256": metadata[
                "audio_preprocessor_sha256"
            ],
            "audio_model_files_sha256": metadata[
                "audio_model_files_sha256"
            ],
            "audio_do_normalize": True,
            "audio_feat_extract_norm": "layer",
            "audio_add_adapter": False,
        }
        CONTRACT.validate_audio_identity(
            metadata,
            face_contract=face,
            wavlm_identity=runtime,
        )

        runtime["audio_model_files_sha256"] = {
            **runtime["audio_model_files_sha256"],
            "model.safetensors": "f" * 64,
        }
        with self.assertRaisesRegex(CONTRACT.ContractError, "file hashes"):
            CONTRACT.validate_audio_identity(
                metadata,
                face_contract=face,
                wavlm_identity=runtime,
            )


class ShowRestBuilderTest(unittest.TestCase):
    @staticmethod
    def _beta(value: float) -> np.ndarray:
        return np.full((300,), value, dtype=np.float32)

    def test_rest_builder_requires_strict_train_metadata(self) -> None:
        metadata = _metadata(4, split="train")
        validated = REST.validate_source_lmdb_metadata(
            json.dumps(metadata).encode("utf-8"),
            expected_pids=EXPECTED_PIDS,
            expected_pid_offset=EXPECTED_PID_OFFSET,
            lmdb_length=4,
        )
        self.assertEqual(validated["split"], "train")

        metadata["split"] = "val"
        with self.assertRaisesRegex(
            REST.RestJointBuildError,
            "expected 'train'",
        ):
            REST.validate_source_lmdb_metadata(
                json.dumps(metadata).encode("utf-8"),
                expected_pids=EXPECTED_PIDS,
                expected_pid_offset=EXPECTED_PID_OFFSET,
                lmdb_length=4,
            )

    def test_deduplicates_windows_before_applying_source_clip_cap(self) -> None:
        records: list[tuple[int, dict[str, Any]]] = []
        index = 0
        for speaker, pid in SPEAKER_IDS.items():
            first = {
                "pid": pid,
                "ori_vid": f"{speaker}__clip_a",
                "betas": self._beta(float(pid)),
            }
            records.append((index, first))
            index += 1
            records.append(
                (
                    index,
                    {
                        **first,
                        "betas": first["betas"].copy(),
                    },
                )
            )
            index += 1
            records.append(
                (
                    index,
                    {
                        "pid": pid,
                        "ori_vid": f"{speaker}__clip_b",
                        "betas": self._beta(float(pid) + 0.5),
                    },
                )
            )
            index += 1

        per_pid, stats = REST.collect_unique_betas(
            records,
            max_clips_per_pid=1,
        )
        self.assertEqual(stats["record_count"], 12)
        self.assertEqual(stats["unique_source_clip_count"], 8)
        self.assertEqual(stats["duplicate_window_count"], 4)
        self.assertEqual(
            stats["unique_clips_per_pid"],
            {21: 2, 22: 2, 23: 2, 24: 2},
        )
        self.assertTrue(all(len(values) == 1 for values in per_pid.values()))

    def test_conflicting_betas_for_one_source_clip_are_rejected(self) -> None:
        records = []
        for index, (speaker, pid) in enumerate(SPEAKER_IDS.items()):
            records.append(
                (
                    index,
                    {
                        "pid": pid,
                        "ori_vid": f"{speaker}__clip",
                        "betas": self._beta(float(pid)),
                    },
                )
            )
        records.append(
            (
                4,
                {
                    "pid": 21,
                    "ori_vid": "oliver__clip",
                    "betas": self._beta(999.0),
                },
            )
        )
        with self.assertRaisesRegex(
            REST.RestJointBuildError,
            "conflicting betas",
        ):
            REST.collect_unique_betas(records, max_clips_per_pid=0)

    def test_strict_beta_format_and_cap_type(self) -> None:
        valid_records = [
            (
                index,
                {
                    "pid": pid,
                    "ori_vid": f"{speaker}__clip",
                    "betas": self._beta(float(pid)),
                },
            )
            for index, (speaker, pid) in enumerate(SPEAKER_IDS.items())
        ]
        with self.assertRaisesRegex(ValueError, "integer"):
            REST.collect_unique_betas(
                valid_records,
                max_clips_per_pid=1.5,
            )

        for bad_beta, message in (
            (np.zeros((10,), dtype=np.float32), "shape"),
            (np.zeros((300,), dtype=np.float64), "dtype"),
            (
                np.full((300,), np.nan, dtype=np.float32),
                "NaN or Inf",
            ),
        ):
            with self.subTest(message=message):
                records = list(valid_records)
                records[0] = (
                    0,
                    {
                        **records[0][1],
                        "betas": bad_beta,
                    },
                )
                with self.assertRaisesRegex(
                    REST.RestJointBuildError,
                    message,
                ):
                    REST.collect_unique_betas(
                        records,
                        max_clips_per_pid=0,
                    )


if __name__ == "__main__":
    unittest.main()
