from __future__ import annotations

import io
import importlib.util
import json
import pickle
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "create_lmdb_show.py"


def _load_module_without_training_dependencies():
    """Import the ETL helpers without requiring torch/librosa/lmdb/scipy."""

    lmdb = types.ModuleType("lmdb")
    lmdb.open = lambda *args, **kwargs: None

    librosa = types.ModuleType("librosa")
    librosa.load = lambda *args, **kwargs: (np.zeros(1), 16000)

    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.no_grad = lambda: None
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.__path__ = []
    torch_functional = types.ModuleType("torch.nn.functional")
    torch_functional.interpolate = lambda value, *args, **kwargs: value
    torch.nn = torch_nn
    torch_nn.functional = torch_functional

    scipy = types.ModuleType("scipy")
    scipy.__path__ = []
    scipy_spatial = types.ModuleType("scipy.spatial")
    scipy_spatial.__path__ = []
    scipy_transform = types.ModuleType("scipy.spatial.transform")

    class _UnusedRotation:
        pass

    scipy_transform.Rotation = _UnusedRotation
    scipy.spatial = scipy_spatial
    scipy_spatial.transform = scipy_transform

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable

    stubs = {
        "lmdb": lmdb,
        "librosa": librosa,
        "torch": torch,
        "torch.nn": torch_nn,
        "torch.nn.functional": torch_functional,
        "scipy": scipy,
        "scipy.spatial": scipy_spatial,
        "scipy.spatial.transform": scipy_transform,
        "tqdm": tqdm_module,
    }
    module_name = "create_lmdb_show_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


MODULE = _load_module_without_training_dependencies()


class _FakeTransaction:
    def __init__(self, values):
        self.values = values
        self.committed = False
        self.aborted = False

    def put(self, key, value):
        self.values[key] = value

    def commit(self):
        self.committed = True

    def abort(self):
        self.aborted = True


class _FakeEnvironment:
    def __init__(self):
        self.values = {}
        self.transactions = []
        self.closed = False

    def begin(self, write=False):
        transaction = _FakeTransaction(self.values)
        self.transactions.append(transaction)
        return transaction

    def close(self):
        self.closed = True


class _FakeAudioFeatureExtractor:
    instances = []
    feature_dim_value = 4

    def __init__(self, model_type, model_id):
        self.model_type = model_type
        self.model_id = model_id
        self.feature_dim = self.feature_dim_value
        self.calls = []
        self.__class__.instances.append(self)

    def __call__(self, raw_audio, seq_size=None):
        self.calls.append((len(raw_audio), seq_size))
        frame_values = np.arange(seq_size, dtype=np.float32)
        features = np.repeat(
            frame_values.reshape(1, seq_size, 1),
            self.feature_dim,
            axis=2,
        )
        return features


def _motion_payload(num_frames):
    return {
        "betas": np.zeros(10, dtype=np.float32),
        "global_orient": np.zeros((num_frames, 3), dtype=np.float32),
        "body_pose_axis": np.zeros((num_frames, 63), dtype=np.float32),
        "jaw_pose": np.zeros((num_frames, 3), dtype=np.float32),
        "leye_pose": np.zeros((num_frames, 3), dtype=np.float32),
        "reye_pose": np.zeros((num_frames, 3), dtype=np.float32),
        "left_hand_pose": np.zeros((num_frames, 12), dtype=np.float32),
        "right_hand_pose": np.zeros((num_frames, 12), dtype=np.float32),
        "expression": np.zeros((num_frames, 100), dtype=np.float32),
    }


def _make_show_clip(
    root,
    *,
    num_frames,
    split="train",
    sequence_id="sequence_1",
):
    sequence_dir = root / "oliver" / "video_1" / split / sequence_id
    sequence_dir.mkdir(parents=True)
    pkl_path = sequence_dir / f"{sequence_id}.pkl"
    wav_path = sequence_dir / f"{sequence_id}.wav"
    pkl_path.write_bytes(pickle.dumps(_motion_payload(num_frames)))
    wav_path.touch()
    return pkl_path, wav_path


def _make_local_wavlm(root):
    model_dir = root / "wavlm"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "wavlm",
                "hidden_size": 4,
                "feat_extract_norm": "layer",
                "add_adapter": False,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps({"do_normalize": True}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"test-weights")
    return model_dir


class CreateLMDBShowTest(unittest.TestCase):
    def setUp(self):
        _FakeAudioFeatureExtractor.instances.clear()

    def _run_etl(
        self,
        data_root,
        raw_audio,
        *,
        feature_scope="full_clip",
        **etl_options,
    ):
        environment = _FakeEnvironment()
        model_dir = _make_local_wavlm(data_root)
        with (
            mock.patch.object(
                MODULE,
                "AudioFeatureExtractor",
                _FakeAudioFeatureExtractor,
            ),
            mock.patch.object(
                MODULE.lmdb,
                "open",
                return_value=environment,
            ),
            mock.patch.object(
                MODULE.librosa,
                "load",
                return_value=(raw_audio, MODULE.AUDIO_SR),
            ),
        ):
            metadata = MODULE.create_lmdb_show(
                data_folder=str(data_root),
                lmdb_path=str(data_root / "output.lmdb"),
                seq_size=256,
                stride_size=20,
                local_rotation=True,
                split="train",
                speakers=["oliver"],
                model_type="wavlm",
                model_id=str(model_dir),
                feature_scope=feature_scope,
                **etl_options,
            )
        return metadata, environment, _FakeAudioFeatureExtractor.instances[-1]

    def test_repository_hand_component_is_the_default_and_is_valid(self):
        self.assertEqual(
            MODULE.DEFAULT_HAND_COMPONENT_PATH,
            REPO_ROOT / "tools" / "show" / "hand_component.json",
        )
        left, right = MODULE.load_hand_pca(
            MODULE.DEFAULT_HAND_COMPONENT_PATH
        )
        self.assertEqual(left.shape, (12, 45))
        self.assertEqual(right.shape, (12, 45))
        self.assertEqual(left.dtype, np.float32)
        self.assertEqual(right.dtype, np.float32)

    def test_console_stats_bound_duration_mismatch_details(self):
        details = [{"clip": f"clip_{index}"} for index in range(25)]
        stats = {"duration_mismatch_skipped_clips": details}

        bounded = MODULE._bounded_console_stats(stats)

        self.assertEqual(
            len(bounded["duration_mismatch_skipped_clips"]),
            20,
        )
        self.assertEqual(
            bounded["duration_mismatch_skipped_clips_omitted"],
            5,
        )
        self.assertEqual(len(stats["duration_mismatch_skipped_clips"]), 25)

    def test_strict_beta_layouts(self):
        padded = MODULE.normalize_show_betas(
            np.arange(10, dtype=np.float32),
            num_frames=3,
        )
        self.assertEqual(padded.shape, (300,))
        np.testing.assert_array_equal(
            padded[:10],
            np.arange(10, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            padded[10:],
            np.zeros(290, dtype=np.float32),
        )

        full = np.arange(300, dtype=np.float32)
        np.testing.assert_array_equal(
            MODULE.normalize_show_betas(full, num_frames=3),
            full,
        )
        per_frame = np.repeat(full.reshape(1, 300), 3, axis=0)
        np.testing.assert_array_equal(
            MODULE.normalize_show_betas(per_frame, num_frames=3),
            full,
        )

        malformed = (
            np.zeros(30, dtype=np.float32),
            np.zeros((2, 10), dtype=np.float32),
            np.zeros((3, 11), dtype=np.float32),
        )
        for value in malformed:
            with self.subTest(shape=value.shape):
                with self.assertRaises(MODULE.ShowClipValidationError):
                    MODULE.normalize_show_betas(value, num_frames=3)

        varying = np.zeros((3, 10), dtype=np.float32)
        varying[1, 0] = 1.0
        with self.assertRaisesRegex(
            MODULE.ShowClipValidationError,
            "not constant",
        ):
            MODULE.normalize_show_betas(varying, num_frames=3)

    def test_local_wavlm_provenance_hashes_key_files(self):
        config = {
            "model_type": "wavlm",
            "hidden_size": 1024,
            "feat_extract_norm": "layer",
            "add_adapter": False,
        }
        preprocessor = {"do_normalize": True}
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            (model_dir / "preprocessor_config.json").write_text(
                json.dumps(preprocessor),
                encoding="utf-8",
            )
            (model_dir / "model.safetensors").write_bytes(b"test-weights")

            provenance = MODULE.collect_audio_model_provenance(
                model_dir,
                "wavlm",
            )

        self.assertEqual(
            provenance["audio_config_sha256"],
            MODULE.canonical_json_sha256(config),
        )
        self.assertEqual(
            provenance["audio_preprocessor_sha256"],
            MODULE.canonical_json_sha256(preprocessor),
        )
        self.assertEqual(provenance["configured_audio_feature_dim"], 1024)
        self.assertTrue(provenance["audio_do_normalize"])
        self.assertEqual(provenance["audio_feat_extract_norm"], "layer")
        self.assertFalse(provenance["audio_add_adapter"])
        self.assertEqual(
            set(provenance["audio_model_files_sha256"]),
            {
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            },
        )

    def test_metadata_is_utf8_json_with_flat_contract_fields(self):
        provenance = {
            "audio_model_id": "/models/wavlm-large",
            "audio_config_sha256": "1" * 64,
            "audio_preprocessor_sha256": "2" * 64,
            "audio_model_files_sha256": {"model.safetensors": "3" * 64},
            "audio_do_normalize": True,
            "audio_feat_extract_norm": "layer",
            "audio_add_adapter": False,
            "configured_audio_feature_dim": 1024,
        }
        metadata = MODULE.build_lmdb_metadata(
            split="train",
            speakers=["oliver", "chemistry", "seth", "conan"],
            pid_offset=0,
            seq_size=256,
            stride_size=20,
            local_rotation=False,
            model_type="wavlm",
            feature_scope="full_clip",
            hand_component_path=MODULE.DEFAULT_HAND_COMPONENT_PATH,
            hand_component_sha256=MODULE.OFFICIAL_HAND_COMPONENT_SHA256,
            audio_provenance=provenance,
            audio_feature_dim=1024,
            split_provenance={
                "split_manifest_sha256": MODULE.OFFICIAL_SPLIT_SHA256,
                "split_view_summary_sha256": "5" * 64,
                "split_view_counts": {
                    "train": 10,
                    "val": 8,
                    "test": 7,
                },
                "split_view_missing_count": 0,
                "split_view_limit_per_speaker_split": None,
            },
            source_clip_count=10,
            output_sample_count=20,
            stats={"skipped": {"short": np.int64(2)}},
        )
        encoded = MODULE.encode_lmdb_metadata(metadata)
        decoded = json.loads(encoded.decode("utf-8"))

        self.assertEqual(decoded["schema_version"], 2)
        self.assertEqual(decoded["dataset"], "SHOW")
        self.assertEqual(decoded["split"], "train")
        self.assertEqual(
            decoded["speaker_ids"],
            {"oliver": 20, "chemistry": 21, "seth": 22, "conan": 23},
        )
        self.assertEqual(decoded["audio_feature_dim"], 1024)
        self.assertEqual(decoded["audio_sr"], 16000)
        self.assertEqual(decoded["pose_fps"], 30)
        self.assertEqual(decoded["feature_scope"], "full_clip")
        self.assertEqual(
            decoded["face_context_frames"],
            MODULE.FACE_CONTEXT_FRAMES,
        )
        self.assertEqual(
            decoded["face_context_protocol"],
            MODULE.FACE_CONTEXT_PROTOCOL,
        )
        self.assertEqual(decoded["rotation_space"], "global")
        self.assertEqual(decoded["pose_order"], list(MODULE.POSE_ORDER))
        self.assertEqual(
            decoded["split_manifest_sha256"],
            MODULE.OFFICIAL_SPLIT_SHA256,
        )
        self.assertEqual(
            decoded["split_view_counts"],
            {"train": 10, "val": 8, "test": 7},
        )
        self.assertEqual(decoded["stats"]["skipped"]["short"], 2)

    def test_split_view_provenance_is_pinned_and_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            summary_path = data_root / "split_view_summary.json"
            summary = {
                "split_sha256": MODULE.OFFICIAL_SPLIT_SHA256,
                "counts": {"train": 10, "val": 8, "test": 7},
                "missing_count": 55,
                "limit_per_speaker_split": None,
            }
            summary_path.write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            provenance = MODULE.load_split_view_provenance(
                data_root,
                split="train",
                speakers=["oliver", "seth", "conan", "chemistry"],
                usable_clip_count=10,
                required=True,
            )

            self.assertEqual(
                provenance["split_manifest_sha256"],
                MODULE.OFFICIAL_SPLIT_SHA256,
            )
            self.assertEqual(
                provenance["split_view_summary_sha256"],
                MODULE.sha256_file(summary_path),
            )
            self.assertEqual(provenance["split_view_counts"], summary["counts"])
            self.assertEqual(provenance["split_view_missing_count"], 55)

            summary["split_sha256"] = "0" * 64
            summary_path.write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "pinned official manifest",
            ):
                MODULE.load_split_view_provenance(
                    data_root,
                    split="train",
                    speakers=["oliver", "seth", "conan", "chemistry"],
                    usable_clip_count=10,
                    required=True,
                )

    def test_required_split_view_summary_cannot_be_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                FileNotFoundError,
                "split_view_summary.json",
            ):
                MODULE.load_split_view_provenance(
                    temporary,
                    split="train",
                    speakers=["oliver", "seth", "conan", "chemistry"],
                    usable_clip_count=10,
                    required=True,
                )

    def test_zero_input_fails_before_model_or_lmdb_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            with (
                mock.patch.object(
                    MODULE,
                    "AudioFeatureExtractor",
                    side_effect=AssertionError("must not load model"),
                ),
                mock.patch.object(
                    MODULE.lmdb,
                    "open",
                    side_effect=AssertionError("must not open LMDB"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "no usable SHOW"):
                    MODULE.create_lmdb_show(
                        data_folder=str(data_root),
                        lmdb_path=str(data_root / "output.lmdb"),
                        seq_size=256,
                        stride_size=20,
                        speakers=["oliver"],
                    )

    def test_full_clip_encodes_once_then_slices_motion_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            _make_show_clip(data_root, num_frames=300)
            raw_audio = np.zeros(160000, dtype=np.float32)
            metadata, environment, extractor = self._run_etl(
                data_root,
                raw_audio,
                feature_scope="full_clip",
            )

        self.assertEqual(extractor.calls, [(160000, 300)])
        self.assertEqual(metadata["feature_scope"], "full_clip")
        self.assertEqual(metadata["output_sample_count"], 3)
        self.assertEqual(environment.values[b"__len__"], b"3")
        stored_metadata = json.loads(
            environment.values[b"__meta__"].decode("utf-8")
        )
        self.assertEqual(stored_metadata["feature_scope"], "full_clip")
        self.assertEqual(stored_metadata["schema_version"], 2)
        self.assertEqual(
            stored_metadata["stats"]["processing"]["windows_written"],
            3,
        )
        for index, expected_start in enumerate((0, 20, 40)):
            record = pickle.loads(
                environment.values[f"{index:010d}".encode("ascii")]
            )
            self.assertEqual(record["huberts"].shape, (1, 256, 4))
            self.assertEqual(record["huberts"][0, 0, 0], expected_start)
            self.assertEqual(record["face_huberts"].shape, (1, 268, 4))
            crop_start = record["face_hubert_crop_start"]
            valid_length = record["face_hubert_valid_length"]
            np.testing.assert_array_equal(
                record["face_huberts"][
                    :, crop_start:crop_start + 256, :
                ],
                record["huberts"],
            )
            self.assertFalse(
                np.any(record["face_huberts"][:, valid_length:, :])
            )
        first = pickle.loads(environment.values[b"0000000000"])
        middle = pickle.loads(environment.values[b"0000000001"])
        last = pickle.loads(environment.values[b"0000000002"])
        self.assertEqual(
            (
                first["face_hubert_valid_length"],
                first["face_hubert_crop_start"],
            ),
            (262, 0),
        )
        self.assertEqual(
            (
                middle["face_hubert_valid_length"],
                middle["face_hubert_crop_start"],
            ),
            (268, 6),
        )
        self.assertEqual(
            (
                last["face_hubert_valid_length"],
                last["face_hubert_crop_start"],
            ),
            (266, 6),
        )

    def test_floor_audio_length_keeps_an_exact_256_frame_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            _make_show_clip(data_root, num_frames=256)
            raw_audio = np.zeros(
                256 * MODULE.AUDIO_SR // MODULE.POSE_FPS,
                dtype=np.float32,
            )
            metadata, environment, extractor = self._run_etl(
                data_root,
                raw_audio,
                feature_scope="full_clip",
            )

        self.assertEqual(metadata["output_sample_count"], 1)
        self.assertIn(b"0000000000", environment.values)
        self.assertEqual(extractor.calls, [(len(raw_audio), 256)])
        record = pickle.loads(environment.values[b"0000000000"])
        self.assertEqual(record["face_huberts"].shape, (1, 268, 4))
        self.assertEqual(record["face_hubert_valid_length"], 256)
        self.assertEqual(record["face_hubert_crop_start"], 0)

    def test_explicit_window_mode_preserves_per_window_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            _make_show_clip(data_root, num_frames=300)
            raw_audio = np.zeros(160000, dtype=np.float32)
            metadata, environment, extractor = self._run_etl(
                data_root,
                raw_audio,
                feature_scope="window",
            )

        self.assertEqual(
            extractor.calls,
            [
                (256 * 16000 // 30, 256),
                (256 * 16000 // 30, 256),
                (256 * 16000 // 30, 256),
            ],
        )
        self.assertEqual(metadata["feature_scope"], "window")
        self.assertEqual(metadata["face_context_frames"], 0)
        self.assertEqual(metadata["face_context_protocol"], "none")
        self.assertEqual(metadata["output_sample_count"], 3)
        record = pickle.loads(environment.values[b"0000000000"])
        self.assertNotIn("face_huberts", record)

    def test_zero_output_from_short_training_clip_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            _make_show_clip(data_root, num_frames=100)
            raw_audio = np.zeros(100 * 16000 // 30, dtype=np.float32)
            model_dir = _make_local_wavlm(data_root)
            environment = _FakeEnvironment()
            with (
                mock.patch.object(
                    MODULE,
                    "AudioFeatureExtractor",
                    _FakeAudioFeatureExtractor,
                ),
                mock.patch.object(
                    MODULE.lmdb,
                    "open",
                    return_value=environment,
                ),
                mock.patch.object(
                    MODULE.librosa,
                    "load",
                    return_value=(raw_audio, MODULE.AUDIO_SR),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "produced zero LMDB records",
                ):
                    MODULE.create_lmdb_show(
                        data_folder=str(data_root),
                        lmdb_path=str(data_root / "output.lmdb"),
                        seq_size=256,
                        stride_size=20,
                        local_rotation=True,
                        split="train",
                        speakers=["oliver"],
                        model_id=str(model_dir),
                    )

        self.assertEqual(_FakeAudioFeatureExtractor.instances[-1].calls, [])
        self.assertNotIn(b"__len__", environment.values)
        self.assertNotIn(b"__meta__", environment.values)
        self.assertTrue(environment.closed)

    def test_duration_mismatch_above_threshold_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            _make_show_clip(data_root, num_frames=300)
            raw_audio = np.zeros(306 * 16000 // 30, dtype=np.float32)
            model_dir = _make_local_wavlm(data_root)
            environment = _FakeEnvironment()
            with (
                mock.patch.object(
                    MODULE,
                    "AudioFeatureExtractor",
                    _FakeAudioFeatureExtractor,
                ),
                mock.patch.object(
                    MODULE.lmdb,
                    "open",
                    return_value=environment,
                ),
                mock.patch.object(
                    MODULE.librosa,
                    "load",
                    return_value=(raw_audio, MODULE.AUDIO_SR),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "duration mismatch exceeds threshold",
                ):
                    MODULE.create_lmdb_show(
                        data_folder=str(data_root),
                        lmdb_path=str(data_root / "output.lmdb"),
                        seq_size=256,
                        stride_size=20,
                        local_rotation=True,
                        split="train",
                        speakers=["oliver"],
                        model_id=str(model_dir),
                        max_av_duration_delta_frames=3,
                    )

        self.assertEqual(_FakeAudioFeatureExtractor.instances[-1].calls, [])
        self.assertNotIn(b"__meta__", environment.values)
        self.assertTrue(environment.closed)

    def test_duration_mismatch_equal_to_threshold_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            _make_show_clip(data_root, num_frames=300)
            raw_audio = np.zeros(
                303 * MODULE.AUDIO_SR // MODULE.POSE_FPS,
                dtype=np.float32,
            )
            metadata, _, extractor = self._run_etl(
                data_root,
                raw_audio,
                max_av_duration_delta_frames=3,
                duration_mismatch_policy="skip",
            )

        processing = metadata["stats"]["processing"]
        self.assertEqual(processing["duration_mismatch_skipped_count"], 0)
        self.assertEqual(processing["duration_mismatch_skip_rate"], 0.0)
        self.assertEqual(extractor.calls, [(len(raw_audio), 300)])

    def test_duration_mismatch_skip_writes_valid_clip_and_reports_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            _make_show_clip(
                data_root,
                num_frames=62,
                sequence_id="bad_sequence",
            )
            _make_show_clip(
                data_root,
                num_frames=300,
                sequence_id="good_sequence",
            )
            raw_audio = np.zeros(
                300 * MODULE.AUDIO_SR // MODULE.POSE_FPS,
                dtype=np.float32,
            )
            model_dir = _make_local_wavlm(data_root)
            environment = _FakeEnvironment()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    MODULE,
                    "AudioFeatureExtractor",
                    _FakeAudioFeatureExtractor,
                ),
                mock.patch.object(
                    MODULE.lmdb,
                    "open",
                    return_value=environment,
                ),
                mock.patch.object(
                    MODULE.librosa,
                    "load",
                    return_value=(raw_audio, MODULE.AUDIO_SR),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                metadata = MODULE.create_lmdb_show(
                    data_folder=str(data_root),
                    lmdb_path=str(data_root / "output.lmdb"),
                    seq_size=256,
                    stride_size=20,
                    local_rotation=True,
                    split="train",
                    speakers=["oliver"],
                    model_id=str(model_dir),
                    feature_scope="full_clip",
                    max_av_duration_delta_frames=3,
                    duration_mismatch_policy="skip",
                )

        extractor = _FakeAudioFeatureExtractor.instances[-1]
        self.assertEqual(extractor.calls, [(len(raw_audio), 300)])
        self.assertEqual(metadata["duration_mismatch_policy"], "skip")
        self.assertEqual(metadata["output_sample_count"], 3)
        self.assertEqual(environment.values[b"__len__"], b"3")
        stored_metadata = json.loads(
            environment.values[b"__meta__"].decode("utf-8")
        )
        self.assertEqual(
            stored_metadata["duration_mismatch_policy"],
            "skip",
        )
        self.assertEqual(
            stored_metadata["max_av_duration_delta_frames"],
            3.0,
        )
        processing = metadata["stats"]["processing"]
        self.assertEqual(processing["clips_seen"], 2)
        self.assertEqual(processing["duration_checked_clips"], 2)
        self.assertEqual(processing["clips_validated"], 1)
        self.assertEqual(processing["clips_written"], 1)
        self.assertEqual(
            processing["clips_skipped_by_reason"]["duration_mismatch"],
            1,
        )
        self.assertEqual(processing["duration_mismatch_skipped_count"], 1)
        self.assertEqual(processing["duration_mismatch_skip_rate"], 0.5)
        self.assertEqual(processing["duration_delta_frames_abs_sum"], 238.0)
        self.assertEqual(processing["duration_delta_frames_abs_max"], 238.0)
        self.assertEqual(processing["duration_delta_frames_abs_mean"], 119.0)
        self.assertEqual(
            processing["duration_mismatch_skipped_clips"],
            [
                {
                    "clip": "oliver/bad_sequence",
                    "motion_frames": 62,
                    "audio_samples": len(raw_audio),
                    "audio_frames": 300.0,
                    "delta_frames": 238.0,
                }
            ],
        )
        self.assertEqual(
            stored_metadata["stats"]["processing"][
                "duration_mismatch_skipped_clips"
            ],
            processing["duration_mismatch_skipped_clips"],
        )
        for index in range(3):
            record = pickle.loads(
                environment.values[f"{index:010d}".encode("ascii")]
            )
            self.assertEqual(record["ori_vid"], "oliver__good_sequence")
        self.assertIn(
            "skipped_clips=1, checked_clips=2, skip_rate=50.000000%",
            stdout.getvalue(),
        )
        self.assertIn(
            "clip=oliver/bad_sequence",
            stderr.getvalue(),
        )

    def test_duration_mismatch_skip_all_clips_still_fails_zero_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            _make_show_clip(data_root, num_frames=62)
            raw_audio = np.zeros(
                300 * MODULE.AUDIO_SR // MODULE.POSE_FPS,
                dtype=np.float32,
            )
            model_dir = _make_local_wavlm(data_root)
            environment = _FakeEnvironment()
            with (
                mock.patch.object(
                    MODULE,
                    "AudioFeatureExtractor",
                    _FakeAudioFeatureExtractor,
                ),
                mock.patch.object(
                    MODULE.lmdb,
                    "open",
                    return_value=environment,
                ),
                mock.patch.object(
                    MODULE.librosa,
                    "load",
                    return_value=(raw_audio, MODULE.AUDIO_SR),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "'duration_mismatch_skipped_count': 1",
                ),
            ):
                MODULE.create_lmdb_show(
                    data_folder=str(data_root),
                    lmdb_path=str(data_root / "output.lmdb"),
                    seq_size=64,
                    stride_size=20,
                    local_rotation=True,
                    split="train",
                    speakers=["oliver"],
                    model_id=str(model_dir),
                    max_av_duration_delta_frames=3,
                    duration_mismatch_policy="skip",
                )

        self.assertEqual(_FakeAudioFeatureExtractor.instances[-1].calls, [])
        self.assertNotIn(b"__len__", environment.values)
        self.assertNotIn(b"__meta__", environment.values)
        self.assertTrue(environment.closed)

    def test_duration_mismatch_skip_requires_a_threshold(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires max_av_duration_delta_frames",
        ):
            MODULE.create_lmdb_show(
                data_folder="unused",
                lmdb_path="unused",
                seq_size=64,
                stride_size=20,
                speakers=["oliver"],
                duration_mismatch_policy="skip",
            )


if __name__ == "__main__":
    unittest.main()
