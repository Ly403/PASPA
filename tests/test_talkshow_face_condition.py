from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "diffusion/model/audio_encoder/talkshow_face.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "talkshow_face_condition_under_test",
    MODULE_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
TalkSHOWFaceConditionModel = MODULE.TalkSHOWFaceConditionModel
_canonical_json_sha256 = MODULE._canonical_json_sha256


def _make_model(**overrides) -> TalkSHOWFaceConditionModel:
    params = dict(
        input_feature_dim=8,
        hidden_dim=16,
        output_feat_dim=5,
        jaw_dim=3,
        num_person=4,
        speaker_id_offset=0,
        identity_dim=6,
    )
    params.update(overrides)
    return TalkSHOWFaceConditionModel(**params)


def _raw_talkshow_state(
    model: TalkSHOWFaceConditionModel,
    *,
    prefix: str = "",
) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict(
        (prefix + key, value.detach().clone())
        for key, value in model.state_dict().items()
    )
    state[prefix + "audio_encoder.fake_weight"] = torch.ones(1)
    state[prefix + "audio_middle.grus.weight_ih_l0"] = torch.ones(1)
    return state


class TalkSHOWFaceConditionModelTest(unittest.TestCase):
    def test_default_state_matches_talkshow_wavlm_head_contract(self) -> None:
        model = TalkSHOWFaceConditionModel()
        state = model.state_dict()
        self.assertEqual(model.supported_main_trainers, ("bld",))
        self.assertEqual(model.required_mixed_precision, "no")
        self.assertTrue(model.handles_global_speaker_ids)
        self.assertEqual(model.speaker_id_offset, 0)
        self.assertEqual(model.expected_lmdb_pid_offset, 1)
        self.assertEqual(
            model.expected_speaker_ids,
            {
                "oliver": 21,
                "chemistry": 22,
                "seth": 23,
                "conan": 24,
            },
        )
        self.assertEqual(len(state), 46)
        self.assertEqual(
            tuple(state["audio_feature_map.weight"].shape),
            (256, 1024),
        )
        self.assertEqual(
            tuple(state["audio_middle.id_mlp.weight"].shape),
            (64, 4, 1),
        )
        self.assertEqual(
            tuple(
                state[
                    "audio_middle.first_net.conv_layers.0."
                    "residual_layer.0.weight"
                ].shape
            ),
            (256, 320, 3),
        )
        self.assertEqual(
            tuple(state["final_out.0.weight"].shape),
            (3, 64, 1),
        )
        self.assertEqual(
            tuple(state["final_out.1.weight"].shape),
            (100, 256, 1),
        )
        self.assertFalse(
            any(key.startswith("audio_middle.grus.") for key in state)
        )

    def test_runtime_data_contract_rejects_window_local_overrides(self) -> None:
        model = _make_model()
        data_config = {
            "audio_sr": 16000,
            "audio_fps": 16000,
            "pose_fps": 30,
            "pose_length": 256,
            "joints": 55,
            "ori_joints": "beat_smplx_joints",
            "load_audio_feat": True,
        }
        model.validate_runtime_data_config(data_config)

        for field, invalid in (
            ("audio_sr", 22050),
            ("pose_fps", 25),
            ("pose_length", 128),
            ("load_audio_feat", False),
        ):
            with self.subTest(field=field):
                overridden = dict(data_config)
                overridden[field] = invalid
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"data.{field}",
                ):
                    model.validate_runtime_data_config(overridden)

    def test_runtime_sequence_and_block_contract_follow_profile(self) -> None:
        model = _make_model(
            expected_pose_length=128,
            expected_block_size=32,
        )
        data_config = {
            "audio_sr": 16000,
            "audio_fps": 16000,
            "pose_fps": 30,
            "pose_length": 128,
            "joints": 55,
            "ori_joints": "beat_smplx_joints",
            "load_audio_feat": True,
        }
        model.validate_runtime_data_config(data_config)
        model.validate_runtime_training_config(
            {"num_frame_per_block": 32}
        )

        with self.assertRaisesRegex(RuntimeError, "data.pose_length"):
            model.validate_runtime_data_config(
                {**data_config, "pose_length": 256}
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "train.num_frame_per_block",
        ):
            model.validate_runtime_training_config(
                {"num_frame_per_block": 64}
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            _make_model(
                expected_pose_length=65,
                expected_block_size=16,
            )

    def test_forward_contract_and_full_face_access(self) -> None:
        torch.manual_seed(7)
        model = _make_model().eval()
        features = torch.randn(2, 11, 8)
        pid = torch.tensor([0, 3])

        with torch.no_grad():
            face = model.forward_face(features, pid)
            expression = model(features, pid=pid)

        self.assertEqual(face.shape, (2, 11, 8))
        self.assertEqual(expression.shape, (2, 11, 5))
        torch.testing.assert_close(expression, face[..., 3:])

    def test_runtime_pid_contract_rejects_legacy_or_raw_ids(self) -> None:
        model = _make_model().eval()
        features = torch.randn(1, 4, 8)

        for invalid_pid in (-1, 4, 21, 24, 31, 33):
            with self.subTest(pid=invalid_pid):
                with self.assertRaisesRegex(
                    ValueError,
                    "runtime network IDs 0..3",
                ):
                    model(features, pid=torch.tensor([invalid_pid]))

    def test_contextual_windows_equal_one_whole_clip_forward(self) -> None:
        torch.manual_seed(17)
        model = _make_model().eval()
        full_features = torch.randn(1, 300, 8)
        starts = (0, 20, 44)
        radius = model.context_radius_frames
        contexts = []
        valid_lengths = []
        crop_starts = []
        for start in starts:
            context_start = max(0, start - radius)
            context_end = min(300, start + 256 + radius)
            valid = full_features[:, context_start:context_end]
            carrier = torch.zeros(1, 256 + 2 * radius, 8)
            carrier[:, : valid.shape[1]] = valid
            contexts.append(carrier)
            valid_lengths.append(valid.shape[1])
            crop_starts.append(start - context_start)

        with torch.no_grad():
            whole_expression = model(
                full_features,
                pid=torch.tensor([0]),
            )
            contextual_expression = model.forward_contextual(
                torch.cat(contexts, dim=0),
                pid=torch.tensor([0, 0, 0]),
                valid_lengths=torch.tensor(valid_lengths),
                crop_starts=torch.tensor(crop_starts),
                target_length=256,
            )

        expected = torch.cat(
            [
                whole_expression[:, start:start + 256]
                for start in starts
            ],
            dim=0,
        )
        torch.testing.assert_close(
            contextual_expression,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_runtime_speaker_ids_are_the_four_local_classes(self) -> None:
        model = _make_model()
        mapped = model._local_speaker_ids(
            torch.tensor([0, 1, 2, 3]),
            batch_size=4,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(mapped, torch.tensor([0, 1, 2, 3]))

        for invalid in (-1, 4, 21, 24):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "0..3"):
                    model(
                        torch.randn(1, 2, 8),
                        pid=torch.tensor([invalid]),
                    )

    def test_raw_lmdb_mapping_must_match_declared_offset(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "canonical source mapping",
        ):
            _make_model(
                expected_lmdb_pid_offset=1,
                expected_speaker_ids={
                    "oliver": 21,
                    "chemistry": 23,
                    "seth": 22,
                    "conan": 24,
                },
            )

    def test_runtime_local_ids_preserve_legacy_face_weight_semantics(
        self,
    ) -> None:
        """Changing the external ID namespace must not change face outputs."""

        torch.manual_seed(23)
        legacy = _make_model(speaker_id_offset=20).eval()
        local = _make_model(speaker_id_offset=0).eval()
        local.load_state_dict(legacy.state_dict(), strict=True)
        features = torch.randn(4, 9, 8)

        with torch.no_grad():
            legacy_face = legacy.forward_face(
                features,
                pid=torch.tensor([20, 21, 22, 23]),
            )
            local_face = local.forward_face(
                features,
                pid=torch.tensor([0, 1, 2, 3]),
            )

        torch.testing.assert_close(local_face, legacy_face)

    def test_rejects_invalid_input_contract(self) -> None:
        model = _make_model()
        with self.assertRaisesRegex(ValueError, r"\[B, T, C\]"):
            model(torch.randn(2, 8), pid=torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "hidden size 8"):
            model(torch.randn(2, 3, 7), pid=torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "one speaker ID per sample"):
            model(torch.randn(2, 3, 8), pid=torch.tensor([0]))
        with self.assertRaisesRegex(TypeError, "requires float32"):
            model(
                torch.randn(2, 3, 8, dtype=torch.float64),
                pid=torch.tensor([0, 1]),
            )

    def test_loads_real_nested_layout_strictly(self) -> None:
        torch.manual_seed(11)
        source = _make_model()
        checkpoint = {
            "generator": {
                "generator": _raw_talkshow_state(source),
                "generator_optim": {},
                "discriminator": None,
            },
            "epoch": 999,
            "global_steps": 123,
        }

        target = _make_model()
        report = target.load_pretrained_checkpoint(checkpoint)

        self.assertEqual(report["layout"], "generator.generator")
        self.assertEqual(report["loaded_keys"], 46)
        self.assertEqual(
            report["ignored_prefix_counts"],
            {"audio_encoder.": 1, "audio_middle.grus.": 1},
        )
        for key, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], expected)

    def test_loads_bare_and_module_prefixed_states(self) -> None:
        source = _make_model()
        for prefix in ("", "module.", "module.generator."):
            with self.subTest(prefix=prefix):
                target = _make_model()
                report = target.load_pretrained_checkpoint(
                    _raw_talkshow_state(source, prefix=prefix)
                )
                self.assertEqual(report["layout"], "<root>")
                for key, expected in source.state_dict().items():
                    torch.testing.assert_close(
                        target.state_dict()[key],
                        expected,
                    )

    def test_loads_checkpoint_file(self) -> None:
        source = _make_model()
        payload = {"generator": {"generator": _raw_talkshow_state(source)}}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "ckpt-999.pth"
            torch.save(payload, checkpoint)
            target = _make_model()
            report = target.load_pretrained_checkpoint(checkpoint)

        self.assertEqual(report["source"], str(checkpoint))
        self.assertEqual(report["layout"], "generator.generator")

    def test_schema_one_checkpoint_remains_weight_compatible(self) -> None:
        source = _make_model()
        contract = {
            "schema_version": 1,
            "trainer": "talkshow_face_large_batch",
            "audio_encoder": "wavlm",
            "audio_hidden_size": 8,
            "audio_config_sha256": "1" * 64,
            "audio_preprocessor_sha256": "2" * 64,
            "audio_do_normalize": True,
            "freeze_audio_encoder": True,
            "num_classes": 4,
            # This legacy field is a dataset traversal list, not class order.
            "speakers": ["oliver", "seth", "conan", "chemistry"],
            "face_loss": "corrected",
        }
        payload = {
            "generator": {"generator": _raw_talkshow_state(source)},
            "resume_contract": contract,
        }

        target = _make_model(required_face_loss="corrected")
        with self.assertWarns(RuntimeWarning):
            report = target.load_pretrained_checkpoint(payload)

        self.assertEqual(report["audio_contract"], "validated")
        for key, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], expected)

    def test_missing_unexpected_and_wrong_shape_fail_closed(self) -> None:
        source = _make_model()

        missing = _raw_talkshow_state(source)
        del missing["final_out.1.weight"]
        with self.assertRaisesRegex(RuntimeError, "missing keys"):
            _make_model().load_pretrained_checkpoint(missing)

        unexpected = _raw_talkshow_state(source)
        unexpected["not_a_talkshow_head.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected keys"):
            _make_model().load_pretrained_checkpoint(unexpected)

        wrong_shape = _raw_talkshow_state(source)
        wrong_shape["audio_feature_map.weight"] = torch.randn(16, 9)
        with self.assertRaisesRegex(RuntimeError, "tensor shapes"):
            _make_model().load_pretrained_checkpoint(wrong_shape)

    def test_non_fp32_checkpoint_is_rejected(self) -> None:
        source = _make_model()
        state = _raw_talkshow_state(source)
        state["audio_feature_map.weight"] = state[
            "audio_feature_map.weight"
        ].to(torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "must remain float32"):
            _make_model().load_pretrained_checkpoint(state)

    def test_non_finite_checkpoint_is_rejected(self) -> None:
        source = _make_model()
        state = _raw_talkshow_state(source)
        state["audio_feature_map.bias"][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
            _make_model().load_pretrained_checkpoint(state)

    def test_modern_checkpoint_matches_runtime_wavlm_and_lmdb(self) -> None:
        source = _make_model()
        audio_config = {
            "model_type": "wavlm",
            "hidden_size": 8,
            "feat_extract_norm": "layer",
            "add_adapter": False,
        }
        preprocessor_config = {"do_normalize": True}
        with tempfile.TemporaryDirectory() as temporary:
            wavlm_dir = Path(temporary)
            (wavlm_dir / "config.json").write_text(
                json.dumps(audio_config),
                encoding="utf-8",
            )
            (wavlm_dir / "preprocessor_config.json").write_text(
                json.dumps(preprocessor_config),
                encoding="utf-8",
            )
            model = TalkSHOWFaceConditionModel(
                input_feature_dim=8,
                hidden_dim=16,
                output_feat_dim=5,
                jaw_dim=3,
                num_person=4,
                speaker_id_offset=0,
                identity_dim=6,
                wavlm_model_path=wavlm_dir,
                require_audio_contract=True,
                required_face_loss="corrected",
            )
            contract = {
                "schema_version": 2,
                "trainer": "talkshow_face_large_batch",
                "audio_encoder": "wavlm",
                "audio_hidden_size": 8,
                "audio_config_sha256": _canonical_json_sha256(audio_config),
                "audio_preprocessor_sha256": _canonical_json_sha256(
                    preprocessor_config
                ),
                "audio_do_normalize": True,
                "freeze_audio_encoder": True,
                "num_classes": 4,
                "speakers": ["oliver", "seth", "conan", "chemistry"],
                "face_loss": "corrected",
                "audio_model_files_sha256": {
                    "config.json": "1" * 64,
                    "preprocessor_config.json": "2" * 64,
                    "model.safetensors": "3" * 64,
                },
            }
            payload = {
                "generator": {"generator": _raw_talkshow_state(source)},
                "resume_contract": contract,
            }
            report = model.load_pretrained_checkpoint(payload)
            self.assertEqual(report["audio_contract"], "validated")

            metadata = {
                "split": "train",
                "feature_scope": "full_clip",
                "face_context_frames": 6,
                "face_context_protocol": "source_clip_valid_slice_v1",
                "audio_feature_dim": 8,
                "pid_offset": 1,
                "speaker_ids": {
                    "oliver": 21,
                    "chemistry": 22,
                    "seth": 23,
                    "conan": 24,
                },
                "audio_config_sha256": contract["audio_config_sha256"],
                "audio_preprocessor_sha256": contract[
                    "audio_preprocessor_sha256"
                ],
                "audio_do_normalize": True,
                "audio_model_files_sha256": dict(
                    contract["audio_model_files_sha256"]
                ),
            }
            model.validate_dataset_contract({"train": metadata})

            wrong_offset = dict(metadata)
            wrong_offset["pid_offset"] = 0
            with self.assertRaisesRegex(RuntimeError, "pid_offset"):
                model.validate_dataset_contract({"train": wrong_offset})

            wrong_mapping = dict(metadata)
            wrong_mapping["speaker_ids"] = {
                "oliver": 21,
                "chemistry": 23,
                "seth": 22,
                "conan": 24,
            }
            with self.assertRaisesRegex(RuntimeError, "speaker/PID mapping"):
                model.validate_dataset_contract({"train": wrong_mapping})

            metadata["split_view_summary_sha256"] = "1" * 64
            validation_metadata = dict(metadata)
            validation_metadata["split"] = "val"
            model.validate_dataset_contract(
                {
                    "train": metadata,
                    "validation": validation_metadata,
                }
            )
            test_as_validation_metadata = dict(validation_metadata)
            test_as_validation_metadata["split"] = "test"
            model.validate_dataset_contract(
                {
                    "train": metadata,
                    "validation": test_as_validation_metadata,
                }
            )
            with self.assertRaisesRegex(RuntimeError, "declares split"):
                model.validate_dataset_contract(
                    {
                        "train": test_as_validation_metadata,
                    }
                )
            validation_metadata["split_view_summary_sha256"] = "2" * 64
            with self.assertRaisesRegex(
                RuntimeError,
                "split_view_summary_sha256",
            ):
                model.validate_dataset_contract(
                    {
                        "train": metadata,
                        "validation": validation_metadata,
                    }
                )

            metadata["audio_preprocessor_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                RuntimeError,
                "audio_preprocessor_sha256",
            ):
                model.validate_dataset_contract({"train": metadata})

    def test_strict_audio_contract_rejects_legacy_checkpoint(self) -> None:
        audio_config = {
            "model_type": "wavlm",
            "hidden_size": 8,
            "feat_extract_norm": "layer",
            "add_adapter": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            wavlm_dir = Path(temporary)
            (wavlm_dir / "config.json").write_text(
                json.dumps(audio_config),
                encoding="utf-8",
            )
            (wavlm_dir / "preprocessor_config.json").write_text(
                json.dumps({"do_normalize": True}),
                encoding="utf-8",
            )
            model = TalkSHOWFaceConditionModel(
                input_feature_dim=8,
                hidden_dim=16,
                output_feat_dim=5,
                jaw_dim=3,
                num_person=4,
                speaker_id_offset=0,
                identity_dim=6,
                wavlm_model_path=wavlm_dir,
                require_audio_contract=True,
            )
            with self.assertRaisesRegex(RuntimeError, "no resume_contract"):
                model.load_pretrained_checkpoint(
                    _raw_talkshow_state(_make_model())
                )

    def test_runtime_audio_encoder_must_use_the_validated_wavlm_path(
        self,
    ) -> None:
        audio_config = {
            "model_type": "wavlm",
            "hidden_size": 8,
            "feat_extract_norm": "layer",
            "add_adapter": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wavlm_dir = root / "wavlm"
            wavlm_dir.mkdir()
            (wavlm_dir / "config.json").write_text(
                json.dumps(audio_config),
                encoding="utf-8",
            )
            (wavlm_dir / "preprocessor_config.json").write_text(
                json.dumps({"do_normalize": True}),
                encoding="utf-8",
            )
            model = TalkSHOWFaceConditionModel(
                input_feature_dim=8,
                hidden_dim=16,
                output_feat_dim=5,
                jaw_dim=3,
                num_person=4,
                speaker_id_offset=0,
                identity_dim=6,
                wavlm_model_path=wavlm_dir,
                require_audio_contract=True,
            )
            model.validate_runtime_audio_encoder_config(
                {
                    "name": "wavlm",
                    "local_model_path": str(wavlm_dir),
                }
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "trainer WavLM path differs",
            ):
                model.validate_runtime_audio_encoder_config(
                    {
                        "name": "wavlm",
                        "local_model_path": str(root / "other-wavlm"),
                    }
                )
            with self.assertRaisesRegex(
                RuntimeError,
                "audio.audio_encoder.name=wavlm",
            ):
                model.validate_runtime_audio_encoder_config(
                    {
                        "name": "hubert-large-ls960-ft",
                        "local_model_path": str(wavlm_dir),
                    }
                )
            with self.assertRaisesRegex(
                RuntimeError,
                "quantization=false",
            ):
                model.validate_runtime_audio_encoder_config(
                    {
                        "name": "wavlm",
                        "local_model_path": str(wavlm_dir),
                        "quantization": True,
                    }
                )


class TalkSHOWFaceConfigContractTest(unittest.TestCase):
    def test_release_config_uses_the_talkshow_face_adapter(self) -> None:
        root = REPO_ROOT
        talkshow = (
            root
            / "configs/model/show_causal_emage_deltanet_talkshow_face.yaml"
        ).read_text(encoding="utf-8")
        talkshow_data = (
            root
            / "configs/data/show_global_256_lmdb_talkshow_face.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "talkshow_face.TalkSHOWFaceConditionModel",
            talkshow,
        )
        backbone_section = talkshow.split("model_wrapper:", maxsplit=1)[0]
        self.assertIn("    num_person: 4", backbone_section)
        self.assertIn("speaker_id_offset: 0", talkshow)
        self.assertIn("expected_lmdb_pid_offset: 1", talkshow)
        for mapping_line in (
            "oliver: 21",
            "chemistry: 22",
            "seth: 23",
            "conan: 24",
        ):
            self.assertIn(mapping_line, talkshow)
        self.assertIn("expected_pose_length: 256", talkshow)
        self.assertIn("expected_block_size: 64", talkshow)
        self.assertIn("mixed_precision: 'no'", talkshow)
        self.assertIn(
            "rest_joint_position_path: ${oc.env:SHOW_REST_JOINT_PATH}",
            talkshow_data,
        )


if __name__ == "__main__":
    unittest.main()
