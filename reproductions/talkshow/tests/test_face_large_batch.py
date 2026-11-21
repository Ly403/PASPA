import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from reproductions.talkshow.scripts.face_resume import capture_rng_state
    from reproductions.talkshow.scripts.train_face_large_batch import (
        PaddedWavLMFaceGenerator,
        _build_resume_contract,
        _masked_face_loss,
        _masked_utterance_normalize,
        _save_checkpoint,
        _preserve_legacy_audio_contract_on_resume,
        _validate_face_loss_config,
    )


@unittest.skipUnless(torch is not None, "PyTorch is installed in the face environment")
class MaskedFaceTrainingTest(unittest.TestCase):
    def test_resume_contract_records_effective_training_options(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD(
            [parameter],
            lr=0.001,
            momentum=0.9,
        )
        file_hash_object = SimpleNamespace()
        file_hash_object.__dict__.update(
            {
                "config.json": "a" * 64,
                "preprocessor_config.json": "b" * 64,
                "pytorch_model.bin": "c" * 64,
            }
        )
        config = SimpleNamespace(
            DataLoader=SimpleNamespace(
                length_bucket=8,
                drop_last=False,
                num_workers=4,
            ),
            Model=SimpleNamespace(
                audio_encoder="wavlm",
                audio_hidden_size=1024,
                audio_do_normalize=True,
                freeze_audio_encoder=True,
                audio_model_files_sha256=file_hash_object,
            ),
            Train=SimpleNamespace(max_gradient_norm=5),
        )
        wrapper = SimpleNamespace(
            num_classes=4,
            expression=True,
            generator_optimizer=optimizer,
        )
        args = SimpleNamespace(
            seed=1,
            speakers=["oliver", "seth", "conan", "chemistry"],
        )

        with mock.patch.dict(
            os.environ,
            {"TALKSHOW_LEGACY_FACE_LOSS": "0"},
        ):
            contract = _build_resume_contract(config, wrapper, args)

        self.assertEqual(contract["learning_rate"], 0.001)
        self.assertEqual(contract["momentum"], 0.9)
        self.assertEqual(contract["face_loss"], "corrected")
        self.assertEqual(contract["length_bucket"], 8)
        self.assertFalse(contract["drop_last"])
        self.assertEqual(contract["schema_version"], 2)
        self.assertEqual(
            contract["audio_model_files_sha256"]["pytorch_model.bin"],
            "c" * 64,
        )
        legacy = _preserve_legacy_audio_contract_on_resume(
            {"resume_contract": {"schema_version": 1}},
            contract,
        )
        self.assertEqual(legacy["schema_version"], 1)
        self.assertNotIn("audio_model_files_sha256", legacy)
        no_contract = _preserve_legacy_audio_contract_on_resume(
            {},
            contract,
        )
        self.assertEqual(no_contract["schema_version"], 1)
        self.assertNotIn("audio_model_files_sha256", no_contract)

    def test_face_loss_config_must_match_the_active_environment(self):
        config = SimpleNamespace(
            Reproduction=SimpleNamespace(face_loss="corrected")
        )
        with mock.patch.dict(
            os.environ,
            {"TALKSHOW_LEGACY_FACE_LOSS": "0"},
        ):
            _validate_face_loss_config(config)
        with mock.patch.dict(
            os.environ,
            {"TALKSHOW_LEGACY_FACE_LOSS": "1"},
        ):
            with self.assertRaisesRegex(RuntimeError, "disagree"):
                _validate_face_loss_config(config)

    def test_padded_batch_matches_individual_valid_outputs(self):
        class Layer(torch.nn.Module):
            norm_type = "ln"

            def __init__(self, in_channels, out_channels, residual=False):
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=3, padding=1
                )
                self.norm = torch.nn.LayerNorm(out_channels)
                self.residual = (
                    torch.nn.Conv1d(
                        in_channels, out_channels, kernel_size=3, padding=1
                    )
                    if residual
                    else None
                )

            def forward(self, values):
                output = self.conv(values)
                output = self.norm(
                    output.transpose(1, 2)
                ).transpose(1, 2)
                if self.residual is not None:
                    output = output + self.residual(values)
                return torch.relu(output)

        class Encoder(torch.nn.Module):
            config = type("Config", (), {"feat_extract_norm": "layer"})()

            def forward(self, waveform, attention_mask):
                hidden = torch.stack((waveform, waveform * 2), dim=-1)
                return type("Output", (), {"last_hidden_state": hidden})()

            def _get_feat_extract_output_lengths(self, lengths):
                return lengths

        class Middle(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.identity = True
                self.dropout = torch.nn.Identity()
                self.id_mlp = torch.nn.Conv1d(2, 2, kernel_size=1)
                self.first_net = torch.nn.Module()
                self.first_net.conv_layers = torch.nn.Sequential(
                    Layer(6, 4, residual=True),
                    Layer(4, 4, residual=True),
                )

        class Generator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_encoder_name = "wavlm"
                self.freeze_audio_encoder = True
                self.audio_do_normalize = False
                self.audio_encoder = Encoder()
                self.audio_feature_map = torch.nn.Linear(2, 4)
                self.audio_middle = Middle()
                self.decoder = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(Layer(4, 3), Layer(3, 3)),
                        torch.nn.Sequential(Layer(4, 3), Layer(3, 3)),
                    ]
                )
                self.final_out = torch.nn.ModuleList(
                    [
                        torch.nn.Conv1d(3, 1, kernel_size=1),
                        torch.nn.Conv1d(3, 2, kernel_size=1),
                    ]
                )

        torch.manual_seed(7)
        adapter = PaddedWavLMFaceGenerator(Generator()).eval()
        waveform = torch.zeros((2, 1, 8), dtype=torch.float32)
        waveform[0, 0, :5] = torch.arange(1, 6, dtype=torch.float32)
        waveform[1, 0, :8] = torch.arange(11, 19, dtype=torch.float32)
        audio_lengths = torch.tensor([5, 8])
        audio_mask = torch.arange(8)[None] < audio_lengths[:, None]
        motion_lengths = torch.tensor([3, 5])
        motion_mask = torch.arange(5)[None] < motion_lengths[:, None]
        identity = torch.tensor([[1, 0], [0, 1]])
        target = torch.zeros((2, 5, 3), dtype=torch.float32)

        padded_prediction, _ = adapter(
            waveform,
            target,
            identity,
            audio_lengths,
            audio_mask.to(torch.long),
            motion_lengths,
            motion_mask,
        )

        for index, (audio_length, motion_length) in enumerate(
            zip(audio_lengths.tolist(), motion_lengths.tolist())
        ):
            individual_prediction, _ = adapter(
                waveform[index : index + 1, :, :audio_length],
                target[index : index + 1, :motion_length],
                identity[index : index + 1],
                torch.tensor([audio_length]),
                torch.ones((1, audio_length), dtype=torch.long),
                torch.tensor([motion_length]),
                torch.ones((1, motion_length), dtype=torch.bool),
            )
            torch.testing.assert_close(
                padded_prediction[index, :motion_length],
                individual_prediction[0],
                atol=1e-6,
                rtol=1e-6,
            )

    def test_rejects_padding_unsafe_wavlm_feature_norm(self):
        class Generator(torch.nn.Module):
            def __init__(self, norm):
                super().__init__()
                self.audio_encoder_name = "wavlm"
                self.freeze_audio_encoder = True
                self.audio_encoder = type(
                    "Encoder",
                    (),
                    {"config": type("Config", (), {"feat_extract_norm": norm})()},
                )()

        with self.assertRaisesRegex(RuntimeError, "feat_extract_norm='layer'"):
            PaddedWavLMFaceGenerator(Generator("group"))

        adapter = PaddedWavLMFaceGenerator(Generator("layer"))
        self.assertEqual(adapter.generator.audio_encoder_name, "wavlm")

    def test_normalization_ignores_and_clears_audio_padding(self):
        waveform = torch.tensor(
            [
                [1.0, 2.0, 3.0, 1000.0, 1000.0],
                [2.0, 4.0, 6.0, 8.0, 10.0],
            ],
            dtype=torch.float32,
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
            dtype=torch.long,
        )

        normalized = _masked_utterance_normalize(
            waveform, attention_mask
        )

        self.assertEqual(normalized.dtype, torch.float32)
        torch.testing.assert_close(normalized[0, 3:], torch.zeros(2))
        for index, valid_length in enumerate((3, 5)):
            valid = normalized[index, :valid_length]
            torch.testing.assert_close(
                valid.mean(), torch.tensor(0.0), atol=1e-6, rtol=0
            )
            torch.testing.assert_close(
                valid.var(unbiased=False),
                torch.tensor(1.0),
                atol=1e-5,
                rtol=0,
            )

    def test_loss_averages_valid_frames_per_clip(self):
        target = torch.zeros((2, 4, 103), dtype=torch.float32)
        prediction = torch.zeros_like(target)
        prediction[0, :2, :3] = 2.0
        prediction[0, :2, -100:] = 3.0
        prediction[0, 2:, :] = 10_000.0
        prediction[1, :4, :3] = 4.0
        prediction[1, :4, -100:] = 5.0
        motion_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )

        loss, losses = _masked_face_loss(
            prediction,
            target,
            motion_mask,
            jaw_width=3,
            expression=True,
        )

        # Equal clip weighting: jaw=(2+4)/2, expression=(3^2+5^2)/2.
        torch.testing.assert_close(losses["MSELoss"], torch.tensor(3.0))
        torch.testing.assert_close(losses["exp_loss"], torch.tensor(17.0))
        torch.testing.assert_close(loss, torch.tensor(20.0))

    def test_checkpoint_keeps_the_released_wrapper_state_shape(self):
        class Wrapper:
            def state_dict(self):
                return {
                    "generator": {"weight": torch.ones(1)},
                    "generator_optim": {
                        "state": {},
                        "param_groups": [{"params": [0]}],
                    },
                    "discriminator": None,
                    "discriminator_optim": None,
                }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ckpt-999.pth"
            _save_checkpoint(
                path,
                Wrapper(),
                epoch=999,
                global_steps=54_000,
                samples_seen=13_688_000,
                world_size=8,
                per_device_batch_size=32,
                steps_per_epoch=54,
                samples_per_epoch=13_688,
                dataset_fingerprint="dataset-sha256",
                resume_contract={"schema_version": 1, "seed": 1},
                rng_states=[capture_rng_state(None) for _ in range(8)],
            )
            payload = torch.load(
                path, map_location="cpu", weights_only=True
            )

        self.assertEqual(payload["epoch"], 999)
        self.assertEqual(payload["global_steps"], 54_000)
        self.assertEqual(payload["samples_seen"], 13_688_000)
        self.assertEqual(
            payload["distributed"],
            {
                "world_size": 8,
                "per_device_batch_size": 32,
                "global_batch_size": 256,
                "steps_per_epoch": 54,
                "samples_per_epoch": 13_688,
            },
        )
        self.assertIn("generator", payload["generator"])
        self.assertEqual(payload["dataset_fingerprint"], "dataset-sha256")
        self.assertEqual(
            payload["resume_contract"],
            {"schema_version": 1, "seed": 1},
        )
        self.assertEqual(len(payload["rng_states"]), 8)
        self.assertFalse(
            any(
                key.startswith("module.")
                for key in payload["generator"]["generator"]
            )
        )


if __name__ == "__main__":
    unittest.main()
