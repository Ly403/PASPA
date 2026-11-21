import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "configure_face.py"
)


class ConfigureFaceTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.upstream = self.root / "upstream"
        (self.upstream / "config").mkdir(parents=True)
        self.data = self.root / "data"
        self.data.mkdir()
        base = {
            "Data": {"data_root": "old", "pklname": "old.pkl"},
            "Model": {"encoder_choice": "faceformer"},
            "DataLoader": {"batch_size": 1, "num_workers": 0},
            "Train": {"epochs": 100},
            "Log": {"save_every": 50, "print_every": 1000},
        }
        (self.upstream / "config" / "face.json").write_text(
            json.dumps(base), encoding="utf-8"
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def make_model(self, do_normalize=True, feat_extract_norm="layer"):
        model = self.root / "wavlm"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "wavlm",
                    "hidden_size": 1024,
                    "feat_extract_norm": feat_extract_norm,
                }
            ),
            encoding="utf-8",
        )
        (model / "pytorch_model.bin").write_bytes(b"test")
        (model / "preprocessor_config.json").write_text(
            json.dumps({"do_normalize": do_normalize}), encoding="utf-8"
        )
        return model

    def run_config(self, *extra):
        output = self.root / "face-output.json"
        command = [
            sys.executable,
            str(SCRIPT),
            "--upstream-dir",
            str(self.upstream),
            "--data-root",
            str(self.data),
            "--output",
            str(output),
            "--cache-tag",
            "test",
            "--epochs",
            "1",
            "--save-every",
            "1",
            "--print-every",
            "1",
            *extra,
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        return result, output

    def test_default_preserves_pinned_wav2vec2_path(self):
        result, output = self.run_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(output.read_text(encoding="utf-8"))
        model = config["Model"]
        self.assertEqual(model["encoder_choice"], "faceformer")
        self.assertEqual(model["audio_encoder"], "wav2vec2")
        self.assertIsNone(model["audio_model_path"])
        self.assertEqual(
            model["audio_model_source"], "facebook/wav2vec2-base-960h"
        )
        self.assertEqual(
            model["audio_model_revision"],
            "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
        )
        self.assertFalse(model["freeze_audio_encoder"])
        self.assertIsNone(model["audio_config_sha256"])
        self.assertIsNone(model["audio_preprocessor_sha256"])
        self.assertIsNone(model["audio_model_files_sha256"])
        data_loader = config["DataLoader"]
        self.assertEqual(data_loader["batch_size"], 1)
        self.assertFalse(data_loader["variable_length_batching"])
        self.assertIsNone(data_loader["length_bucket"])
        self.assertTrue(data_loader["drop_last"])
        self.assertEqual(config["Reproduction"], {"face_loss": "legacy"})
        self.assertNotIn("Resume", config)
        self.assertEqual(
            config["Distributed"],
            {
                "requested_world_size": 1,
                "effective_global_batch_size": 1,
            },
        )

    def test_records_eight_gpu_variable_length_batching(self):
        wavlm = self.make_model()
        result, output = self.run_config(
            "--epochs",
            "1000",
            "--batch-size",
            "32",
            "--world-size",
            "8",
            "--variable-length-batching",
            "--length-bucket",
            "8",
            "--no-drop-last",
            "--audio-encoder",
            "wavlm",
            "--audio-model-path",
            str(wavlm),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(config["Train"]["epochs"], 1000)
        self.assertEqual(
            config["DataLoader"],
            {
                "batch_size": 32,
                "num_workers": 0,
                "variable_length_batching": True,
                "length_bucket": 8,
                "drop_last": False,
            },
        )
        self.assertEqual(
            config["Distributed"],
            {
                "requested_world_size": 8,
                "effective_global_batch_size": 256,
            },
        )

    def test_distributed_batching_arguments_must_be_positive(self):
        invalid_arguments = (
            ("--batch-size", "0"),
            ("--world-size", "0"),
            (
                "--variable-length-batching",
                "--length-bucket",
                "0",
            ),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                result, _ = self.run_config(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be positive", result.stderr)

    def test_large_batch_requires_variable_length_batching(self):
        result, _ = self.run_config("--batch-size", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--variable-length-batching", result.stderr)

        result, _ = self.run_config("--world-size", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--variable-length-batching", result.stderr)

        result, _ = self.run_config("--length-bucket", "8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--variable-length-batching", result.stderr)

        result, _ = self.run_config("--variable-length-batching")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--length-bucket", result.stderr)

    def test_variable_length_batching_requires_padding_safe_wavlm(self):
        result, _ = self.run_config(
            "--variable-length-batching",
            "--length-bucket",
            "8",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--audio-encoder=wavlm", result.stderr)

        wavlm = self.make_model(feat_extract_norm="group")
        result, _ = self.run_config(
            "--variable-length-batching",
            "--length-bucket",
            "8",
            "--audio-encoder",
            "wavlm",
            "--audio-model-path",
            str(wavlm),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("feat_extract_norm='layer'", result.stderr)

    def test_wavlm_records_local_model_and_preprocessing(self):
        wavlm = self.make_model(do_normalize=True)
        result, output = self.run_config(
            "--audio-encoder",
            "wavlm",
            "--audio-model-path",
            str(wavlm),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        model = json.loads(output.read_text(encoding="utf-8"))["Model"]
        self.assertEqual(model["audio_encoder"], "wavlm")
        self.assertEqual(model["audio_model_path"], str(wavlm.resolve()))
        self.assertEqual(model["audio_model_source"], str(wavlm.resolve()))
        self.assertIsNone(model["audio_model_revision"])
        self.assertTrue(model["audio_do_normalize"])
        self.assertEqual(model["audio_hidden_size"], 1024)
        self.assertTrue(model["freeze_audio_encoder"])
        self.assertEqual(len(model["audio_config_sha256"]), 64)
        self.assertEqual(len(model["audio_preprocessor_sha256"]), 64)
        self.assertEqual(
            set(model["audio_model_files_sha256"]),
            {
                "config.json",
                "preprocessor_config.json",
                "pytorch_model.bin",
            },
        )
        self.assertTrue(
            all(
                len(digest) == 64
                for digest in model[
                    "audio_model_files_sha256"
                ].values()
            )
        )

    def test_wavlm_requires_a_complete_local_snapshot(self):
        result, _ = self.run_config("--audio-encoder", "wavlm")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--audio-model-path is required", result.stderr)

        wavlm = self.make_model()
        (wavlm / "preprocessor_config.json").unlink()
        result, _ = self.run_config(
            "--audio-encoder",
            "wavlm",
            "--audio-model-path",
            str(wavlm),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preprocessor_config.json", result.stderr)

    def test_training_args_file_is_not_accepted_as_model_weights(self):
        model = self.root / "incomplete-wavlm"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "wavlm", "hidden_size": 1024}),
            encoding="utf-8",
        )
        (model / "preprocessor_config.json").write_text(
            json.dumps({"do_normalize": True}), encoding="utf-8"
        )
        (model / "training_args.bin").write_bytes(b"not model weights")

        result, _ = self.run_config(
            "--audio-encoder",
            "wavlm",
            "--audio-model-path",
            str(model),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no Hugging Face model weights", result.stderr)

    def test_records_corrected_loss_and_resume_lineage(self):
        checkpoint = self.root / "ckpt-49.pth"
        checkpoint.touch()
        result, output = self.run_config(
            "--face-loss",
            "corrected",
            "--resume-from",
            str(checkpoint),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            config["Reproduction"],
            {"face_loss": "corrected"},
        )
        self.assertEqual(
            config["Resume"],
            {
                "checkpoint": str(checkpoint.resolve()),
                "boundary": "end_of_epoch",
            },
        )


if __name__ == "__main__":
    unittest.main()
