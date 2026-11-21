import json
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ModuleNotFoundError:
    np = torch = None

if torch is not None:
    from reproductions.talkshow.scripts.face_resume import (
        capture_rng_state,
        get_rank_rng_state,
        get_resume_progress,
        load_face_checkpoint,
        restore_rng_state,
        restore_wrapper_state,
        validate_distributed_resume,
    )
    from reproductions.talkshow.scripts.train_face_single_resume import (
        validate_single_epoch_progress,
        validate_previous_run_config,
    )
    from reproductions.talkshow.scripts.verify_face_checkpoint import (
        validate_resume_metadata,
    )


@unittest.skipUnless(torch is not None, "PyTorch is installed in the face environment")
class FaceResumeTest(unittest.TestCase):
    def payload(self):
        return {
            "generator": {"generator": {"weight": torch.ones(1)}},
            "epoch": 4,
            "global_steps": 50,
            "samples_seen": 500,
            "distributed": {
                "world_size": 2,
                "per_device_batch_size": 4,
                "global_batch_size": 8,
                "steps_per_epoch": 10,
                "samples_per_epoch": 100,
            },
        }

    def test_progress_starts_after_the_saved_epoch(self):
        progress = get_resume_progress(
            self.payload(),
            total_epochs=10,
            require_samples_seen=True,
        )
        self.assertEqual(progress.checkpoint_epoch, 4)
        self.assertEqual(progress.start_epoch, 5)
        self.assertEqual(progress.global_steps, 50)
        self.assertEqual(progress.samples_seen, 500)

        with self.assertRaisesRegex(RuntimeError, "--epochs must be greater"):
            get_resume_progress(
                self.payload(),
                total_epochs=5,
                require_samples_seen=True,
            )

    def test_load_rejects_missing_wrapper_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ckpt.pth"
            torch.save({"epoch": 0, "global_steps": 1}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "generator/wrapper"):
                load_face_checkpoint(checkpoint)

    def test_load_rejects_weight_only_checkpoint_without_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weight-only.pth"
            torch.save(
                {
                    "generator": {
                        "generator": {"weight": torch.ones(1)},
                    },
                    "epoch": 0,
                    "global_steps": 1,
                },
                checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "generator_optim"):
                load_face_checkpoint(checkpoint)

    def test_load_accepts_complete_legacy_training_checkpoint(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD(
            [parameter],
            lr=0.001,
            momentum=0.9,
        )
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ckpt-0.pth"
            torch.save(
                {
                    "generator": {
                        "generator": {"weight": parameter.detach().clone()},
                        "generator_optim": optimizer.state_dict(),
                        "discriminator": None,
                        "discriminator_optim": None,
                    },
                    "epoch": 0,
                    "global_steps": 4,
                },
                checkpoint,
            )
            payload = load_face_checkpoint(checkpoint)

        self.assertEqual(payload["epoch"], 0)
        self.assertIn("generator_optim", payload["generator"])

    def test_restore_loads_wrapper_and_internal_step(self):
        class Wrapper:
            def __init__(self):
                self.generator = torch.nn.Linear(1, 1)
                self.generator_optimizer = torch.optim.SGD(
                    self.generator.parameters(),
                    lr=0.1,
                    momentum=0.9,
                )
                self.global_step = 0

            def state_dict(self):
                return {
                    "generator": self.generator.state_dict(),
                    "generator_optim": self.generator_optimizer.state_dict(),
                }

            def load_state_dict(self, state):
                self.generator.load_state_dict(state["generator"])
                self.generator_optimizer.load_state_dict(
                    state["generator_optim"]
                )

        source = Wrapper()
        source.generator.weight.data.fill_(7)
        payload = {
            "generator": source.state_dict(),
            "epoch": 0,
            "global_steps": 3,
        }
        progress = get_resume_progress(
            payload,
            total_epochs=2,
            require_samples_seen=False,
        )
        target = Wrapper()
        restore_wrapper_state(target, payload, progress)

        torch.testing.assert_close(
            target.generator.weight,
            torch.full_like(target.generator.weight, 7),
        )
        self.assertEqual(target.global_step, 3)

    def test_distributed_contract_is_strict_and_legacy_metadata_warns(self):
        payload = self.payload()
        progress = get_resume_progress(
            payload,
            total_epochs=10,
            require_samples_seen=True,
        )
        warnings = validate_distributed_resume(
            payload,
            progress,
            world_size=2,
            per_device_batch_size=4,
            steps_per_epoch=10,
            samples_per_epoch=100,
            dataset_fingerprint="abc",
            expected_contract={"seed": 1},
        )
        self.assertEqual(len(warnings), 2)

        payload["distributed"]["world_size"] = 1
        with self.assertRaisesRegex(RuntimeError, "world_size"):
            validate_distributed_resume(
                payload,
                progress,
                world_size=2,
                per_device_batch_size=4,
                steps_per_epoch=10,
                samples_per_epoch=100,
                dataset_fingerprint="abc",
                expected_contract={"seed": 1},
            )

    def test_distributed_fingerprint_and_contract_must_match(self):
        payload = self.payload()
        payload["dataset_fingerprint"] = "abc"
        payload["resume_contract"] = {"seed": 1, "face_loss": "corrected"}
        progress = get_resume_progress(
            payload,
            total_epochs=10,
            require_samples_seen=True,
        )
        self.assertEqual(
            validate_distributed_resume(
                payload,
                progress,
                world_size=2,
                per_device_batch_size=4,
                steps_per_epoch=10,
                samples_per_epoch=100,
                dataset_fingerprint="abc",
                expected_contract={"seed": 1, "face_loss": "corrected"},
            ),
            [],
        )

        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            validate_distributed_resume(
                payload,
                progress,
                world_size=2,
                per_device_batch_size=4,
                steps_per_epoch=10,
                samples_per_epoch=100,
                dataset_fingerprint="different",
                expected_contract={"seed": 1, "face_loss": "corrected"},
            )
        with self.assertRaisesRegex(RuntimeError, "face_loss"):
            validate_distributed_resume(
                payload,
                progress,
                world_size=2,
                per_device_batch_size=4,
                steps_per_epoch=10,
                samples_per_epoch=100,
                dataset_fingerprint="abc",
                expected_contract={"seed": 1, "face_loss": "legacy"},
            )

    def test_schema_one_audio_contract_lineage_warns_on_resume(self):
        payload = self.payload()
        payload["dataset_fingerprint"] = "abc"
        payload["resume_contract"] = {
            "schema_version": 1,
            "seed": 1,
            "audio_config_sha256": "a" * 64,
        }
        progress = get_resume_progress(
            payload,
            total_epochs=10,
            require_samples_seen=True,
        )
        warnings = validate_distributed_resume(
            payload,
            progress,
            world_size=2,
            per_device_batch_size=4,
            steps_per_epoch=10,
            samples_per_epoch=100,
            dataset_fingerprint="abc",
            expected_contract={
                "schema_version": 1,
                "seed": 1,
                "audio_config_sha256": "a" * 64,
            },
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("schema 1", warnings[0])

    def test_rng_round_trip_and_weights_only_serialization(self):
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        state = capture_rng_state(None)
        expected = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
        )
        restore_rng_state(state, None)
        observed = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
        )
        self.assertEqual(observed[0], expected[0])
        self.assertEqual(observed[1], expected[1])
        torch.testing.assert_close(observed[2], expected[2])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pth"
            torch.save({"rng_states": [state]}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(len(loaded["rng_states"]), 1)

    def test_old_checkpoint_rng_is_an_explicit_non_bitwise_fallback(self):
        state, warning = get_rank_rng_state(
            self.payload(),
            rank=0,
            world_size=2,
        )
        self.assertIsNone(state)
        self.assertIn("not bitwise identical", warning)

    def test_new_distributed_checkpoint_metadata_is_complete(self):
        payload = self.payload()
        payload["dataset_fingerprint"] = "a" * 64
        payload["resume_contract"] = {
            "schema_version": 1,
            "trainer": "talkshow_face_large_batch",
            "seed": 1,
            "speakers": ["oliver", "seth", "conan", "chemistry"],
            "length_bucket": 8,
            "drop_last": False,
            "num_workers": 4,
            "audio_encoder": "wavlm",
            "audio_model_revision": None,
            "audio_hidden_size": 1024,
            "audio_config_sha256": "b" * 64,
            "audio_preprocessor_sha256": "c" * 64,
            "audio_do_normalize": True,
            "freeze_audio_encoder": True,
            "num_classes": 4,
            "expression": True,
            "optimizer": "SGD",
            "learning_rate": 0.001,
            "momentum": 0.9,
            "max_gradient_norm": 5.0,
            "face_loss": "corrected",
        }
        states = []
        for _ in range(2):
            state = capture_rng_state(None)
            state["torch_cuda"] = state["torch_cpu"].clone()
            states.append(state)
        payload["rng_states"] = states

        validate_resume_metadata(payload)
        del payload["rng_states"]
        with self.assertRaisesRegex(RuntimeError, "rng_states"):
            validate_resume_metadata(payload)

    def test_single_resume_checks_adjacent_run_config(self):
        current = {
            "Model": {
                "model_name": "s2g_face",
                "encoder_choice": "faceformer",
                "audio_encoder": "wavlm",
                "audio_model_revision": None,
                "audio_hidden_size": 1024,
                "audio_config_sha256": "a" * 64,
                "audio_preprocessor_sha256": "b" * 64,
                "audio_do_normalize": True,
                "freeze_audio_encoder": True,
            },
            "Data": {
                "data_root": "/shared/talkshow_split",
                "pose": {"convert_to_6d": False, "expression": True},
            },
            "DataLoader": {"batch_size": 1},
            "Reproduction": {"face_loss": "corrected"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "ckpt-49.pth"
            checkpoint.touch()
            (root / "face.json").write_text(
                json.dumps(current),
                encoding="utf-8",
            )
            self.assertEqual(
                validate_previous_run_config(checkpoint, current),
                [],
            )

            changed = json.loads(json.dumps(current))
            changed["Reproduction"]["face_loss"] = "legacy"
            with self.assertRaisesRegex(RuntimeError, "face_loss"):
                validate_previous_run_config(checkpoint, changed)

    def test_single_progress_matches_completed_epoch_boundary(self):
        validate_single_epoch_progress(
            start_epoch=5,
            global_steps=50,
            steps_per_epoch=10,
        )
        with self.assertRaisesRegex(RuntimeError, "global_steps"):
            validate_single_epoch_progress(
                start_epoch=5,
                global_steps=49,
                steps_per_epoch=10,
            )

    def test_launcher_exposes_resume_and_rejects_smoke_combination(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "train_face.sh"
        )
        help_result = subprocess.run(
            ["bash", str(launcher), "--help"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--resume-from FILE", help_result.stderr)

        conflict = subprocess.run(
            [
                "bash",
                str(launcher),
                "--upstream-dir",
                "/unused/upstream",
                "--data-root",
                "/unused/data",
                "--output-dir",
                "/unused/output",
                "--run-id",
                "resume-test",
                "--smoke",
                "--resume-from",
                "/unused/ckpt.pth",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(conflict.returncode, 2)
        self.assertIn("mutually exclusive", conflict.stderr)

    def test_launcher_rejects_a_missing_resume_checkpoint_early(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "train_face.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            data = root / "data"
            upstream.mkdir()
            data.mkdir()
            result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--upstream-dir",
                    str(upstream),
                    "--data-root",
                    str(data),
                    "--output-dir",
                    str(root / "output"),
                    "--run-id",
                    "resume-test",
                    "--resume-from",
                    str(root / "missing.pth"),
                ],
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("resume checkpoint does not exist", result.stderr)

    def test_launcher_routes_default_single_single_resume_and_ddp_resume(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "train_face.sh"
        )
        expected_commit = "9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            calls = root / "calls.txt"
            python_stub = fake_bin / "python"
            python_stub.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'python %s\\n' \"$*\" >> \"$TALKSHOW_TEST_CALLS\"\n",
                encoding="utf-8",
            )
            python_stub.chmod(0o755)
            git_stub = fake_bin / "git"
            git_stub.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' '{expected_commit}'\n",
                encoding="utf-8",
            )
            git_stub.chmod(0o755)
            torchrun_stub = fake_bin / "torchrun"
            torchrun_stub.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'torchrun %s\\n' \"$*\" >> \"$TALKSHOW_TEST_CALLS\"\n",
                encoding="utf-8",
            )
            torchrun_stub.chmod(0o755)

            upstream = root / "upstream"
            data = root / "data"
            wavlm = root / "wavlm"
            upstream.mkdir()
            data.mkdir()
            wavlm.mkdir()
            checkpoint = root / "ckpt-49.pth"
            checkpoint.touch()
            environment = os.environ.copy()
            environment["PATH"] = (
                str(fake_bin) + os.pathsep + environment["PATH"]
            )
            environment["TALKSHOW_TEST_CALLS"] = str(calls)

            common = [
                "bash",
                str(launcher),
                "--upstream-dir",
                str(upstream),
                "--data-root",
                str(data),
            ]
            commands = (
                [
                    *common,
                    "--output-dir",
                    str(root / "single-output"),
                    "--run-id",
                    "single-default",
                ],
                [
                    *common,
                    "--output-dir",
                    str(root / "single-resume-output"),
                    "--run-id",
                    "single-resume",
                    "--resume-from",
                    str(checkpoint),
                ],
                [
                    *common,
                    "--output-dir",
                    str(root / "ddp-resume-output"),
                    "--run-id",
                    "ddp-resume",
                    "--audio-encoder",
                    "wavlm",
                    "--audio-model-path",
                    str(wavlm),
                    "--num-gpus",
                    "2",
                    "--batch-size-per-gpu",
                    "2",
                    "--resume-from",
                    str(checkpoint),
                ],
            )
            for command in commands:
                result = subprocess.run(
                    command,
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            recorded = calls.read_text(encoding="utf-8")

        self.assertIn("python -W ignore scripts/train.py", recorded)
        self.assertIn("train_face_single_resume.py", recorded)
        self.assertIn("torchrun --standalone --nproc_per_node 2", recorded)
        self.assertIn("train_face_large_batch.py", recorded)
        self.assertIn(f"--resume-from {checkpoint}", recorded)


if __name__ == "__main__":
    unittest.main()
