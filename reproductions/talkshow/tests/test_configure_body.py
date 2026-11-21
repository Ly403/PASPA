import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "configure_body.py"
)


class ConfigureBodyTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.upstream = self.root / "upstream"
        (self.upstream / "config").mkdir(parents=True)
        self.data = self.root / "data"
        self.data.mkdir()
        base = {
            "Data": {"data_root": "old", "pklname": "old.pkl"},
            "DataLoader": {"batch_size": 128},
            "Train": {"epochs": 100},
            "Log": {"save_every": 50, "print_every": 200},
            "Model": {"vq_path": "old.pth"},
        }
        for name in ("body_vq.json", "body_pixel.json"):
            (self.upstream / "config" / name).write_text(
                json.dumps(base), encoding="utf-8"
            )

    def tearDown(self):
        self.tempdir.cleanup()

    def run_config(self, stage, checkpoint=None):
        output = self.root / f"{stage}.json"
        command = [
            sys.executable,
            str(SCRIPT),
            "--stage",
            stage,
            "--upstream-dir",
            str(self.upstream),
            "--data-root",
            str(self.data),
            "--output",
            str(output),
            "--cache-tag",
            "smoke",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--save-every",
            "1",
            "--print-every",
            "1",
        ]
        if checkpoint is not None:
            command.extend(["--vq-checkpoint", str(checkpoint)])
        result = subprocess.run(command, text=True, capture_output=True)
        return result, output

    def test_vq_config_preserves_model_and_overrides_run_fields(self):
        result, output = self.run_config("vq")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(config["Data"]["data_root"], str(self.data.resolve()))
        self.assertEqual(config["DataLoader"]["batch_size"], 1)
        self.assertEqual(config["Train"]["epochs"], 1)
        self.assertEqual(config["Model"]["vq_path"], "old.pth")

    def test_pixel_config_requires_and_resolves_vq_checkpoint(self):
        missing_result, _ = self.run_config("pixel")
        self.assertNotEqual(missing_result.returncode, 0)

        checkpoint = self.root / "ckpt-0.pth"
        checkpoint.touch()
        result, output = self.run_config("pixel", checkpoint)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            config["Model"]["vq_path"], str(checkpoint.resolve())
        )


if __name__ == "__main__":
    unittest.main()
