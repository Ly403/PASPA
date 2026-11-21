from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = REPO_ROOT / "configs" / "talkshow_profile"
LAUNCHER = REPO_ROOT / "train_scripts" / "run_talkshow.sh"

PROFILES = {
    "block16_seq64": (16, 64, "show_global_64_lmdb_talkshow_face"),
    "block32_seq128": (32, 128, "show_global_128_lmdb_talkshow_face"),
    "block64_seq256": (64, 256, "show_global_256_lmdb_talkshow_face"),
}


def _has_scalar(text: str, key: str, value: int | str) -> bool:
    return re.search(
        rf"(?m)^\s*{re.escape(key)}:\s*{re.escape(str(value))}\s*$",
        text,
    ) is not None


class TalkSHOWScaleProfileTest(unittest.TestCase):
    def test_three_profiles_keep_one_atomic_scale_contract(self) -> None:
        self.assertEqual(
            {path.stem for path in PROFILE_ROOT.glob("*.yaml")},
            set(PROFILES),
        )
        for name, (block, sequence, data_config) in PROFILES.items():
            with self.subTest(profile=name):
                text = (PROFILE_ROOT / f"{name}.yaml").read_text(
                    encoding="utf-8"
                )
                self.assertIn(f"- override /data: {data_config}", text)
                self.assertTrue(_has_scalar(text, "name", name))
                self.assertTrue(_has_scalar(text, "seq_size", sequence))
                self.assertTrue(_has_scalar(text, "block_size", block))
                self.assertTrue(_has_scalar(text, "stride_size", 20))
                self.assertTrue(_has_scalar(text, "pose_length", sequence))
                self.assertTrue(
                    _has_scalar(text, "num_frame_per_block", block)
                )
                self.assertTrue(_has_scalar(text, "train_batch_size", 72))
                self.assertTrue(_has_scalar(text, "num_epochs", 1000))
                self.assertEqual(
                    text.count(f"expected_seq_size: {sequence}"), 3
                )
                self.assertEqual(sequence, 4 * block)

    def test_data_configs_keep_talkshow_identity_contract(self) -> None:
        for _, (_, sequence, data_config) in PROFILES.items():
            with self.subTest(sequence=sequence):
                text = (
                    REPO_ROOT / "configs" / "data" / f"{data_config}.yaml"
                ).read_text(encoding="utf-8")
                self.assertEqual(
                    text.count("expected_pids: [21, 22, 23, 24]"), 3
                )
                self.assertEqual(text.count("expected_pid_offset: 1"), 3)
                self.assertEqual(text.count("model_pid_offset: 21"), 3)
                self.assertEqual(
                    text.count(f"expected_seq_size: {sequence}"), 3
                )

    def test_launcher_exposes_one_mode_and_one_profile_interface(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for expected in (
            'TALKSHOW_PROFILE="${TALKSHOW_PROFILE:-block64_seq256}"',
            'TALKSHOW_MODE="${TALKSHOW_MODE:-train}"',
            'NUM_PROCESSES="${NUM_PROCESSES:-4}"',
            'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-72}"',
            'NUM_EPOCHS="${NUM_EPOCHS:-1000}"',
            'TALKSHOW_CFG_AUDIO_SCALE="${TALKSHOW_CFG_AUDIO_SCALE:-1.5}"',
            'TALKSHOW_CFG_HISTORY_SCALE="${TALKSHOW_CFG_HISTORY_SCALE:-1.5}"',
            'TALKSHOW_CFG_MODE="${TALKSHOW_CFG_MODE:-interaction}"',
            "model=show_causal_emage_deltanet_talkshow_face",
            'train.num_frame_per_block="$BLOCK_SIZE"',
            "verify_talkshow_resume_profile.py",
            "verify_lmdb_contract.py",
        ):
            self.assertIn(expected, text)
        for mode in ("train", "validation", "test"):
            self.assertRegex(text, rf"(?m)^\s+{mode}\)$")
        for name, (block, sequence, _) in PROFILES.items():
            self.assertIn(f"    {name})", text)
            self.assertIn(f"        BLOCK_SIZE={block}", text)
            self.assertIn(f"        SEQ_SIZE={sequence}", text)

    def test_talkshow_release_does_not_modify_beat_launcher(self) -> None:
        talkshow = LAUNCHER.read_text(encoding="utf-8")
        beat = (
            REPO_ROOT / "train_scripts" / "train_beat2.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("beatx_global_256_lmdb", talkshow)
        self.assertNotIn("TALKSHOW", beat)
        self.assertNotIn("show_global", beat)

    def test_documentation_uses_the_release_launcher(self) -> None:
        docs = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "README_zh.md",
            PROFILE_ROOT / "README_zh.md",
        )
        for path in docs:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("train_scripts/run_talkshow.sh", text)
                self.assertNotIn(
                    "train_talkshow_causal_emage_deltanet_canonical.sh",
                    text,
                )


if __name__ == "__main__":
    unittest.main()
