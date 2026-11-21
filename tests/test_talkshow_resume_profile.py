from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.show.verify_talkshow_resume_profile import (
    IDENTITY_CONTRACT,
    ResumeProfileError,
    verify_resume_profile,
)


class TalkSHOWResumeProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            tempfile.gettempdir()
        except FileNotFoundError as error:
            raise unittest.SkipTest(
                "environment has no writable temporary directory"
            ) from error

    def _make_run(
        self,
        root: Path,
        *,
        seq_size: int,
        block_size: int,
        profile: str | None,
        identity_contract: str | None = IDENTITY_CONTRACT,
        raw_pid_min: int = 21,
        raw_pid_max: int = 24,
        model_pid_min: int = 0,
        model_pid_max: int = 3,
        main_num_person: int = 4,
        face_speaker_id_offset: int = 0,
    ) -> Path:
        output = root / "output"
        checkpoints = output / "checkpoints"
        checkpoints.mkdir(parents=True)
        checkpoint = checkpoints / "epoch_10_step_20.pth"
        checkpoint.touch()
        profile_fields = []
        if profile is not None:
            profile_fields.append(f"  name: {profile}")
        if identity_contract is not None:
            profile_fields.extend(
                [
                    f"  identity_contract: {identity_contract}",
                    f"  raw_pid_min: {raw_pid_min}",
                    f"  raw_pid_max: {raw_pid_max}",
                    f"  model_pid_min: {model_pid_min}",
                    f"  model_pid_max: {model_pid_max}",
                ]
            )
        marker = (
            "talkshow_profile:\n" + "\n".join(profile_fields) + "\n"
            if profile_fields
            else ""
        )
        (output / "config.yaml").write_text(
            "data:\n"
            f"  pose_length: {seq_size}\n"
            "train:\n"
            f"  num_frame_per_block: {block_size}\n"
            "model:\n"
            "  model:\n"
            "    params:\n"
            f"      num_person: {main_num_person}\n"
            "  speech_model:\n"
            "    params:\n"
            f"      speaker_id_offset: {face_speaker_id_offset}\n"
            f"{marker}",
            encoding="utf-8",
        )
        return checkpoint

    def test_matching_profile_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_run(
                Path(temporary),
                seq_size=128,
                block_size=32,
                profile="block32_seq128",
            )
            config_path = verify_resume_profile(
                checkpoint,
                profile="block32_seq128",
                expected_seq_size=128,
                expected_block_size=32,
            )
            self.assertEqual(config_path, checkpoint.parent.parent / "config.yaml")

    def test_cross_profile_resume_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_run(
                Path(temporary),
                seq_size=128,
                block_size=32,
                profile="block32_seq128",
            )
            with self.assertRaisesRegex(
                ResumeProfileError,
                "checkpoint/profile mismatch",
            ):
                verify_resume_profile(
                    checkpoint,
                    profile="block16_seq64",
                    expected_seq_size=64,
                    expected_block_size=16,
                )

    def test_legacy_run_without_identity_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_run(
                Path(temporary),
                seq_size=256,
                block_size=64,
                profile=None,
                identity_contract=None,
                main_num_person=31,
                face_speaker_id_offset=20,
            )
            with self.assertRaisesRegex(
                ResumeProfileError,
                "no identity contract",
            ):
                verify_resume_profile(
                    checkpoint,
                    profile="block64_seq256",
                    expected_seq_size=256,
                    expected_block_size=64,
                )

    def test_identity_range_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_run(
                Path(temporary),
                seq_size=256,
                block_size=64,
                profile="block64_seq256",
                raw_pid_min=20,
            )
            with self.assertRaisesRegex(
                ResumeProfileError,
                "raw_pid_min=20",
            ):
                verify_resume_profile(
                    checkpoint,
                    profile="block64_seq256",
                    expected_seq_size=256,
                    expected_block_size=64,
                )

    def test_different_identity_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_run(
                Path(temporary),
                seq_size=256,
                block_size=64,
                profile="block64_seq256",
                identity_contract="raw20_23_to_model0_3",
            )
            with self.assertRaisesRegex(
                ResumeProfileError,
                "identity contract mismatch",
            ):
                verify_resume_profile(
                    checkpoint,
                    profile="block64_seq256",
                    expected_seq_size=256,
                    expected_block_size=64,
                )

    def test_main_and_face_runtime_identity_drift_is_rejected(self) -> None:
        cases = (
            (
                {"main_num_person": 31},
                r"model\.model\.params\.num_person=31",
            ),
            (
                {"face_speaker_id_offset": 20},
                r"model\.speech_model\.params\.speaker_id_offset=20",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as temporary:
                    checkpoint = self._make_run(
                        Path(temporary),
                        seq_size=128,
                        block_size=32,
                        profile="block32_seq128",
                        **overrides,
                    )
                    with self.assertRaisesRegex(
                        ResumeProfileError,
                        message,
                    ):
                        verify_resume_profile(
                            checkpoint,
                            profile="block32_seq128",
                            expected_seq_size=128,
                            expected_block_size=32,
                        )


if __name__ == "__main__":
    unittest.main()
