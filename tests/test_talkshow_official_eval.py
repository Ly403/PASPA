from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from scripts.talkshow_official_eval import (
    AUDIO_SR,
    BODY_INDEX,
    BODY_SAMPLE_COUNT,
    POSE_FPS,
    ProtocolError,
    _sha256_lf,
    average_clip_metrics,
    beat2_to_talkshow,
    build_input_manifest,
    isolate_face_parameters,
    make_body_partial_parameters,
    select_official_body,
    talkshow_to_beat2,
    validate_input_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _motion_arrays(
    frames: int,
    *,
    pose_offset: float = 0.0,
    expression_offset: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = (
        np.arange(frames * 165, dtype=np.float32).reshape(frames, 165)
        + pose_offset
    )
    expressions = (
        np.arange(frames * 100, dtype=np.float32).reshape(frames, 100)
        + expression_offset
    )
    betas = np.arange(300, dtype=np.float64) / 1000.0
    return poses, expressions, betas


def _write_npz(
    path: Path,
    *,
    frames: int,
    pose_offset: float = 0.0,
    expression_offset: float = 1.0,
) -> None:
    poses, expressions, betas = _motion_arrays(
        frames,
        pose_offset=pose_offset,
        expression_offset=expression_offset,
    )
    np.savez(
        path,
        poses=poses,
        expressions=expressions,
        betas=betas,
        trans=np.zeros((frames, 3), dtype=np.float32),
        mocap_frame_rate=POSE_FPS,
    )


def _write_wav(path: Path, *, frames: int) -> None:
    sample_count = AUDIO_SR * frames // POSE_FPS
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_SR)
        handle.writeframes(b"\x00\x00" * sample_count)


def _write_source_motion(path: Path, *, frames: int) -> None:
    with path.open("wb") as handle:
        pickle.dump(
            {
                "body_pose_axis": np.zeros(
                    (frames, 63),
                    dtype=np.float32,
                )
            },
            handle,
        )


class TalkSHOWOfficialProtocolTest(unittest.TestCase):
    def test_channel_mapping_and_inverse_are_exact(self) -> None:
        poses = np.arange(165, dtype=np.float32)[None, :]
        expressions = (1000 + np.arange(100, dtype=np.float32))[None, :]

        talkshow = beat2_to_talkshow(poses, expressions)

        np.testing.assert_array_equal(talkshow[:, 0:9], poses[:, 66:75])
        np.testing.assert_array_equal(talkshow[:, 9:75], poses[:, 0:66])
        np.testing.assert_array_equal(talkshow[:, 75:165], poses[:, 75:165])
        np.testing.assert_array_equal(talkshow[:, 165:265], expressions)
        restored_poses, restored_expressions = talkshow_to_beat2(talkshow)
        np.testing.assert_array_equal(restored_poses, poses)
        np.testing.assert_array_equal(restored_expressions, expressions)

    def test_body_selection_locks_the_official_129_channels(self) -> None:
        parameters = np.arange(265, dtype=np.float32)[None, :]
        selected = select_official_body(parameters)
        expected = parameters[:, :165][:, BODY_INDEX]

        self.assertEqual(selected.shape, (1, 129))
        np.testing.assert_array_equal(selected, expected)
        np.testing.assert_array_equal(
            BODY_INDEX,
            np.asarray(
                [
                    *range(18, 21),
                    *range(27, 30),
                    *range(36, 45),
                    *range(51, 165),
                ]
            ),
        )

        changed_only_outside_body = parameters.copy()
        excluded = sorted(set(range(265)) - set(BODY_INDEX.tolist()))
        changed_only_outside_body[:, excluded] += 12345
        np.testing.assert_array_equal(
            select_official_body(changed_only_outside_body),
            selected,
        )

    def test_body_partial_parameters_zero_face(self) -> None:
        body = np.arange(
            BODY_SAMPLE_COUNT * 4 * 129,
            dtype=np.float32,
        ).reshape(BODY_SAMPLE_COUNT, 4, 129)

        partial = make_body_partial_parameters(body)

        self.assertEqual(partial.shape, (2, 4, 232))
        np.testing.assert_array_equal(partial[..., :3], 0)
        np.testing.assert_array_equal(partial[..., 3:132], body)
        np.testing.assert_array_equal(partial[..., 132:], 0)

    def test_face_isolation_keeps_only_jaw_and_expression(self) -> None:
        parameters = np.arange(2 * 265, dtype=np.float32).reshape(2, 265)

        isolated = isolate_face_parameters(parameters)

        self.assertEqual(isolated.shape, parameters.shape)
        np.testing.assert_array_equal(isolated[:, :3], parameters[:, :3])
        np.testing.assert_array_equal(isolated[:, 3:165], 0)
        np.testing.assert_array_equal(
            isolated[:, 165:265],
            parameters[:, 165:265],
        )

        body_changed = parameters.copy()
        body_changed[:, 3:165] += 999
        np.testing.assert_array_equal(
            isolate_face_parameters(body_changed),
            isolated,
        )

    def test_metrics_are_averaged_per_clip_not_per_frame(self) -> None:
        result = average_clip_metrics(
            [
                {"body_lvd": 2.0, "body_error": 10.0},
                {"body_lvd": 6.0, "body_error": 20.0},
            ]
        )
        self.assertEqual(result, {"body_lvd": 4.0, "body_error": 15.0})

    def test_source_hash_normalizes_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lf = root / "lf.py"
            crlf = root / "crlf.py"
            lf.write_bytes(b"one\ntwo\n")
            crlf.write_bytes(b"one\r\ntwo\r\n")

            expected = hashlib.sha256(b"one\ntwo\n").hexdigest()
            self.assertEqual(_sha256_lf(lf), expected)
            self.assertEqual(_sha256_lf(crlf), expected)


class TalkSHOWOfficialInputTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        frames: int = 88,
    ) -> tuple[Path, Path, Path]:
        sample0 = root / "sample0" / "npz" / "test"
        sample1 = root / "sample1" / "npz" / "test"
        audio = root / "sample0" / "wav" / "test"
        for directory in (sample0, sample1, audio):
            directory.mkdir(parents=True)

        clip_id = "oliver__clip_with_under_scores"
        for directory in (sample0, sample1):
            _write_npz(
                directory / f"gt_{clip_id}.npz",
                frames=frames,
            )
        _write_npz(
            sample0 / f"res_{clip_id}.npz",
            frames=frames,
            pose_offset=10.0,
        )
        _write_npz(
            sample1 / f"res_{clip_id}.npz",
            frames=frames,
            pose_offset=20.0,
        )
        _write_wav(
            audio / f"gt_sr16000_{clip_id}.wav",
            frames=frames,
        )
        return sample0, sample1, audio

    def test_valid_two_sample_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample0, sample1, audio = self._fixture(Path(temporary))

            clips = build_input_manifest(
                [sample0, sample1],
                sample0,
                audio_dir=audio,
            )
            validation = validate_input_manifest(clips)

            self.assertEqual(len(validation.clips), 1)
            self.assertEqual(validation.total_frames, 88)
            self.assertEqual(
                validation.clips[0].clip_id,
                "oliver__clip_with_under_scores",
            )
            self.assertEqual(
                validation.clips[0].audio_protocol,
                "paspa_aligned_output",
            )

    def test_single_prediction_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample0, _, audio = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ProtocolError, "exactly two"):
                build_input_manifest(
                    [sample0],
                    sample0,
                    audio_dir=audio,
                )

    def test_length_mismatch_is_not_cropped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample0, sample1, audio = self._fixture(root)
            clip_id = "oliver__clip_with_under_scores"
            _write_npz(
                sample1 / f"res_{clip_id}.npz",
                frames=89,
                pose_offset=20.0,
            )

            clips = build_input_manifest(
                [sample0, sample1],
                sample0,
                audio_dir=audio,
            )
            with self.assertRaisesRegex(ProtocolError, "complete frame set"):
                validate_input_manifest(clips)

    def test_official_minimum_length_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample0, sample1, audio = self._fixture(
                Path(temporary),
                frames=87,
            )
            clips = build_input_manifest(
                [sample0, sample1],
                sample0,
                audio_dir=audio,
            )
            with self.assertRaisesRegex(ProtocolError, "shorter than 88"):
                validate_input_manifest(clips)

    def test_original_source_audio_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample0, sample1, _ = self._fixture(root)
            source_wav = (
                root
                / "split"
                / "oliver"
                / "video"
                / "test"
                / "clip_with_under_scores"
                / "clip_with_under_scores.wav"
            )
            source_wav.parent.mkdir(parents=True)
            _write_wav(source_wav, frames=88)
            _write_source_motion(source_wav.with_suffix(".pkl"), frames=88)

            clips = build_input_manifest(
                [sample0, sample1],
                sample0,
                source_audio_root=root / "split",
            )
            validation = validate_input_manifest(clips)

            self.assertEqual(
                validation.clips[0].audio_path,
                source_wav.resolve(),
            )
            self.assertEqual(
                validation.clips[0].audio_protocol,
                "talkshow_original_source",
            )

    def test_source_subset_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample0, sample1, _ = self._fixture(root)
            split_root = root / "split"
            for sequence in ("clip_with_under_scores", "another_clip"):
                source_wav = (
                    split_root
                    / "oliver"
                    / "video"
                    / "test"
                    / sequence
                    / f"{sequence}.wav"
                )
                source_wav.parent.mkdir(parents=True, exist_ok=True)
                _write_wav(source_wav, frames=88)
                _write_source_motion(
                    source_wav.with_suffix(".pkl"),
                    frames=88,
                )

            with self.assertRaisesRegex(ProtocolError, "do not cover"):
                build_input_manifest(
                    [sample0, sample1],
                    sample0,
                    source_audio_root=split_root,
                )

            manifest = build_input_manifest(
                [sample0, sample1],
                sample0,
                source_audio_root=split_root,
                allow_source_subset=True,
            )
            validation = validate_input_manifest(manifest)
            self.assertEqual(
                validation.missing_source_clip_ids,
                ("oliver__another_clip",),
            )

    def test_source_clips_shorter_than_88_match_official_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample0, sample1, _ = self._fixture(root)
            split_root = root / "split"
            for sequence, frames in (
                ("clip_with_under_scores", 88),
                ("officially_filtered_short_clip", 62),
            ):
                source_wav = (
                    split_root
                    / "oliver"
                    / "video"
                    / "test"
                    / sequence
                    / f"{sequence}.wav"
                )
                source_wav.parent.mkdir(parents=True, exist_ok=True)
                _write_wav(source_wav, frames=frames)
                _write_source_motion(
                    source_wav.with_suffix(".pkl"),
                    frames=frames,
                )

            manifest = build_input_manifest(
                [sample0, sample1],
                sample0,
                source_audio_root=split_root,
            )
            validation = validate_input_manifest(manifest)

            self.assertEqual(validation.source_view_clip_count, 2)
            self.assertEqual(validation.source_test_clip_count, 1)
            self.assertEqual(
                validation.official_short_source_clip_ids,
                ("oliver__officially_filtered_short_clip",),
            )
            self.assertFalse(validation.missing_source_clip_ids)

    def test_help_does_not_require_heavy_runtime_dependencies(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "talkshow_official_eval.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--validate-input-only", result.stdout)

    def test_validate_only_cli_writes_a_provenance_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample0, sample1, audio = self._fixture(root)
            report_path = root / "validation.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(
                        REPO_ROOT
                        / "scripts"
                        / "talkshow_official_eval.py"
                    ),
                    "--pred-dir",
                    str(sample0),
                    "--pred-dir",
                    str(sample1),
                    "--gt-dir",
                    str(sample0),
                    "--audio-dir",
                    str(audio),
                    "--validate-input-only",
                    "--output",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "input_validation_passed")
            self.assertEqual(report["counts"]["clips"], 1)
            self.assertEqual(
                report["protocol"]["body_samples_per_clip"],
                2,
            )
            self.assertEqual(
                len(report["inputs"]["generation_runs"]),
                2,
            )

    def test_resume_can_use_an_explicit_inference_seed(self) -> None:
        trainer_source = (
            REPO_ROOT / "diffusion" / "trainer" / "trainer.py"
        ).read_text(encoding="utf-8")
        resume_block = trainer_source.split(
            "    def resume_training(self):",
            maxsplit=1,
        )[1].split("    def configure_scheduler(self):", maxsplit=1)[0]
        self.assertIn(
            'config.train.get("restore_rng_state", True)',
            resume_block,
        )
        self.assertIn("self._restore_rng_state(rng_state)", resume_block)
        self.assertIn("self.rng_state = None", resume_block)


if __name__ == "__main__":
    unittest.main()
