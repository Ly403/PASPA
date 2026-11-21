from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Barrier, Lock, get_ident
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_talkshow_pth_dir.py"
SPEC = importlib.util.spec_from_file_location(
    "eval_talkshow_pth_dir_under_test",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _metric_payload(fgd: float) -> dict[str, object]:
    return {
        "metrics": {
            name: fgd if name == "fgd" else 1.0
            for name in MODULE.OFFICIAL_METRIC_NAMES
        }
    }


class TalkSHOWCheckpointSweepTest(unittest.TestCase):
    def test_collect_checkpoints_uses_numeric_epoch_step_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "epoch_10_step_2.pth",
                "epoch_2_step_30_loss.pth",
                "epoch_2.pth",
                "best.pth",
            ):
                (root / name).touch()

            checkpoints, checkpoint_root = MODULE.collect_checkpoints(root)

            self.assertEqual(checkpoint_root, root.resolve())
            self.assertEqual(
                [path.name for path in checkpoints],
                [
                    "epoch_2.pth",
                    "epoch_2_step_30_loss.pth",
                    "epoch_10_step_2.pth",
                ],
            )

    def test_epoch_range_is_inclusive_and_keeps_all_steps(self) -> None:
        checkpoints = [
            Path("/checkpoints/epoch_499_step_1.pth"),
            Path("/checkpoints/epoch_500_step_2.pth"),
            Path("/checkpoints/epoch_500_step_3.pth"),
            Path("/checkpoints/epoch_800_step_4.pth"),
            Path("/checkpoints/epoch_801_step_5.pth"),
        ]
        selected = MODULE.filter_checkpoints_by_epoch(
            checkpoints,
            (500, 800),
        )
        self.assertEqual(
            [checkpoint.name for checkpoint in selected],
            [
                "epoch_500_step_2.pth",
                "epoch_500_step_3.pth",
                "epoch_800_step_4.pth",
            ],
        )
        with self.assertRaisesRegex(
            MODULE.SweepError,
            "start must not exceed",
        ):
            MODULE.filter_checkpoints_by_epoch(checkpoints, (800, 500))

    def test_commands_use_test_only_generation_and_official_b2_inputs(
        self,
    ) -> None:
        compatibility_args = MODULE._build_parser().parse_args(
            [
                "/checkpoints",
                "--profile",
                "block64_seq256",
                "--talkshow-root",
                "/TalkSHOW",
                "--feature-extractor",
                "/TalkSHOW/feature_extractor.pth",
                "--num-processes",
                "2",
                "--epoch-range",
                "500",
                "800",
            ]
        )
        self.assertEqual(compatibility_args.parallel_workers, 1)
        self.assertEqual(compatibility_args.gpus_per_worker, 2)
        self.assertEqual(compatibility_args.epoch_range, [500, 800])

        single_gpu_args = MODULE._build_parser().parse_args(
            [
                "/checkpoints",
                "--profile",
                "block64_seq256",
                "--talkshow-root",
                "/TalkSHOW",
                "--feature-extractor",
                "/TalkSHOW/feature_extractor.pth",
                "--gpus-per-worker",
                "1",
            ]
        )
        single_gpu_environment = MODULE._worker_environment(
            {
                "LOCAL_RANK": "9",
                "RANK": "9",
                "WORLD_SIZE": "99",
                "MASTER_ADDR": "stale",
                "MASTER_PORT": "1",
            },
            slot=MODULE.WorkerSlot(
                index=0,
                devices=("7",),
                main_process_port=29520,
            ),
            args=single_gpu_args,
        )
        self.assertEqual(single_gpu_environment["LOCAL_RANK"], "0")
        self.assertEqual(single_gpu_environment["RANK"], "0")
        self.assertEqual(single_gpu_environment["WORLD_SIZE"], "1")
        self.assertEqual(
            single_gpu_environment["MASTER_ADDR"],
            "127.0.0.1",
        )
        self.assertEqual(single_gpu_environment["MASTER_PORT"], "29520")

        generation = MODULE.build_generation_command(
            canonical_launcher=Path("/repo/train_talkshow.sh"),
            checkpoint=Path("/checkpoints/epoch_4_step_20.pth"),
            seed=1001,
            hydra_run_dir=Path("/results/sample0"),
        )
        self.assertIn(
            "model.load_from=/checkpoints/epoch_4_step_20.pth",
            generation,
        )
        self.assertIn("train.seed=1001", generation)
        self.assertIn("++hydra.run.dir=/results/sample0", generation)

        official = MODULE.build_official_eval_command(
            evaluator=Path("/repo/talkshow_official_eval.py"),
            sample0_work_dir=Path("/results/sample0/output"),
            sample1_work_dir=Path("/results/sample1/output"),
            talkshow_root=Path("/TalkSHOW"),
            feature_extractor=Path("/TalkSHOW/feature_extractor.pth"),
            output_json=Path("/results/metrics.json"),
            source_audio_root=Path("/split-view"),
            smplx_path=None,
            device="cuda:0",
            expected_feature_extractor_sha256=None,
            expected_smplx_sha256=None,
            allow_unpinned_talkshow=False,
            allow_source_subset=False,
        )
        pred_positions = [
            index
            for index, value in enumerate(official)
            if value == "--pred-dir"
        ]
        self.assertEqual(len(pred_positions), 2)
        self.assertIn("/results/sample0/output/npz/test", official)
        self.assertIn("/results/sample1/output/npz/test", official)
        self.assertIn("--source-audio-root", official)
        self.assertNotIn("--audio-dir", official)

    def test_metric_report_is_strict_and_best_is_minimum_fgd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "metrics.json"
            report.write_text(
                json.dumps(_metric_payload(3.25)),
                encoding="utf-8",
            )
            metrics = MODULE.load_official_metrics(report)
            self.assertEqual(metrics["fgd"], 3.25)

        best = MODULE.best_successful_row(
            [
                {"checkpoint": "a.pth", "status": "ok", "fgd": 3.25},
                {"checkpoint": "b.pth", "status": "failed", "fgd": 0.1},
                {"checkpoint": "c.pth", "status": "ok", "fgd": 2.0},
            ]
        )
        self.assertIsNotNone(best)
        self.assertEqual(best["checkpoint"], "c.pth")

    def test_core_sweep_writes_incremental_summary_without_deleting_ckpts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_dir = root / "checkpoints"
            talkshow_root = root / "TalkSHOW"
            source_audio_root = root / "split-view"
            output_dir = root / "sweep"
            for directory in (
                checkpoint_dir,
                talkshow_root,
                source_audio_root,
            ):
                directory.mkdir()
            feature_extractor = root / "feature_extractor.pth"
            feature_extractor.touch()
            checkpoints = (
                checkpoint_dir / "epoch_1_step_10.pth",
                checkpoint_dir / "epoch_2_step_20.pth",
            )
            for checkpoint in checkpoints:
                checkpoint.touch()

            args = MODULE._build_parser().parse_args(
                [
                    str(checkpoint_dir),
                    "--profile",
                    "block64_seq256",
                    "--talkshow-root",
                    str(talkshow_root),
                    "--feature-extractor",
                    str(feature_extractor),
                    "--source-audio-root",
                    str(source_audio_root),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            generation_environments: list[dict[str, str]] = []

            def fake_run(
                command: list[str],
                *,
                cwd: Path,
                environment: dict[str, str],
            ) -> None:
                del cwd
                if command[0] == "bash":
                    generation_environments.append(dict(environment))
                    hydra_override = next(
                        value
                        for value in command
                        if value.startswith("++hydra.run.dir=")
                    )
                    run_dir = Path(hydra_override.split("=", 1)[1])
                    npz_dir = run_dir / "output" / "npz" / "test"
                    wav_dir = run_dir / "output" / "wav" / "test"
                    npz_dir.mkdir(parents=True)
                    wav_dir.mkdir(parents=True)
                    (npz_dir / "res_clip.npz").touch()
                    (npz_dir / "gt_clip.npz").touch()
                    (wav_dir / "gt_sr16000_clip.wav").touch()
                    return

                output_index = command.index("--output") + 1
                report_path = Path(command[output_index])
                report_path.parent.mkdir(parents=True, exist_ok=True)
                fgd = 1.0 if "epoch_2_step_20" in str(report_path) else 2.0
                report_path.write_text(
                    json.dumps(_metric_payload(fgd)),
                    encoding="utf-8",
                )

            required_environment = {
                name: f"/absolute/{name.lower()}"
                for name in MODULE.REQUIRED_GENERATION_ENV
            }
            required_environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            with mock.patch.dict(
                os.environ,
                required_environment,
                clear=False,
            ), mock.patch.object(
                MODULE,
                "_run_command",
                side_effect=fake_run,
            ), mock.patch.object(
                MODULE,
                "_run_generation_command",
                side_effect=lambda command, **kwargs: (
                    fake_run(
                        command,
                        cwd=kwargs["cwd"],
                        environment=kwargs["environment"],
                    )
                    or MODULE.GenerationRunResult(
                        completion_marker_seen=True,
                        watchdog_triggered=False,
                    )
                ),
            ):
                result = MODULE.run_sweep(args)

            self.assertEqual(result, 0)
            self.assertEqual(
                [env["RUN_CONTRACT_PREFLIGHT"] for env in generation_environments],
                ["1", "0", "0", "0"],
            )
            self.assertTrue(
                all(
                    env["TALKSHOW_MODE"] == "test"
                    for env in generation_environments
                )
            )
            summary = json.loads(
                (
                    output_dir / "talkshow_checkpoint_metrics.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                summary["best_checkpoint"].endswith(
                    "epoch_2_step_20.pth"
                )
            )
            self.assertEqual(summary["best_fgd"], 1.0)
            self.assertTrue(all(path.is_file() for path in checkpoints))

    def test_parallel_workers_keep_each_checkpoint_on_one_gpu_slot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_dir = root / "checkpoints"
            talkshow_root = root / "TalkSHOW"
            output_dir = root / "parallel-sweep"
            checkpoint_dir.mkdir()
            talkshow_root.mkdir()
            feature_extractor = root / "feature_extractor.pth"
            feature_extractor.touch()
            for epoch in range(1, 5):
                (checkpoint_dir / f"epoch_{epoch}_step_{epoch * 10}.pth").touch()

            args = MODULE._build_parser().parse_args(
                [
                    str(checkpoint_dir),
                    "--profile",
                    "block64_seq256",
                    "--talkshow-root",
                    str(talkshow_root),
                    "--feature-extractor",
                    str(feature_extractor),
                    "--output-dir",
                    str(output_dir),
                    "--parallel-workers",
                    "2",
                    "--gpus-per-worker",
                    "1",
                    "--cuda-visible-devices",
                    "4,7",
                ]
            )
            parallel_barrier = Barrier(2, timeout=5)
            parallel_barrier_lock = Lock()
            zero_preflight_barrier_count = 0
            record_lock = Lock()
            command_records: dict[str, list[tuple[str, str, int]]] = {}
            generation_preflights: list[str] = []

            def fake_parallel_run(
                command: list[str],
                *,
                cwd: Path,
                environment: dict[str, str],
            ) -> None:
                nonlocal zero_preflight_barrier_count
                del cwd
                if command[0] == "bash":
                    load_override = next(
                        value
                        for value in command
                        if value.startswith("model.load_from=")
                    )
                    checkpoint_name = Path(
                        load_override.split("=", 1)[1]
                    ).stem
                    should_wait = False
                    with parallel_barrier_lock:
                        if (
                            environment["RUN_CONTRACT_PREFLIGHT"] == "0"
                            and zero_preflight_barrier_count < 2
                        ):
                            zero_preflight_barrier_count += 1
                            should_wait = True
                    if should_wait:
                        parallel_barrier.wait()
                    hydra_override = next(
                        value
                        for value in command
                        if value.startswith("++hydra.run.dir=")
                    )
                    run_dir = Path(hydra_override.split("=", 1)[1])
                    npz_dir = run_dir / "output" / "npz" / "test"
                    wav_dir = run_dir / "output" / "wav" / "test"
                    npz_dir.mkdir(parents=True)
                    wav_dir.mkdir(parents=True)
                    (npz_dir / "res_clip.npz").touch()
                    (npz_dir / "gt_clip.npz").touch()
                    (wav_dir / "gt_sr16000_clip.wav").touch()
                    with record_lock:
                        generation_preflights.append(
                            environment["RUN_CONTRACT_PREFLIGHT"]
                        )
                else:
                    output_index = command.index("--output") + 1
                    report_path = Path(command[output_index])
                    checkpoint_name = report_path.parent.name
                    report_path.write_text(
                        json.dumps(_metric_payload(float(checkpoint_name[6]))),
                        encoding="utf-8",
                    )

                with record_lock:
                    command_records.setdefault(checkpoint_name, []).append(
                        (
                            environment["CUDA_VISIBLE_DEVICES"],
                            environment["MAIN_PROCESS_PORT"],
                            get_ident(),
                        )
                    )

            required_environment = {
                name: f"/absolute/{name.lower()}"
                for name in MODULE.REQUIRED_GENERATION_ENV
            }
            with mock.patch.dict(
                os.environ,
                required_environment,
                clear=False,
            ), mock.patch.object(
                MODULE,
                "_run_command",
                new=fake_parallel_run,
            ), mock.patch.object(
                MODULE,
                "_run_generation_command",
                side_effect=lambda command, **kwargs: (
                    fake_parallel_run(
                        command,
                        cwd=kwargs["cwd"],
                        environment=kwargs["environment"],
                    )
                    or MODULE.GenerationRunResult(
                        completion_marker_seen=True,
                        watchdog_triggered=False,
                    )
                ),
            ):
                result = MODULE.run_sweep(args)

            self.assertEqual(result, 0)
            self.assertEqual(len(command_records), 4)
            self.assertEqual(generation_preflights.count("1"), 1)
            self.assertEqual(generation_preflights.count("0"), 7)
            observed_devices = set()
            observed_threads = set()
            for records in command_records.values():
                self.assertEqual(len(records), 3)
                self.assertEqual(len({record[0] for record in records}), 1)
                self.assertEqual(len({record[1] for record in records}), 1)
                observed_devices.add(records[0][0])
                observed_threads.update(record[2] for record in records)
            self.assertEqual(observed_devices, {"4", "7"})
            self.assertGreaterEqual(len(observed_threads), 2)

            summary = json.loads(
                (
                    output_dir / "talkshow_checkpoint_metrics.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["parallel_workers"], 2)
            self.assertEqual(summary["gpus_per_worker"], 1)
            self.assertEqual(len(summary["rows"]), 4)

    @unittest.skipUnless(
        os.name == "posix",
        "process-group watchdog requires POSIX semantics",
    )
    def test_watchdog_reaps_generation_after_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker_path = root / "complete.json"
            child_code = (
                "import os,subprocess,sys,time;"
                "from scripts.seed_sweep.completion_marker "
                "import write_completion_marker;"
                "subprocess.Popen([sys.executable,'-c',"
                "'import time;time.sleep(60)']);"
                "write_completion_marker("
                "os.environ['SEED_SWEEP_INFERENCE_COMPLETION_MARKER'],"
                "producer='main.py',"
                "run_id=os.environ["
                "'SEED_SWEEP_INFERENCE_COMPLETION_RUN_ID'],"
                "token=os.environ["
                "'SEED_SWEEP_INFERENCE_COMPLETION_TOKEN']);"
                "time.sleep(60)"
            )
            started_at = time.monotonic()
            result = MODULE._run_generation_command(
                [sys.executable, "-c", child_code],
                cwd=REPO_ROOT,
                environment=os.environ.copy(),
                completion_marker_path=marker_path,
                completion_run_id="watchdog-test",
                watchdog=MODULE.GenerationWatchdogConfig(
                    poll_seconds=0.01,
                    post_completion_grace_seconds=0.01,
                    terminate_grace_seconds=0.2,
                    hard_timeout_seconds=5.0,
                ),
            )
            elapsed = time.monotonic() - started_at

            self.assertTrue(result.completion_marker_seen)
            self.assertTrue(result.watchdog_triggered)
            self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
