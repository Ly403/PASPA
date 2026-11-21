#!/usr/bin/env python3
r"""Evaluate a directory of TalkSHOW checkpoints with the official protocol.

For every matching checkpoint this script:

1. runs the canonical TalkSHOW test-only launcher with two independent seeds;
2. evaluates the two generated samples with ``talkshow_official_eval.py``;
3. writes incremental CSV/JSON summaries; and
4. reports the checkpoint with the lowest official FGD.

The script never deletes checkpoints or generated outputs.  Checkpoint
selection on a test split is methodologically different from final test-only
reporting; prefer a validation split for model selection whenever possible.

用法
----

先设置主模型 test 所需的五个绝对路径：

.. code-block:: bash

   export SHOW_LMDB_DIR=/absolute/path/to/show_lmdb
   export TALKSHOW_FACE_CKPT=/absolute/path/to/face/ckpt-999.pth
   export WAVLM_DIR=/absolute/path/to/wavlm-libri-clean-100h-large
   export SHOW_REST_JOINT_PATH="$SHOW_LMDB_DIR/rest_joint_position_show.mat"
   export EMAGE_EVALTOOLS_DIR=/absolute/path/to/emage_evaltools

然后传入 checkpoint 目录、与训练一致的 profile 以及 TalkSHOW 官方评估资产。
下面示例建立 4 个端到端 worker，每个 worker 固定占用 1 张 GPU：

.. code-block:: bash

   python scripts/eval_talkshow_pth_dir.py \
     /absolute/path/to/run/output/checkpoints \
     --profile block64_seq256 \
     --talkshow-root /absolute/path/to/TalkSHOW \
     --feature-extractor /absolute/path/to/experiments/feature_extractor.pth \
     --source-audio-root /absolute/path/to/talkshow_split_view \
     --smplx-path /absolute/path/to/TalkSHOW/visualise \
     --parallel-workers 4 \
     --gpus-per-worker 1 \
     --cuda-visible-devices 0,1,2,3 \
     --num-workers 2 \
     --epoch-range 500 800

每个 worker 把一个 checkpoint 的完整流程作为原子任务：先后生成 seed 1001、
seed 2002，再立即在同一 GPU slot 上执行官方 evaluator，最后把一行指标返回
主线程。主线程只负责继续分配 checkpoint、增量写 CSV/JSON，并在全部 worker
结束后按最低 FGD 汇总。它不会先生成全部 checkpoint 再统一评估。
每次纯测试均通过 ``model.load_from`` 只加载模型权重，不使用
``resume_from``，因此不会恢复训练 epoch/step、optimizer、scheduler 或
checkpoint RNG。

默认值仍是 ``--parallel-workers 1 --gpus-per-worker 4``，即原来的“单个
checkpoint 使用四卡、checkpoint 之间串行”。``--num-processes`` 保留为
``--gpus-per-worker`` 的兼容别名。全局设备数量必须等于
``parallel_workers * gpus_per_worker``，每个 worker 使用独立设备子集和独立
Accelerate 端口。

``--epoch-range 500 800`` 只选择 epoch 位于闭区间
``[500, 800]`` 内的 checkpoint；同一 epoch 如果有多个 step 权重，
它们都会被评估。不传该参数时仍扫描全部 checkpoint。

每次 test 会在所有 rank 完成 ``trainer.test()`` 后写入带随机 token 的
completion marker。若 Accelerate/DDP 在此后未正常退出，默认 watchdog
等待 10 秒后会清理该 worker 的整个进程组，然后继续官方评估。
可用 ``--watchdog-post-completion-grace-seconds`` 调整宽限期；
``--generation-hard-timeout-seconds`` 可选地限制完成信号之前的最长
采样时间，默认 0 表示不设硬超时。

三档 profile 与测试数据必须严格对应：

* ``block16_seq64``：sequence 64 / block 16；
* ``block32_seq128``：sequence 128 / block 32；
* ``block64_seq256``：sequence 256 / block 64。

checkpoint 目录只进行一层扫描，文件名需匹配
``epoch_<N>.pth`` 或 ``epoch_<N>_step_<N>*.pth``。每个 checkpoint 必须保留
训练目录旁的 resolved config：

.. code-block:: text

   <run>/output/config.yaml
   <run>/output/checkpoints/epoch_X_step_Y.pth

默认对每个 checkpoint 使用 seed 1001 和 2002，按最低官方 FGD 选优。严格
test-LMDB/资产 preflight 默认只在第一轮执行一次，可用
``--preflight always|once|never`` 调整。省略 ``--source-audio-root`` 时会使用
sample0 的对齐 WAV，但官方报告会记录该协议差异。

默认输出到
``<checkpoint-dir>/_talkshow_eval_<run-id>/``，其中
``talkshow_checkpoint_metrics.csv`` 和
``talkshow_checkpoint_metrics.json`` 会在每个 checkpoint 完成后增量更新。
脚本不会删除权重或生成结果；完整 B=2 test 会占用较大磁盘空间。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import shlex
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

try:
    from seed_sweep.completion_marker import read_completion_marker
except ImportError:
    from scripts.seed_sweep.completion_marker import read_completion_marker


PROFILE_CHOICES = (
    "block16_seq64",
    "block32_seq128",
    "block64_seq256",
)
CFG_MODE_CHOICES = (
    "joint",
    "independent",
    "audio_first",
    "state_first",
    "interaction",
)
REQUIRED_GENERATION_ENV = (
    "SHOW_LMDB_DIR",
    "TALKSHOW_FACE_CKPT",
    "WAVLM_DIR",
    "SHOW_REST_JOINT_PATH",
    "EMAGE_EVALTOOLS_DIR",
)
OFFICIAL_METRIC_NAMES = (
    "fgd",
    "feat_dist",
    "bc",
    "body_lvd",
    "body_error",
    "diverse",
    "jaw_l1",
    "landmark_l1",
    "face_lvd",
)
CSV_FIELDS = (
    "checkpoint",
    "epoch",
    "step",
    "worker",
    "cuda_visible_devices",
    "main_process_port",
    "watchdog_terminated_samples",
    *OFFICIAL_METRIC_NAMES,
    "status",
    "error",
    "sample0_dir",
    "sample1_dir",
    "metrics_json",
)
CHECKPOINT_PATTERN = re.compile(
    r"^epoch_(?P<epoch>\d+)"
    r"(?:_step_(?P<step>\d+))?"
    r"(?:_.*)?\.pth$"
)


class SweepError(RuntimeError):
    """A TalkSHOW checkpoint sweep cannot continue as configured."""


@dataclass(frozen=True)
class WorkerSlot:
    """One non-overlapping GPU/port allocation reused by checkpoint tasks."""

    index: int
    devices: tuple[str, ...]
    main_process_port: int

    @property
    def visible_devices(self) -> str:
        return ",".join(self.devices)


@dataclass(frozen=True)
class SweepResources:
    project_root: Path
    canonical_launcher: Path
    official_evaluator: Path
    output_root: Path
    talkshow_root: Path
    feature_extractor: Path
    source_audio_root: Path | None
    smplx_path: Path | None
    watchdog: "GenerationWatchdogConfig"


@dataclass(frozen=True)
class GenerationWatchdogConfig:
    """Completion-aware lifecycle policy for one test subprocess group."""

    poll_seconds: float
    post_completion_grace_seconds: float
    terminate_grace_seconds: float
    hard_timeout_seconds: float


@dataclass(frozen=True)
class GenerationRunResult:
    """Lifecycle result for one completed TalkSHOW generation."""

    completion_marker_seen: bool
    watchdog_triggered: bool


class PreflightCoordinator:
    """Serialize the one-time contract scan before parallel inference starts."""

    def __init__(self, mode: str):
        self.mode = mode
        self._lock = Lock()
        self._completed = False

    def run_generation(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        completion_marker_path: Path,
        completion_run_id: str,
        watchdog: GenerationWatchdogConfig,
    ) -> GenerationRunResult:
        if self.mode == "always":
            run_environment = dict(environment)
            run_environment["RUN_CONTRACT_PREFLIGHT"] = "1"
            return _run_generation_command(
                command,
                cwd=cwd,
                environment=run_environment,
                completion_marker_path=completion_marker_path,
                completion_run_id=completion_run_id,
                watchdog=watchdog,
            )
        if self.mode == "never":
            run_environment = dict(environment)
            run_environment["RUN_CONTRACT_PREFLIGHT"] = "0"
            return _run_generation_command(
                command,
                cwd=cwd,
                environment=run_environment,
                completion_marker_path=completion_marker_path,
                completion_run_id=completion_run_id,
                watchdog=watchdog,
            )

        # Only the first worker enters the expensive strict scan. Other
        # workers wait at this lock, then start with preflight disabled.
        with self._lock:
            if not self._completed:
                run_environment = dict(environment)
                run_environment["RUN_CONTRACT_PREFLIGHT"] = "1"
                result = _run_generation_command(
                    command,
                    cwd=cwd,
                    environment=run_environment,
                    completion_marker_path=completion_marker_path,
                    completion_run_id=completion_run_id,
                    watchdog=watchdog,
                )
                self._completed = True
                return result

        run_environment = dict(environment)
        run_environment["RUN_CONTRACT_PREFLIGHT"] = "0"
        return _run_generation_command(
            command,
            cwd=cwd,
            environment=run_environment,
            completion_marker_path=completion_marker_path,
            completion_run_id=completion_run_id,
            watchdog=watchdog,
        )


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return value


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _port(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return value


def _non_negative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(
            "value must be a finite non-negative number"
        )
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be a finite positive number"
        )
    return value


def _absolute_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise SweepError(f"unsupported checkpoint name: {path.name}")
    epoch = int(match.group("epoch"))
    step = int(match.group("step") or -1)
    return epoch, step, path.name


def collect_checkpoints(raw_path: str | Path) -> tuple[list[Path], Path]:
    """Collect direct child checkpoints in numeric epoch/step order."""

    target = _absolute_path(raw_path)
    if target.is_file():
        if target.suffix != ".pth":
            raise SweepError(f"checkpoint is not a .pth file: {target}")
        if CHECKPOINT_PATTERN.fullmatch(target.name) is None:
            raise SweepError(
                "checkpoint filename must match epoch_<N>.pth or "
                f"epoch_<N>_step_<N>*.pth: {target.name}"
            )
        return [target], target.parent
    if not target.is_dir():
        raise SweepError(
            f"checkpoint path is neither a file nor a directory: {target}"
        )

    all_pth = sorted(path for path in target.glob("*.pth") if path.is_file())
    checkpoints = [
        path
        for path in all_pth
        if CHECKPOINT_PATTERN.fullmatch(path.name) is not None
    ]
    if not checkpoints:
        ignored = ", ".join(path.name for path in all_pth)
        detail = f" Ignored: {ignored}" if ignored else ""
        raise SweepError(
            "no epoch_<N>[_step_<N>]*.pth checkpoints found directly in "
            f"{target}.{detail}"
        )
    checkpoints.sort(key=_checkpoint_sort_key)
    return checkpoints, target


def filter_checkpoints_by_epoch(
    checkpoints: Sequence[Path],
    epoch_range: Sequence[int] | None,
) -> list[Path]:
    """Keep every checkpoint whose epoch is in the inclusive range."""

    if epoch_range is None:
        return list(checkpoints)
    start_epoch, end_epoch = epoch_range
    if start_epoch > end_epoch:
        raise SweepError(
            "epoch range start must not exceed its end: "
            f"{start_epoch} > {end_epoch}"
        )
    selected = [
        checkpoint
        for checkpoint in checkpoints
        if start_epoch <= _checkpoint_sort_key(checkpoint)[0] <= end_epoch
    ]
    if selected:
        return selected

    available_epochs = [
        _checkpoint_sort_key(checkpoint)[0] for checkpoint in checkpoints
    ]
    available_detail = (
        f"{min(available_epochs)}–{max(available_epochs)}"
        if available_epochs
        else "none"
    )
    raise SweepError(
        "no checkpoints fall in the inclusive epoch range "
        f"[{start_epoch}, {end_epoch}]; available epoch range: "
        f"{available_detail}"
    )


def _require_file(raw_path: str | Path, *, label: str) -> Path:
    path = _absolute_path(raw_path)
    if not path.is_file():
        raise SweepError(f"{label} does not exist or is not a file: {path}")
    return path


def _require_directory(raw_path: str | Path, *, label: str) -> Path:
    path = _absolute_path(raw_path)
    if not path.is_dir():
        raise SweepError(
            f"{label} does not exist or is not a directory: {path}"
        )
    return path


def _validated_base_environment(
    args: argparse.Namespace,
) -> tuple[dict[str, str], tuple[str, ...]]:
    environment = os.environ.copy()
    for name in REQUIRED_GENERATION_ENV:
        value = environment.get(name, "").strip()
        if not value:
            raise SweepError(
                f"missing required TalkSHOW environment variable: {name}"
            )
        if not Path(value).is_absolute():
            raise SweepError(f"{name} must be an absolute path: {value}")

    visible_devices = (
        args.cuda_visible_devices
        if args.cuda_visible_devices is not None
        else environment.get("CUDA_VISIBLE_DEVICES")
    )
    if visible_devices is None or not visible_devices.strip():
        required_device_count = (
            args.parallel_workers * args.gpus_per_worker
        )
        visible_devices = ",".join(
            str(index) for index in range(required_device_count)
        )
    device_items = tuple(
        item.strip() for item in visible_devices.split(",") if item.strip()
    )
    required_device_count = args.parallel_workers * args.gpus_per_worker
    if len(device_items) != required_device_count:
        raise SweepError(
            "CUDA_VISIBLE_DEVICES must contain exactly "
            f"{required_device_count} devices for "
            f"--parallel-workers={args.parallel_workers} and "
            f"--gpus-per-worker={args.gpus_per_worker}; "
            f"got {visible_devices!r}"
        )
    if args.main_process_port + args.parallel_workers - 1 > 65535:
        raise SweepError(
            "--main-process-port plus the worker count exceeds 65535: "
            f"{args.main_process_port} + {args.parallel_workers} - 1"
        )

    environment.update(
        {
            "TALKSHOW_PROFILE": args.profile,
            "NUM_MACHINES": "1",
            "TRAIN_BATCH_SIZE": "72",
            "NUM_EPOCHS": "1000",
            "NUM_WORKERS": str(args.num_workers),
            "REPORT_TO": "tensorboard",
            "TALKSHOW_MODE": "test",
            "TALKSHOW_CFG_AUDIO_SCALE": str(args.cfg_audio_scale),
            "TALKSHOW_CFG_HISTORY_SCALE": str(args.cfg_history_scale),
            "TALKSHOW_CFG_MODE": args.cfg_mode,
        }
    )
    # Validation is intentionally absent in test-only mode.  Do not let stale
    # shell variables impose an irrelevant validation contract on a sweep.
    environment.pop("TALKSHOW_VALIDATION_LMDB", None)
    environment.pop("TALKSHOW_VALIDATION_SPLIT", None)
    environment.pop("TALKSHOW_TEST_ONLY", None)
    environment.pop("TALKSHOW_VALIDATION_ONLY", None)
    return environment, device_items


def build_worker_slots(
    args: argparse.Namespace,
    visible_devices: Sequence[str],
) -> tuple[WorkerSlot, ...]:
    """Partition the global visible-device list into fixed worker slots."""

    slots = []
    for worker_index in range(args.parallel_workers):
        start = worker_index * args.gpus_per_worker
        end = start + args.gpus_per_worker
        devices = tuple(visible_devices[start:end])
        if len(devices) != args.gpus_per_worker:
            raise SweepError(
                f"worker {worker_index} received {len(devices)} devices, "
                f"expected {args.gpus_per_worker}"
            )
        slots.append(
            WorkerSlot(
                index=worker_index,
                devices=devices,
                main_process_port=args.main_process_port + worker_index,
            )
        )
    return tuple(slots)


def _worker_environment(
    base_environment: Mapping[str, str],
    *,
    slot: WorkerSlot,
    args: argparse.Namespace,
) -> dict[str, str]:
    environment = dict(base_environment)
    for distributed_name in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(distributed_name, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": slot.visible_devices,
            "NUM_PROCESSES": str(args.gpus_per_worker),
            "MAIN_PROCESS_PORT": str(slot.main_process_port),
        }
    )
    if args.gpus_per_worker == 1:
        # Accelerate single-process launch does not consistently inject these,
        # while main.py reads them directly. Keep this scoped to the
        # TalkSHOW sweep and use the slot's private rendezvous port.
        environment.update(
            {
                "LOCAL_RANK": "0",
                "RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(slot.main_process_port),
            }
        )
    return environment


def build_generation_command(
    *,
    canonical_launcher: Path,
    checkpoint: Path,
    seed: int,
    hydra_run_dir: Path,
) -> list[str]:
    """Build one canonical TalkSHOW test-only generation command."""

    return [
        "bash",
        str(canonical_launcher),
        f"model.load_from={checkpoint}",
        f"train.seed={seed}",
        f"++hydra.run.dir={hydra_run_dir}",
    ]


def build_official_eval_command(
    *,
    evaluator: Path,
    sample0_work_dir: Path,
    sample1_work_dir: Path,
    talkshow_root: Path,
    feature_extractor: Path,
    output_json: Path,
    source_audio_root: Path | None,
    smplx_path: Path | None,
    device: str,
    expected_feature_extractor_sha256: str | None,
    expected_smplx_sha256: str | None,
    allow_unpinned_talkshow: bool,
    allow_source_subset: bool,
) -> list[str]:
    """Build the official B=2 body/face evaluator command."""

    sample0_npz = sample0_work_dir / "npz" / "test"
    sample1_npz = sample1_work_dir / "npz" / "test"
    command = [
        sys.executable,
        str(evaluator),
        "--pred-dir",
        str(sample0_npz),
        "--pred-dir",
        str(sample1_npz),
        "--gt-dir",
        str(sample0_npz),
    ]
    if source_audio_root is not None:
        command.extend(["--source-audio-root", str(source_audio_root)])
    else:
        command.extend(
            ["--audio-dir", str(sample0_work_dir / "wav" / "test")]
        )
    command.extend(
        [
            "--talkshow-root",
            str(talkshow_root),
            "--feature-extractor",
            str(feature_extractor),
            "--device",
            device,
            "--output",
            str(output_json),
        ]
    )
    if smplx_path is not None:
        command.extend(["--smplx-path", str(smplx_path)])
    if expected_feature_extractor_sha256:
        command.extend(
            [
                "--expected-feature-extractor-sha256",
                expected_feature_extractor_sha256,
            ]
        )
    if expected_smplx_sha256:
        command.extend(
            ["--expected-smplx-sha256", expected_smplx_sha256]
        )
    if allow_unpinned_talkshow:
        command.append("--allow-unpinned-talkshow")
    if allow_source_subset:
        command.append("--allow-source-subset")
    return command


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    print(f"[run] {shlex.join(command)}", flush=True)
    subprocess.run(
        list(command),
        cwd=str(cwd),
        env=dict(environment),
        check=True,
    )


def _process_group_exists(process: subprocess.Popen[Any]) -> bool:
    """Return whether the supervised process or a descendant still exists."""

    process.poll()
    if os.name != "posix":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _send_process_group_signal(
    process: subprocess.Popen[Any],
    signum: int,
) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        elif process.poll() is None and signum == signal.SIGTERM:
            process.terminate()
        elif process.poll() is None:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def _terminate_and_reap_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float,
) -> tuple[int, bool]:
    """Terminate the complete Accelerate process group and reap its leader."""

    escalated = False
    if _process_group_exists(process):
        _send_process_group_signal(process, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        poll_seconds = min(0.1, max(0.001, grace_seconds / 10.0))
        while (
            time.monotonic() < deadline
            and _process_group_exists(process)
        ):
            time.sleep(poll_seconds)

        if _process_group_exists(process):
            escalated = True
            _send_process_group_signal(process, signal.SIGKILL)
            kill_deadline = time.monotonic() + min(
                2.0,
                max(0.1, grace_seconds),
            )
            while (
                time.monotonic() < kill_deadline
                and _process_group_exists(process)
            ):
                time.sleep(min(poll_seconds, 0.1))

    try:
        returncode = process.wait(timeout=max(1.0, grace_seconds + 1.0))
    except subprocess.TimeoutExpired as exc:
        _send_process_group_signal(process, signal.SIGKILL)
        try:
            returncode = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as final_exc:
            raise SweepError(
                "Accelerate launcher leader survived SIGKILL: "
                f"pid={process.pid}"
            ) from final_exc
        escalated = True

    if _process_group_exists(process):
        raise SweepError(
            "Accelerate subprocess group still exists after SIGKILL: "
            f"pgid={process.pid}"
        )
    return returncode, escalated


def _run_generation_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    completion_marker_path: Path,
    completion_run_id: str,
    watchdog: GenerationWatchdogConfig,
) -> GenerationRunResult:
    """Run one test with a completion-aware process-group watchdog."""

    token = uuid.uuid4().hex
    run_environment = dict(environment)
    run_environment.update(
        {
            "SEED_SWEEP_INFERENCE_COMPLETION_MARKER": str(
                completion_marker_path.resolve()
            ),
            "SEED_SWEEP_INFERENCE_COMPLETION_RUN_ID": completion_run_id,
            "SEED_SWEEP_INFERENCE_COMPLETION_TOKEN": token,
        }
    )
    print(f"[run] {shlex.join(command)}", flush=True)
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=run_environment,
            start_new_session=(os.name == "posix"),
        )
        started_at = time.monotonic()
        marker_seen_at: float | None = None

        while True:
            now = time.monotonic()
            if marker_seen_at is None:
                marker = read_completion_marker(
                    completion_marker_path,
                    expected_run_id=completion_run_id,
                    expected_token=token,
                    expected_producer="main.py",
                )
                if marker is not None:
                    marker_seen_at = now
                    print(
                        "[watchdog] TalkSHOW test completion marker received; "
                        "waiting "
                        f"{watchdog.post_completion_grace_seconds:g}s for "
                        "normal process-group exit.",
                        flush=True,
                    )

            returncode = process.poll()
            if returncode is not None:
                leaked_group = _process_group_exists(process)
                if leaked_group:
                    _, escalated = _terminate_and_reap_process_group(
                        process,
                        grace_seconds=watchdog.terminate_grace_seconds,
                    )
                    print(
                        "[watchdog] Accelerate launcher exited but left child "
                        "processes; cleaned its process group "
                        f"(escalated_to_sigkill={escalated}).",
                        flush=True,
                    )
                if returncode != 0:
                    raise subprocess.CalledProcessError(
                        returncode,
                        list(command),
                    )
                return GenerationRunResult(
                    completion_marker_seen=marker_seen_at is not None,
                    watchdog_triggered=leaked_group,
                )

            if (
                marker_seen_at is not None
                and now - marker_seen_at
                >= watchdog.post_completion_grace_seconds
            ):
                returncode, escalated = _terminate_and_reap_process_group(
                    process,
                    grace_seconds=watchdog.terminate_grace_seconds,
                )
                print(
                    "[watchdog] TalkSHOW test completed but Accelerate/DDP "
                    "did not exit; terminated its complete process group "
                    f"(returncode={returncode}, "
                    f"escalated_to_sigkill={escalated}).",
                    flush=True,
                )
                return GenerationRunResult(
                    completion_marker_seen=True,
                    watchdog_triggered=True,
                )

            if (
                watchdog.hard_timeout_seconds > 0.0
                and now - started_at >= watchdog.hard_timeout_seconds
            ):
                _terminate_and_reap_process_group(
                    process,
                    grace_seconds=watchdog.terminate_grace_seconds,
                )
                raise subprocess.TimeoutExpired(
                    list(command),
                    watchdog.hard_timeout_seconds,
                )

            time.sleep(min(watchdog.poll_seconds, 1.0))
    except BaseException:
        if process is not None and _process_group_exists(process):
            _terminate_and_reap_process_group(
                process,
                grace_seconds=watchdog.terminate_grace_seconds,
            )
        raise


def _assert_generation_output(work_dir: Path, *, require_gt: bool) -> None:
    npz_dir = work_dir / "npz" / "test"
    if not npz_dir.is_dir():
        raise SweepError(
            f"TalkSHOW test produced no npz/test directory: {npz_dir}"
        )
    if next(npz_dir.glob("res_*.npz"), None) is None:
        raise SweepError(
            f"TalkSHOW test produced no res_*.npz files: {npz_dir}"
        )
    if require_gt and next(npz_dir.glob("gt_*.npz"), None) is None:
        raise SweepError(
            f"TalkSHOW test produced no gt_*.npz files: {npz_dir}"
        )


def load_official_metrics(report_path: Path) -> dict[str, float]:
    """Read and validate the metric mapping emitted by the official evaluator."""

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepError(
            f"cannot read official metric report {report_path}: {exc}"
        ) from exc
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SweepError(
            f"official metric report has no metrics mapping: {report_path}"
        )

    normalized: dict[str, float] = {}
    for name in OFFICIAL_METRIC_NAMES:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise SweepError(
                f"official metric {name!r} is missing or invalid in "
                f"{report_path}"
            ) from exc
        if not math.isfinite(value):
            raise SweepError(
                f"official metric {name!r} is not finite in {report_path}: "
                f"{value}"
            )
        normalized[name] = value
    return normalized


def best_successful_row(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    successful = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("fgd") not in ("", None)
    ]
    if not successful:
        return None
    return min(successful, key=lambda row: float(row["fgd"]))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in CSV_FIELDS}
            for row in rows
        )
    os.replace(temporary, path)


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> None:
    best = best_successful_row(rows)
    summary = {
        "protocol": "talkshow_official_body_face_b2_checkpoint_sweep_v1",
        "selection_metric": "fgd",
        "selection_direction": "min",
        "profile": args.profile,
        "seeds": list(args.seeds),
        "parallel_workers": args.parallel_workers,
        "gpus_per_worker": args.gpus_per_worker,
        "num_workers_per_process": args.num_workers,
        "epoch_range": (
            list(args.epoch_range) if args.epoch_range is not None else None
        ),
        "generation_watchdog": {
            "poll_seconds": args.watchdog_poll_seconds,
            "post_completion_grace_seconds": (
                args.watchdog_post_completion_grace_seconds
            ),
            "terminate_grace_seconds": (
                args.watchdog_terminate_grace_seconds
            ),
            "hard_timeout_seconds": args.generation_hard_timeout_seconds,
        },
        "checkpoint_path": str(_absolute_path(args.checkpoint_path)),
        "output_root": str(output_root),
        "best_checkpoint": best.get("checkpoint") if best else None,
        "best_fgd": float(best["fgd"]) if best else None,
        "rows": list(rows),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _persist_results(
    *,
    csv_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> None:
    _write_csv(csv_path, rows)
    _write_summary(
        summary_path,
        args=args,
        rows=rows,
        output_root=output_root,
    )


def _initial_checkpoint_row(
    *,
    checkpoint: Path,
    slot: WorkerSlot,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[
    dict[str, Any],
    tuple[Path, Path],
    tuple[Path, Path],
    Path,
]:
    epoch, step, _ = _checkpoint_sort_key(checkpoint)
    checkpoint_output = output_root / checkpoint.stem
    sample_run_dirs = (
        checkpoint_output / f"sample0_seed{args.seeds[0]}",
        checkpoint_output / f"sample1_seed{args.seeds[1]}",
    )
    sample_work_dirs = tuple(path / "output" for path in sample_run_dirs)
    metrics_json = checkpoint_output / "talkshow_official_metrics.json"
    row: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "step": "" if step < 0 else step,
        "worker": slot.index,
        "cuda_visible_devices": slot.visible_devices,
        "main_process_port": slot.main_process_port,
        "watchdog_terminated_samples": "",
        **{name: "" for name in OFFICIAL_METRIC_NAMES},
        "status": "ok",
        "error": "",
        "sample0_dir": str(sample_work_dirs[0]),
        "sample1_dir": str(sample_work_dirs[1]),
        "metrics_json": str(metrics_json),
    }
    return row, sample_run_dirs, sample_work_dirs, metrics_json


def _evaluate_checkpoint(
    *,
    ordinal: int,
    total: int,
    checkpoint: Path,
    slot: WorkerSlot,
    args: argparse.Namespace,
    resources: SweepResources,
    base_environment: Mapping[str, str],
    preflight: PreflightCoordinator,
) -> dict[str, Any]:
    """Run both samples and official metrics as one atomic worker task."""

    row, sample_run_dirs, sample_work_dirs, metrics_json = (
        _initial_checkpoint_row(
            checkpoint=checkpoint,
            slot=slot,
            args=args,
            output_root=resources.output_root,
        )
    )
    worker_environment = _worker_environment(
        base_environment,
        slot=slot,
        args=args,
    )
    worker_label = (
        f"worker={slot.index} gpu={slot.visible_devices} "
        f"port={slot.main_process_port}"
    )
    print(
        f"[{ordinal}/{total}] [{worker_label}] "
        f"checkpoint={checkpoint.name}",
        flush=True,
    )
    watchdog_terminated_samples: list[str] = []

    try:
        for sample_index, (seed, hydra_run_dir) in enumerate(
            zip(args.seeds, sample_run_dirs)
        ):
            run_environment = worker_environment.copy()
            run_environment["RUN_NAME"] = (
                "talkshow_pth_sweep_"
                f"{checkpoint.stem}_sample{sample_index}_worker{slot.index}"
            )
            generation_command = build_generation_command(
                canonical_launcher=resources.canonical_launcher,
                checkpoint=checkpoint,
                seed=seed,
                hydra_run_dir=hydra_run_dir,
            )
            generation_result = preflight.run_generation(
                generation_command,
                cwd=resources.project_root,
                environment=run_environment,
                completion_marker_path=(
                    hydra_run_dir / ".inference_complete.json"
                ),
                completion_run_id=uuid.uuid4().hex,
                watchdog=resources.watchdog,
            )
            if generation_result.watchdog_triggered:
                watchdog_terminated_samples.append(f"sample{sample_index}")
                row["watchdog_terminated_samples"] = ",".join(
                    watchdog_terminated_samples
                )
            _assert_generation_output(
                sample_work_dirs[sample_index],
                require_gt=(sample_index == 0),
            )

        official_command = build_official_eval_command(
            evaluator=resources.official_evaluator,
            sample0_work_dir=sample_work_dirs[0],
            sample1_work_dir=sample_work_dirs[1],
            talkshow_root=resources.talkshow_root,
            feature_extractor=resources.feature_extractor,
            output_json=metrics_json,
            source_audio_root=resources.source_audio_root,
            smplx_path=resources.smplx_path,
            device=args.official_device,
            expected_feature_extractor_sha256=(
                args.expected_feature_extractor_sha256
            ),
            expected_smplx_sha256=args.expected_smplx_sha256,
            allow_unpinned_talkshow=args.allow_unpinned_talkshow,
            allow_source_subset=args.allow_source_subset,
        )
        _run_command(
            official_command,
            cwd=resources.project_root,
            environment=worker_environment,
        )
        row.update(load_official_metrics(metrics_json))
        print(
            f"[ok] [{worker_label}] {checkpoint.name}: "
            f"FGD={float(row['fgd']):.6f}",
            flush=True,
        )
    except (
        OSError,
        SweepError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        print(
            f"[failed] [{worker_label}] {checkpoint.name}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    return row


def run_sweep(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    canonical_launcher = (
        project_root
        / "train_scripts"
        / "run_talkshow.sh"
    )
    official_evaluator = project_root / "scripts" / "talkshow_official_eval.py"
    _require_file(canonical_launcher, label="canonical TalkSHOW launcher")
    _require_file(official_evaluator, label="TalkSHOW official evaluator")

    checkpoints, checkpoint_root = collect_checkpoints(args.checkpoint_path)
    checkpoints = filter_checkpoints_by_epoch(
        checkpoints,
        args.epoch_range,
    )
    talkshow_root = _require_directory(
        args.talkshow_root,
        label="TalkSHOW repository root",
    )
    feature_extractor = _require_file(
        args.feature_extractor,
        label="TalkSHOW feature extractor",
    )
    source_audio_root = (
        _require_directory(
            args.source_audio_root,
            label="TalkSHOW source-audio split view",
        )
        if args.source_audio_root
        else None
    )
    smplx_path = (
        _require_directory(args.smplx_path, label="SMPL-X path")
        if args.smplx_path
        else None
    )
    if args.seeds[0] == args.seeds[1]:
        raise SweepError("the two TalkSHOW generation seeds must be different")

    base_environment, visible_devices = _validated_base_environment(args)
    worker_slots = build_worker_slots(args, visible_devices)
    run_id = args.run_id or (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    output_root = (
        _absolute_path(args.output_dir)
        if args.output_dir
        else checkpoint_root / f"_talkshow_eval_{run_id}"
    )
    if output_root.exists():
        raise SweepError(
            "sweep output directory already exists; choose a new "
            f"--output-dir or --run-id: {output_root}"
        )
    output_root.mkdir(parents=True)

    csv_path = output_root / "talkshow_checkpoint_metrics.csv"
    summary_path = output_root / "talkshow_checkpoint_metrics.json"
    rows: list[dict[str, Any]] = []
    checkpoint_order = {
        str(checkpoint): index
        for index, checkpoint in enumerate(checkpoints)
    }
    resources = SweepResources(
        project_root=project_root,
        canonical_launcher=canonical_launcher,
        official_evaluator=official_evaluator,
        output_root=output_root,
        talkshow_root=talkshow_root,
        feature_extractor=feature_extractor,
        source_audio_root=source_audio_root,
        smplx_path=smplx_path,
        watchdog=GenerationWatchdogConfig(
            poll_seconds=args.watchdog_poll_seconds,
            post_completion_grace_seconds=(
                args.watchdog_post_completion_grace_seconds
            ),
            terminate_grace_seconds=(
                args.watchdog_terminate_grace_seconds
            ),
            hard_timeout_seconds=args.generation_hard_timeout_seconds,
        ),
    )
    preflight = PreflightCoordinator(args.preflight)

    print(
        f"TalkSHOW checkpoint sweep: profile={args.profile}, "
        f"checkpoints={len(checkpoints)}, seeds={tuple(args.seeds)}, "
        f"parallel_workers={args.parallel_workers}, "
        f"gpus_per_worker={args.gpus_per_worker}, "
        "epoch_range="
        f"{tuple(args.epoch_range) if args.epoch_range else 'all'}, "
        f"output={output_root}",
        flush=True,
    )
    for slot in worker_slots:
        print(
            f"  worker {slot.index}: CUDA_VISIBLE_DEVICES="
            f"{slot.visible_devices}, port={slot.main_process_port}",
            flush=True,
        )

    checkpoint_iterator = iter(enumerate(checkpoints, start=1))
    pending: dict[Future, tuple[int, Path, WorkerSlot]] = {}
    stop_scheduling = False

    def submit_next(
        executor: ThreadPoolExecutor,
        slot: WorkerSlot,
    ) -> bool:
        try:
            ordinal, checkpoint = next(checkpoint_iterator)
        except StopIteration:
            return False
        future = executor.submit(
            _evaluate_checkpoint,
            ordinal=ordinal,
            total=len(checkpoints),
            checkpoint=checkpoint,
            slot=slot,
            args=args,
            resources=resources,
            base_environment=base_environment,
            preflight=preflight,
        )
        pending[future] = (ordinal, checkpoint, slot)
        return True

    with ThreadPoolExecutor(
        max_workers=args.parallel_workers,
        thread_name_prefix="talkshow-ckpt",
    ) as executor:
        for slot in worker_slots:
            if not submit_next(executor, slot):
                break

        while pending:
            completed, _ = wait(
                tuple(pending),
                return_when=FIRST_COMPLETED,
            )
            reusable_slots: list[WorkerSlot] = []
            for future in completed:
                ordinal, checkpoint, slot = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row, _, _, _ = _initial_checkpoint_row(
                        checkpoint=checkpoint,
                        slot=slot,
                        args=args,
                        output_root=output_root,
                    )
                    row["status"] = "failed"
                    row["error"] = f"unexpected worker failure: {exc!r}"
                    print(
                        f"[failed] [worker={slot.index}] "
                        f"{checkpoint.name}: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )

                rows.append(row)
                rows.sort(
                    key=lambda item: checkpoint_order[item["checkpoint"]]
                )
                _persist_results(
                    csv_path=csv_path,
                    summary_path=summary_path,
                    args=args,
                    rows=rows,
                    output_root=output_root,
                )
                if args.fail_fast and row["status"] != "ok":
                    stop_scheduling = True
                reusable_slots.append(slot)

            if not stop_scheduling:
                for slot in reusable_slots:
                    submit_next(executor, slot)

    best = best_successful_row(rows)
    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved JSON: {summary_path}", flush=True)
    if best is None:
        print(
            "No checkpoint completed official TalkSHOW evaluation.",
            file=sys.stderr,
        )
        return 1
    print(
        "Best checkpoint by official TalkSHOW FGD: "
        f"{best['checkpoint']} (FGD={float(best['fgd']):.6f})",
        flush=True,
    )
    failed_count = sum(row["status"] != "ok" for row in rows)
    if failed_count:
        print(
            f"Warning: {failed_count}/{len(rows)} checkpoints failed; "
            "inspect the CSV error column.",
            file=sys.stderr,
        )
    if stop_scheduling:
        print(
            "Fail-fast stopped scheduling new checkpoints after a failure.",
            file=sys.stderr,
        )
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_path",
        help=(
            "A checkpoint directory or one epoch_<N>[_step_<N>]*.pth file. "
            "Directory traversal is non-recursive."
        ),
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=PROFILE_CHOICES,
        help="TalkSHOW sequence/block profile used to train every checkpoint.",
    )
    parser.add_argument(
        "--talkshow-root",
        required=True,
        help="Pinned TalkSHOW source repository used by the official evaluator.",
    )
    parser.add_argument(
        "--feature-extractor",
        required=True,
        help="Official TalkSHOW experiments/feature_extractor.pth.",
    )
    parser.add_argument(
        "--source-audio-root",
        help=(
            "Recommended TalkSHOW split-view root containing original test "
            "WAVs. If omitted, generated aligned wav/test files are used."
        ),
    )
    parser.add_argument(
        "--smplx-path",
        help=(
            "Path passed to smplx.create; defaults inside the evaluator to "
            "<talkshow-root>/visualise."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_non_negative_int,
        nargs=2,
        default=(1001, 2002),
        metavar=("SAMPLE0", "SAMPLE1"),
        help="Two distinct generation seeds (default: 1001 2002).",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "New sweep output directory. Defaults to "
            "<checkpoint-dir>/_talkshow_eval_<run-id>."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Optional stable label used by the default output directory.",
    )
    parser.add_argument(
        "--epoch-range",
        type=_non_negative_int,
        nargs=2,
        metavar=("START", "END"),
        help=(
            "Evaluate only checkpoints in the inclusive epoch range "
            "[START, END]. All checkpoints are used when omitted."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=_positive_int,
        default=1,
        help=(
            "Concurrent end-to-end checkpoint workers. Each worker performs "
            "both samples and official evaluation (default: 1)."
        ),
    )
    parser.add_argument(
        "--gpus-per-worker",
        "--num-processes",
        dest="gpus_per_worker",
        type=_positive_int,
        default=4,
        help=(
            "GPUs/Accelerate processes assigned to each checkpoint worker "
            "(default: 4). --num-processes is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help=(
            "Global comma-separated device pool. Its size must equal "
            "parallel_workers * gpus_per_worker."
        ),
    )
    parser.add_argument(
        "--main-process-port",
        type=_port,
        default=29500,
        help=(
            "Base Accelerate rendezvous port; worker i uses base+i "
            "(default: 29500)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=_non_negative_int,
        default=8,
        help="Test dataloader workers per GPU process (default: 8).",
    )
    parser.add_argument(
        "--preflight",
        choices=("once", "always", "never"),
        default="once",
        help=(
            "Run the strict test-LMDB/asset preflight once, for every sample, "
            "or never (default: once)."
        ),
    )
    parser.add_argument(
        "--watchdog-poll-seconds",
        type=_positive_float,
        default=1.0,
        help="Completion-marker polling interval (default: 1 second).",
    )
    parser.add_argument(
        "--watchdog-post-completion-grace-seconds",
        type=_non_negative_float,
        default=10.0,
        help=(
            "Seconds to wait for normal Accelerate/DDP exit after test "
            "completion is proven (default: 10)."
        ),
    )
    parser.add_argument(
        "--watchdog-terminate-grace-seconds",
        type=_positive_float,
        default=20.0,
        help=(
            "Seconds between process-group SIGTERM and SIGKILL "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--generation-hard-timeout-seconds",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Optional per-sample hard timeout before completion; 0 disables "
            "it (default: 0)."
        ),
    )
    parser.add_argument(
        "--cfg-audio-scale",
        type=_non_negative_float,
        default=1.5,
        help="Inference audio CFG scale (default: 1.5).",
    )
    parser.add_argument(
        "--cfg-history-scale",
        type=_non_negative_float,
        default=1.5,
        help="Inference history CFG scale (default: 1.5).",
    )
    parser.add_argument(
        "--cfg-mode",
        choices=CFG_MODE_CHOICES,
        default="interaction",
        help="Inference CFG composition mode (default: interaction).",
    )
    parser.add_argument(
        "--official-device",
        default="cuda:0",
        help="Device used by talkshow_official_eval.py (default: cuda:0).",
    )
    parser.add_argument(
        "--expected-feature-extractor-sha256",
        help="Optional expected SHA-256 for feature_extractor.pth.",
    )
    parser.add_argument(
        "--expected-smplx-sha256",
        help="Optional expected SHA-256 for the loaded neutral SMPL-X asset.",
    )
    parser.add_argument(
        "--allow-unpinned-talkshow",
        action="store_true",
        help="Forward the official evaluator's source-hash escape hatch.",
    )
    parser.add_argument(
        "--allow-source-subset",
        action="store_true",
        help="Allow incomplete source-audio prediction coverage.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed checkpoint instead of continuing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run_sweep(args)
    except (OSError, SweepError) as exc:
        parser.exit(2, f"[abort] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
