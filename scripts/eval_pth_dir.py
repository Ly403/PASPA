r"""批量生成并评估 checkpoint，按最低 FGD 汇总最佳权重。

默认仍是原来的串行模式：一个 checkpoint 跑完生成和指标后再跑下一个；
若未设置 ``eval.num_processes``，单 worker 会继续使用
``CUDA_VISIBLE_DEVICES`` 中的全部 GPU。

八张卡、每张卡同时评估一个 checkpoint 的典型命令如下（其余模型、数据和
evaluator 参数继续按原实验传入）：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   python scripts/eval_pth_dir.py \
     model.load_from=/absolute/path/to/checkpoints \
     ++eval.parallel_workers=8 \
     ++eval.num_processes=1 \
     ++eval.main_process_port=29500 \
     ++eval.device=cuda:0 \
     ++eval.cleanup_non_best=no \
     ...其余与单卡测试相同的 Hydra 参数

这里 ``parallel_workers`` 是同时处理的 checkpoint 数，
``num_processes`` 是每个 checkpoint 使用的 GPU 数。设备总数必须等于
二者乘积。每个 worker 把“模型生成 + evaluator 计算指标”作为一个完整任务，
主线程增量写 CSV，某个 slot 完成后再领取下一个 checkpoint。

普通直接运行默认启用 completion-marker watchdog：所有 rank 完成
``trainer.test()`` 后若 Accelerate/DDP 仍未退出，等待 10 秒便清理整个
子进程组并继续评估。可用下列参数调整：

* ``++eval.watchdog_enabled=true|false``
* ``++eval.watchdog_post_completion_grace_seconds=10``
* ``++eval.watchdog_terminate_grace_seconds=20``
* ``++eval.watchdog_hard_timeout_seconds=0``（0 表示完成前不设硬超时）

单 checkpoint 的 workspace 和 seed-sweep completion-marker 合同保持不变。
目录模式则为每个 checkpoint 创建独立子目录，避免并行生成互相覆盖。
"""

import csv
import json
import math
import os
import re
import select
import signal
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import current_thread, main_thread
from typing import Any, Mapping, Sequence

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

try:
    from seed_sweep.completion_marker import (
        read_completion_marker,
        write_completion_marker,
    )
except ImportError:
    from scripts.seed_sweep.completion_marker import (
        read_completion_marker,
        write_completion_marker,
    )


# configs/config.yaml uses hydra.run.dir=${calc_work_dir:${resume_from}}
# The resolver is normally registered in main.py at import time.
# This script has its own Hydra entrypoint, so we provide a compatible fallback.
if not OmegaConf.has_resolver("calc_work_dir"):
    OmegaConf.register_new_resolver(
        "calc_work_dir",
        lambda resume_path=None: str((Path.cwd() / "outputs" / "eval_pth_dir").resolve()),
    )


def _resolve_main_process_port(raw_port: Any, num_machines: int) -> int:
    """Choose a rendezvous port without a bind-close-rebind race."""

    if raw_port is None:
        if num_machines != 1:
            raise ValueError(
                "Multi-machine evaluation must set eval.main_process_port "
                "to the same non-zero port on every machine"
            )
        # Accelerate >= 1.9 and PyTorch TCPStore support port 0 for automatic
        # allocation.  For the seed sweep's world_size=1 workers, TCPStore owns
        # the bind directly, so allocation and listening are one atomic step.
        return 0

    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"eval.main_process_port must be an integer, got {raw_port!r}"
        ) from exc
    if not 0 <= port <= 65535:
        raise ValueError(
            f"eval.main_process_port must be between 0 and 65535, got {port}"
        )
    if num_machines != 1 and port == 0:
        raise ValueError(
            "Multi-machine evaluation cannot use eval.main_process_port=0; "
            "set one shared non-zero port"
        )
    return port


def _as_abs_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if not p.is_absolute():
        p = Path(get_original_cwd()) / p
    return p.resolve()


# 匹配 epoch_{N}_step_{N}... 格式的 pth 文件名
_CKPT_PATTERN = re.compile(r"^epoch_(\d+)_step_(\d+)")
_CSV_FIELDNAMES = (
    "checkpoint",
    "fgd",
    "bc",
    "l1",
    "lvd",
    "mse",
    "style_swd",
    "style_acf_error",
    "status",
    "error",
)


@dataclass(frozen=True)
class _WorkerSlot:
    """A non-overlapping GPU allocation reused by checkpoint tasks."""

    index: int
    devices: tuple[str, ...]
    main_process_port: int

    @property
    def visible_devices(self) -> str:
        return ",".join(self.devices)


@dataclass(frozen=True)
class _WatchdogConfig:
    """Completion-aware lifecycle policy for one inference process group."""

    enabled: bool
    poll_seconds: float
    post_completion_grace_seconds: float
    terminate_grace_seconds: float
    hard_timeout_seconds: float


@dataclass(frozen=True)
class _EvalContext:
    """Immutable settings shared by checkpoint workers."""

    project_root: Path
    workspace_root: Path
    single_ckpt_mode: bool
    run_id: str
    forwarded_overrides: tuple[str, ...]
    main_config_args: tuple[str, ...]
    num_machines: int
    num_processes: int
    mixed_precision: str
    dynamo_backend: str
    eval_script_name: str
    eval_person: str
    eval_download_path: str | None
    eval_split_file: str | None
    eval_device: str | None
    eval_only_fgd: bool
    eval_style_metrics: bool
    emage_style_metrics_enabled: bool
    watchdog: _WatchdogConfig
    base_environment: Mapping[str, str]


def _parse_epoch(ckpt_path: Path) -> int | None:
    """从文件名中提取 epoch 数字，无法匹配则返回 None。"""
    m = _CKPT_PATTERN.match(ckpt_path.stem)
    return int(m.group(1)) if m else None


def _collect_checkpoints(load_from_value: str) -> tuple[list[Path], Path]:
    target = _as_abs_path(load_from_value)
    if target.is_file() and target.suffix == ".pth":
        return [target], target.parent
    if target.is_dir():
        all_pths = sorted(target.glob("*.pth"))
        # 只保留 epoch_{N}_step_{N}... 格式的 checkpoint
        ckpts = [p for p in all_pths if _CKPT_PATTERN.match(p.stem)]
        if not ckpts and all_pths:
            print(
                f"Warning: found {len(all_pths)} .pth file(s) but none match "
                f"'epoch_*_step_*' pattern. Skipped: {[p.name for p in all_pths]}"
            )
        return ckpts, target
    raise FileNotFoundError(f"model.load_from must be a .pth file or a directory, got: {target}")


def _infer_num_processes_from_visible_devices() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw == "":
        return 1
    devices = [d.strip() for d in raw.split(",") if d.strip() != ""]
    return max(1, len(devices))


def _visible_devices(
    raw_devices: Any,
    *,
    required_count: int,
) -> tuple[str, ...]:
    """Resolve and validate the global device pool used by all workers."""

    if raw_devices is None:
        raw_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_devices is None:
        devices: tuple[str, ...] = ()
    elif isinstance(raw_devices, str):
        devices = tuple(
            item.strip() for item in raw_devices.split(",") if item.strip()
        )
    elif isinstance(raw_devices, Sequence):
        devices = tuple(
            str(item).strip() for item in raw_devices if str(item).strip()
        )
    else:
        raise ValueError(
            "eval.cuda_visible_devices must be a comma-separated string or "
            f"a list, got {raw_devices!r}"
        )

    if not devices:
        devices = tuple(str(index) for index in range(required_count))
    if len(devices) != required_count:
        raise ValueError(
            "The GPU pool must contain exactly "
            f"{required_count} device(s) for eval.parallel_workers * "
            f"eval.num_processes, got {devices!r}. Set CUDA_VISIBLE_DEVICES "
            "or eval.cuda_visible_devices explicitly."
        )
    if len(set(devices)) != len(devices):
        raise ValueError(f"GPU identifiers must be unique, got {devices!r}")
    return devices


def _build_worker_slots(
    *,
    parallel_workers: int,
    num_processes: int,
    num_machines: int,
    main_process_port: int,
    raw_devices: Any,
) -> tuple[_WorkerSlot, ...]:
    """Partition the device pool and assign an independent port per worker."""

    if parallel_workers < 1:
        raise ValueError(
            f"eval.parallel_workers must be at least 1, got {parallel_workers}"
        )
    if num_processes < 1:
        raise ValueError(
            f"eval.num_processes must be at least 1, got {num_processes}"
        )
    if parallel_workers > 1 and num_machines != 1:
        raise ValueError(
            "Checkpoint-level parallel evaluation currently supports one "
            "machine only; set eval.num_machines=1."
        )
    if (
        main_process_port != 0
        and main_process_port + parallel_workers - 1 > 65535
    ):
        raise ValueError(
            "eval.main_process_port plus eval.parallel_workers exceeds "
            f"65535: {main_process_port} + {parallel_workers} - 1"
        )

    required_count = parallel_workers * num_processes
    devices = _visible_devices(raw_devices, required_count=required_count)
    slots = []
    for worker_index in range(parallel_workers):
        start = worker_index * num_processes
        end = start + num_processes
        slots.append(
            _WorkerSlot(
                index=worker_index,
                devices=devices[start:end],
                main_process_port=(
                    0
                    if main_process_port == 0
                    else main_process_port + worker_index
                ),
            )
        )
    return tuple(slots)


def _worker_environment(
    base_environment: Mapping[str, str],
    *,
    slot: _WorkerSlot,
    num_processes: int,
) -> dict[str, str]:
    """Build an isolated CUDA/DDP environment for one checkpoint worker."""

    environment = dict(base_environment)
    for name in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    environment["CUDA_VISIBLE_DEVICES"] = slot.visible_devices
    if num_processes == 1:
        # main.py reads these values directly. Accelerate single-process
        # launch does not consistently inject them, so never inherit stale
        # rank state from a parent shell or scheduler.
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


def _filter_forward_overrides(task_overrides: list[str]) -> list[str]:
    blocked_prefixes = [
        "model.load_from=",
        "model.load_from~",
        "_train=",
        "_validation=",
        "_test=",
        "_predict=",
        "work_dir=",
        "+work_dir=",
        "++work_dir=",
        "hydra.run.dir=",
        "+hydra.run.dir=",
        "++hydra.run.dir=",
        "hydra.job.chdir=",
        "+hydra.job.chdir=",
        "++hydra.job.chdir=",
        "eval.primary_config=",
        "+eval.primary_config=",
        "++eval.primary_config=",
        "~eval.primary_config",
        "+~eval.primary_config",
        "++~eval.primary_config",
    ]
    kept: list[str] = []
    for ov in task_overrides:
        if any(ov.startswith(prefix) for prefix in blocked_prefixes):
            continue
        kept.append(ov)
    return kept


def _run_command(
    cmd: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    subprocess.run(
        list(cmd),
        cwd=str(cwd),
        check=True,
        env=dict(environment) if environment is not None else None,
    )


class _ForwardedSignal(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


class _ProcessGroupCleanupError(BaseException):
    """Nested accelerate PGID remains alive, so evaluation must not continue."""


def _as_bool(raw: Any, *, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def _watchdog_enabled(default: bool = True) -> bool:
    raw = os.environ.get("SEED_SWEEP_WATCHDOG_ENABLED")
    if raw is None:
        return default
    return _as_bool(raw, name="SEED_SWEEP_WATCHDOG_ENABLED")


def _watchdog_float(name: str, default: float, *, positive: bool = False) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    if not (value > 0.0 if positive else value >= 0.0):
        operator = "greater than zero" if positive else "non-negative"
        raise ValueError(f"{name} must be {operator}, got {raw!r}")
    return value


def _config_float(
    raw: Any,
    *,
    name: str,
    positive: bool = False,
) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    if not (value > 0.0 if positive else value >= 0.0):
        operator = "greater than zero" if positive else "non-negative"
        raise ValueError(f"{name} must be {operator}, got {raw!r}")
    return value


def _build_watchdog_config(eval_cfg: Any) -> _WatchdogConfig:
    """Resolve standalone defaults while preserving seed-sweep overrides."""

    configured_enabled = _as_bool(
        eval_cfg.get("watchdog_enabled", True),
        name="eval.watchdog_enabled",
    )
    enabled = _watchdog_enabled(configured_enabled)
    seed_sweep_managed = "SEED_SWEEP_WATCHDOG_ENABLED" in os.environ

    if seed_sweep_managed:
        poll_seconds = _watchdog_float(
            "SEED_SWEEP_WATCHDOG_POLL_SECONDS",
            1.0,
            positive=True,
        )
        post_completion_grace_seconds = _watchdog_float(
            "SEED_SWEEP_WATCHDOG_POST_COMPLETION_GRACE_SECONDS",
            10.0,
        )
        outer_terminate_grace_seconds = _watchdog_float(
            "SEED_SWEEP_WATCHDOG_TERMINATE_GRACE_SECONDS",
            20.0,
            positive=True,
        )
        terminate_grace_seconds = _watchdog_float(
            "SEED_SWEEP_WATCHDOG_INNER_TERMINATE_GRACE_SECONDS",
            outer_terminate_grace_seconds * 0.25,
            positive=True,
        )
        hard_timeout_seconds = _watchdog_float(
            "SEED_SWEEP_WATCHDOG_HARD_TIMEOUT_SECONDS",
            0.0,
        )
    else:
        poll_seconds = _config_float(
            eval_cfg.get("watchdog_poll_seconds", 1.0),
            name="eval.watchdog_poll_seconds",
            positive=True,
        )
        post_completion_grace_seconds = _config_float(
            eval_cfg.get("watchdog_post_completion_grace_seconds", 10.0),
            name="eval.watchdog_post_completion_grace_seconds",
        )
        terminate_grace_seconds = _config_float(
            eval_cfg.get("watchdog_terminate_grace_seconds", 20.0),
            name="eval.watchdog_terminate_grace_seconds",
            positive=True,
        )
        hard_timeout_seconds = _config_float(
            eval_cfg.get("watchdog_hard_timeout_seconds", 0.0),
            name="eval.watchdog_hard_timeout_seconds",
        )
    return _WatchdogConfig(
        enabled=enabled,
        poll_seconds=poll_seconds,
        post_completion_grace_seconds=post_completion_grace_seconds,
        terminate_grace_seconds=terminate_grace_seconds,
        hard_timeout_seconds=hard_timeout_seconds,
    )


def _process_group_exists(process: subprocess.Popen[Any]) -> bool:
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
        elif process.poll() is not None:
            return
        elif signum == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def _terminate_process_group(
    process: subprocess.Popen[Any],
    grace_seconds: float,
) -> bool:
    """Terminate an accelerate process group; return whether SIGKILL was needed."""

    if not _process_group_exists(process):
        return False
    _send_process_group_signal(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    poll_seconds = min(0.05, max(0.001, grace_seconds / 4.0))
    while time.monotonic() < deadline:
        if not _process_group_exists(process):
            return False
        time.sleep(poll_seconds)
    if _process_group_exists(process):
        _send_process_group_signal(process, signal.SIGKILL)
        kill_deadline = time.monotonic() + min(1.0, max(0.01, grace_seconds))
        while time.monotonic() < kill_deadline:
            if not _process_group_exists(process):
                break
            time.sleep(min(poll_seconds, 1.0))
        if _process_group_exists(process):
            raise _ProcessGroupCleanupError(
                f"Nested accelerate process group pgid={process.pid} "
                "still exists after SIGKILL"
            )
        return True
    return False


def _terminate_and_reap(
    process: subprocess.Popen[Any],
    grace_seconds: float,
) -> tuple[int, bool]:
    escalated = _terminate_process_group(process, grace_seconds)
    try:
        returncode = process.wait(timeout=max(0.5, grace_seconds + 0.5))
    except subprocess.TimeoutExpired:
        _send_process_group_signal(process, signal.SIGKILL)
        try:
            returncode = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            raise _ProcessGroupCleanupError(
                f"Nested accelerate leader pid={process.pid} survived SIGKILL"
            ) from exc
        escalated = True
    if _process_group_exists(process):
        raise _ProcessGroupCleanupError(
            f"Nested accelerate process group pgid={process.pid} was not reaped"
        )
    return returncode, escalated


def _run_inference_command(
    cmd: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    *,
    eval_workspace: Path,
    run_id: str,
    watchdog: _WatchdogConfig,
) -> None:
    """Run main.py with a completion-aware watchdog around DDP teardown."""

    if not watchdog.enabled:
        _run_command(cmd, cwd=cwd, environment=environment)
        return

    poll_seconds = watchdog.poll_seconds
    post_completion_grace_seconds = (
        watchdog.post_completion_grace_seconds
    )
    terminate_grace_seconds = watchdog.terminate_grace_seconds
    hard_timeout_seconds = watchdog.hard_timeout_seconds

    marker_path = eval_workspace / ".inference_complete.json"
    token = uuid.uuid4().hex
    nested_group_path_raw = environment.get("SEED_SWEEP_NESTED_PGID_FILE")
    nested_group_path = (
        Path(nested_group_path_raw).resolve()
        if nested_group_path_raw
        else None
    )
    outer_run_id = environment.get("SEED_SWEEP_COMPLETION_RUN_ID", "")
    outer_token = environment.get("SEED_SWEEP_COMPLETION_TOKEN", "")
    run_environment = dict(environment)
    run_environment.update(
        {
            "SEED_SWEEP_INFERENCE_COMPLETION_MARKER": str(marker_path.resolve()),
            "SEED_SWEEP_INFERENCE_COMPLETION_RUN_ID": run_id,
            "SEED_SWEEP_INFERENCE_COMPLETION_TOKEN": token,
        }
    )

    originals: dict[int, Any] = {}
    process: subprocess.Popen[Any] | None = None
    ready_read_fd: int | None = None
    ready_write_fd: int | None = None
    pending_spawn_signals: list[int] = []
    supervision_ready = False

    def forward_signal(signum: int, _frame: Any) -> None:
        if not supervision_ready:
            pending_spawn_signals.append(signum)
            return
        raise _ForwardedSignal(signum)

    # Python only permits signal registration in the main thread. Directory
    # parallelism calls this function from worker threads; each worker still
    # supervises and reaps its own process group, while serial/seed-sweep mode
    # retains the original signal-forwarding contract.
    manages_signals = current_thread() is main_thread()
    if manages_signals:
        for signum in (
            signal.SIGINT,
            signal.SIGTERM,
            *([signal.SIGHUP] if hasattr(signal, "SIGHUP") else []),
        ):
            originals[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)

    try:
        started = time.monotonic()
        if nested_group_path is not None:
            if not manages_signals:
                raise RuntimeError(
                    "Seed-sweep nested process registration must run in the "
                    "main thread; do not combine it with "
                    "eval.parallel_workers>1."
                )
            if os.name != "posix" or not sys.platform.startswith("linux"):
                raise RuntimeError(
                    "The seed-sweep nested process-group watchdog requires "
                    "Linux (process groups plus PR_SET_PDEATHSIG)"
                )
            # The child-side trampoline creates the new session, arms Linux
            # parent-death protection, atomically registers its own PID/PGID,
            # and only then execs Accelerate.  Therefore an uncatchable death
            # of this eval process cannot land between workload startup and
            # nested-PGID registration.
            registered_exec_path = (
                Path(__file__).resolve().parent
                / "seed_sweep"
                / "registered_exec.py"
            )
            ready_read_fd, ready_write_fd = os.pipe()
            registered_command = [
                sys.executable,
                str(registered_exec_path),
                "--record-path",
                str(nested_group_path),
                "--producer",
                "eval_pth_dir",
                "--run-id",
                outer_run_id,
                "--token",
                outer_token,
                "--expected-parent-pid",
                str(os.getpid()),
                "--ready-fd",
                str(ready_write_fd),
                "--",
                *cmd,
            ]
            try:
                process = subprocess.Popen(
                    registered_command,
                    cwd=str(cwd),
                    env=run_environment,
                    start_new_session=True,
                    pass_fds=(ready_write_fd,),
                )
            finally:
                os.close(ready_write_fd)
                ready_write_fd = None

            # Do not consider supervision armed until the trampoline proves
            # that the durable record exists.  Signals received meanwhile are
            # deferred; SIGKILL/OOM is handled by its parent-death guard.
            while True:
                if pending_spawn_signals:
                    raise _ForwardedSignal(pending_spawn_signals[0])
                readable, _, _ = select.select(
                    [ready_read_fd],
                    [],
                    [],
                    min(poll_seconds, 1.0),
                )
                if readable:
                    readiness = os.read(ready_read_fd, 1)
                    if readiness == b"\x01":
                        break
                    returncode = process.poll()
                    if returncode not in (None, 0):
                        raise subprocess.CalledProcessError(returncode, cmd)
                    raise RuntimeError(
                        "Registered-exec trampoline closed its readiness "
                        "channel before confirming nested-PGID registration"
                    )
                returncode = process.poll()
                if returncode is not None:
                    if returncode != 0:
                        raise subprocess.CalledProcessError(returncode, cmd)
                    raise RuntimeError(
                        "Registered-exec trampoline exited before confirming "
                        "nested-PGID registration"
                    )
                if (
                    hard_timeout_seconds > 0.0
                    and time.monotonic() - started >= hard_timeout_seconds
                ):
                    _terminate_and_reap(process, terminate_grace_seconds)
                    raise subprocess.TimeoutExpired(cmd, hard_timeout_seconds)
            os.close(ready_read_fd)
            ready_read_fd = None
        else:
            # Standalone eval (outside seed-sweep) keeps its original launch
            # path.  The nested registration contract is only supplied by the
            # outer seed runner.
            process = subprocess.Popen(
                list(cmd),
                cwd=str(cwd),
                env=run_environment,
                start_new_session=(os.name == "posix"),
            )

        supervision_ready = True
        if pending_spawn_signals:
            raise _ForwardedSignal(pending_spawn_signals[0])
        marker_seen_at: float | None = None
        completion_proved = False
        while True:
            now = time.monotonic()
            if not completion_proved:
                completion_proved = (
                    read_completion_marker(
                        marker_path,
                        expected_run_id=run_id,
                        expected_token=token,
                        expected_producer="main.py",
                    )
                    is not None
                )
                if completion_proved:
                    marker_seen_at = now
                    print(
                        "Inference completion marker received; waiting "
                        f"{post_completion_grace_seconds:g}s for normal DDP exit."
                    )

            returncode = process.poll()
            if returncode is not None:
                if _process_group_exists(process):
                    _terminate_and_reap(process, terminate_grace_seconds)
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd)
                return

            if (
                marker_seen_at is not None
                and now - marker_seen_at >= post_completion_grace_seconds
            ):
                returncode, escalated = _terminate_and_reap(
                    process,
                    terminate_grace_seconds,
                )
                print(
                    "Inference test() completed but its process group did not "
                    "exit within the watchdog grace; terminated it "
                    f"(returncode={returncode}, escalated_to_sigkill={escalated})."
                )
                return

            if (
                hard_timeout_seconds > 0.0
                and now - started >= hard_timeout_seconds
            ):
                _terminate_and_reap(process, terminate_grace_seconds)
                raise subprocess.TimeoutExpired(cmd, hard_timeout_seconds)

            time.sleep(min(poll_seconds, 1.0))
    except _ForwardedSignal as exc:
        # A second scheduler signal must not re-enter and interrupt nested-PGID
        # cleanup. The outer seed runner still has a longer grace before KILL.
        for signum in originals:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None:
            try:
                _terminate_and_reap(process, terminate_grace_seconds)
            except (OSError, subprocess.SubprocessError):
                pass
        raise SystemExit(128 + exc.signum) from None
    except BaseException:
        for signum in originals:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None:
            try:
                _terminate_and_reap(process, terminate_grace_seconds)
            except (OSError, subprocess.SubprocessError):
                pass
        raise
    finally:
        if ready_read_fd is not None:
            os.close(ready_read_fd)
        if ready_write_fd is not None:
            os.close(ready_write_fd)
        if (
            nested_group_path is not None
            and process is not None
            and not _process_group_exists(process)
        ):
            try:
                nested_group_path.unlink()
            except FileNotFoundError:
                pass
        for signum, handler in originals.items():
            signal.signal(signum, handler)


def _initial_row(checkpoint: Path) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "fgd": "",
        "bc": "",
        "l1": "",
        "lvd": "",
        "mse": "",
        "style_swd": "",
        "style_acf_error": "",
        "status": "ok",
        "error": "",
    }


def _checkpoint_workspace(
    context: _EvalContext,
    checkpoint: Path,
) -> Path:
    if context.single_ckpt_mode:
        # Keep the historical single-checkpoint workspace contract used by
        # external evaluation orchestration.
        return context.workspace_root
    return context.workspace_root / checkpoint.stem


def _evaluate_checkpoint(
    *,
    ordinal: int,
    total: int,
    checkpoint: Path,
    slot: _WorkerSlot,
    context: _EvalContext,
) -> dict[str, Any]:
    """Generate and score one checkpoint as an atomic GPU-slot task."""

    row = _initial_row(checkpoint)
    eval_workspace = _checkpoint_workspace(context, checkpoint)
    eval_npz_dir = eval_workspace / "npz" / "test"
    eval_wav_dir = eval_workspace / "wav" / "test"
    eval_json_path = eval_workspace / "emage_metrics.json"
    worker_environment = _worker_environment(
        context.base_environment,
        slot=slot,
        num_processes=context.num_processes,
    )
    worker_label = (
        f"worker={slot.index} gpu={slot.visible_devices} "
        f"port={slot.main_process_port}"
    )
    print(
        f"[{ordinal}/{total}] [{worker_label}] "
        f"Evaluating checkpoint: {checkpoint}",
        flush=True,
    )

    try:
        # Dedicated workspaces prevent checkpoint workers from deleting or
        # consuming one another's generated files.
        if eval_npz_dir.exists():
            shutil.rmtree(eval_npz_dir, ignore_errors=True)
        if eval_wav_dir.exists():
            shutil.rmtree(eval_wav_dir, ignore_errors=True)
        eval_json_path.unlink(missing_ok=True)
        (eval_workspace / ".inference_complete.json").unlink(
            missing_ok=True
        )
        eval_workspace.mkdir(parents=True, exist_ok=True)

        main_cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_machines",
            str(context.num_machines),
            "--num_processes",
            str(context.num_processes),
            "--main_process_port",
            str(slot.main_process_port),
            "--mixed_precision",
            context.mixed_precision,
            "--dynamo_backend",
            context.dynamo_backend,
            str((context.project_root / "main.py").as_posix()),
            *context.main_config_args,
            *context.forwarded_overrides,
            f"model.load_from={checkpoint.as_posix()}",
            "_train=false",
            "_validation=false",
            "_test=true",
            "_predict=false",
            "train.visualize=false",
            f"++hydra.run.dir={eval_workspace.as_posix()}",
            "++hydra.job.chdir=false",
            f"++work_dir={eval_workspace.as_posix()}",
        ]
        _run_inference_command(
            main_cmd,
            cwd=context.project_root,
            environment=worker_environment,
            eval_workspace=eval_workspace,
            run_id=context.run_id,
            watchdog=context.watchdog,
        )

        eval_cmd = [
            sys.executable,
            str(
                (
                    context.project_root
                    / "scripts"
                    / context.eval_script_name
                ).as_posix()
            ),
            "-i",
            eval_npz_dir.as_posix(),
            "-ia",
            eval_wav_dir.as_posix(),
            "-o",
            eval_json_path.as_posix(),
            "-p",
            context.eval_person,
        ]
        if context.eval_download_path is not None:
            eval_cmd.extend(["-dp", context.eval_download_path])
        if context.eval_split_file is not None:
            eval_cmd.extend(["-sf", context.eval_split_file])
        if context.eval_device is not None:
            eval_cmd.extend(["-d", context.eval_device])
        if context.eval_only_fgd:
            eval_cmd.extend(["--only_fgd", "True"])
        elif (
            context.eval_script_name == "emage_eval.py"
            and not context.eval_style_metrics
        ):
            eval_cmd.append("--no_style_metrics")
        _run_command(
            eval_cmd,
            cwd=context.project_root,
            environment=worker_environment,
        )

        with eval_json_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        row["fgd"] = float(metrics.get("fgd"))
        if not context.eval_only_fgd:
            row["bc"] = float(metrics.get("bc"))
            row["l1"] = float(metrics.get("l1"))
            row["lvd"] = float(metrics.get("lvd"))
            row["mse"] = float(metrics.get("mse"))
        if context.emage_style_metrics_enabled:
            for metric_name in ("style_swd", "style_acf_error"):
                value = metrics.get(metric_name)
                if value is not None:
                    row[metric_name] = float(value)
        print(
            f"[ok] [{worker_label}] {checkpoint.name}: "
            f"FGD={float(row['fgd']):.6f}",
            flush=True,
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        print(
            f"[failed] [{worker_label}] {checkpoint.name}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    return row


def _best_successful_row(
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


def _write_rows_csv(
    csv_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically persist all completed rows after every checkpoint."""

    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in _CSV_FIELDNAMES}
            for row in rows
        )
    os.replace(temporary, csv_path)


def _copy_best_effective_config(
    *,
    best_row: Mapping[str, Any],
    context: _EvalContext,
) -> None:
    """Keep the historical root-workspace config lookup working."""

    if context.single_ckpt_mode:
        return
    source_workspace = _checkpoint_workspace(
        context,
        Path(str(best_row["checkpoint"])),
    )
    for relative_path in (Path("config.yaml"), Path(".hydra/config.yaml")):
        source = source_workspace / relative_path
        if not source.is_file():
            continue
        destination = context.workspace_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _run_checkpoint_tasks(
    *,
    checkpoints: Sequence[Path],
    worker_slots: Sequence[_WorkerSlot],
    context: _EvalContext,
    csv_path: Path,
) -> list[dict[str, Any]]:
    """Run serially or reuse one fixed GPU slot per concurrent future."""

    rows: list[dict[str, Any]] = []
    checkpoint_order = {
        str(checkpoint): index
        for index, checkpoint in enumerate(checkpoints)
    }

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        rows.sort(key=lambda item: checkpoint_order[item["checkpoint"]])
        _write_rows_csv(csv_path, rows)

    if len(worker_slots) == 1:
        slot = worker_slots[0]
        for ordinal, checkpoint in enumerate(checkpoints, start=1):
            record(
                _evaluate_checkpoint(
                    ordinal=ordinal,
                    total=len(checkpoints),
                    checkpoint=checkpoint,
                    slot=slot,
                    context=context,
                )
            )
        return rows

    checkpoint_iterator = iter(enumerate(checkpoints, start=1))
    pending: dict[Future[dict[str, Any]], _WorkerSlot] = {}

    def submit_next(
        executor: ThreadPoolExecutor,
        slot: _WorkerSlot,
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
            context=context,
        )
        pending[future] = slot
        return True

    with ThreadPoolExecutor(
        max_workers=len(worker_slots),
        thread_name_prefix="eval-pth",
    ) as executor:
        for slot in worker_slots:
            if not submit_next(executor, slot):
                break

        while pending:
            completed, _ = wait(
                tuple(pending),
                return_when=FIRST_COMPLETED,
            )
            reusable_slots: list[_WorkerSlot] = []
            for future in completed:
                slot = pending.pop(future)
                record(future.result())
                reusable_slots.append(slot)
            for slot in reusable_slots:
                submit_next(executor, slot)
    return rows


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg) -> None:
    if cfg.model.get("load_from", None) is None:
        raise ValueError("Please provide model.load_from as a checkpoint directory (or a single .pth file).")

    project_root = Path(__file__).resolve().parents[1]
    load_from_path = _as_abs_path(str(cfg.model.load_from))
    single_ckpt_mode = load_from_path.is_file()
    checkpoints, ckpt_dir = _collect_checkpoints(str(cfg.model.load_from))
    if len(checkpoints) == 0:
        raise FileNotFoundError(f"No .pth files found in: {ckpt_dir}")

    if single_ckpt_mode:
        print(f"Single-checkpoint mode: only evaluating {load_from_path.name}")

    hydra_task_overrides = list(HydraConfig.get().overrides.task)
    forwarded_overrides = _filter_forward_overrides(hydra_task_overrides)

    eval_cfg = cfg.get("eval", {})
    num_machines = int(eval_cfg.get("num_machines", 1))
    parallel_workers = int(eval_cfg.get("parallel_workers", 1))
    configured_num_processes = eval_cfg.get("num_processes", None)
    if configured_num_processes is None:
        # Backward compatibility: the original serial launcher used every
        # visible GPU. In checkpoint-parallel mode, the natural default is one
        # GPU/process per worker.
        num_processes = (
            1
            if parallel_workers > 1
            else _infer_num_processes_from_visible_devices()
        )
    else:
        num_processes = int(configured_num_processes)
    main_process_port = _resolve_main_process_port(
        eval_cfg.get("main_process_port", None),
        num_machines,
    )
    worker_slots = _build_worker_slots(
        parallel_workers=parallel_workers,
        num_processes=num_processes,
        num_machines=num_machines,
        main_process_port=main_process_port,
        raw_devices=eval_cfg.get("cuda_visible_devices", None),
    )
    if (
        os.environ.get("SEED_SWEEP_NESTED_PGID_FILE")
        and parallel_workers != 1
    ):
        raise ValueError(
            "The outer seed-sweep process-group contract supports "
            "eval.parallel_workers=1 only."
        )
    mixed_precision = str(eval_cfg.get("mixed_precision", "no"))
    dynamo_backend = str(eval_cfg.get("dynamo_backend", "no"))
    watchdog = _build_watchdog_config(eval_cfg)

    # emage_eval.py forwarded parameters
    eval_person = str(eval_cfg.get("person", "2"))  # -p, default "2"
    eval_download_path = eval_cfg.get("download_path", None)  # -dp
    eval_split_file = eval_cfg.get("split_file", None)  # -sf
    eval_device = eval_cfg.get("device", None)  # -d
    eval_only_fgd = _as_bool(
        eval_cfg.get("only_fgd", False),
        name="eval.only_fgd",
    )
    eval_style_metrics = _as_bool(
        eval_cfg.get("style_metrics", True),
        name="eval.style_metrics",
    )
    primary_config_raw = eval_cfg.get("primary_config", None)
    primary_config: Path | None = None
    if primary_config_raw is not None:
        primary_config = _as_abs_path(str(primary_config_raw))
        if not primary_config.is_file() or primary_config.suffix not in {".yaml", ".yml"}:
            raise FileNotFoundError(
                "eval.primary_config must point to one existing .yaml/.yml file, "
                f"got: {primary_config}"
            )

    # 选择评估后端: 'emage' (默认, 使用 emage_eval.py) 或 'semtalk' (使用 emage_eval_semtalk.py)
    backend = str(eval_cfg.get("backend", "emage")).lower()
    if backend not in ("emage", "semtalk"):
        raise ValueError(f"Unknown eval.backend: {backend!r}. Must be 'emage' or 'semtalk'.")
    eval_script_name = "emage_eval.py" if backend == "emage" else "emage_eval_semtalk.py"
    emage_style_metrics_enabled = (
        backend == "emage" and eval_style_metrics and not eval_only_fgd
    )

    # 生成 run_id，隔离并发运行的中间文件和输出 CSV
    run_id = eval_cfg.get("run_id", None)
    if run_id is None:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_id = str(run_id)

    workspace_root = ckpt_dir / f"_eval_tmp_{run_id}"
    workspace_root.mkdir(parents=True, exist_ok=True)
    main_config_args = (
        (
            "--config-path",
            primary_config.parent.as_posix(),
            "--config-name",
            primary_config.stem,
        )
        if primary_config is not None
        else ()
    )
    context = _EvalContext(
        project_root=project_root,
        workspace_root=workspace_root,
        single_ckpt_mode=single_ckpt_mode,
        run_id=run_id,
        forwarded_overrides=tuple(forwarded_overrides),
        main_config_args=main_config_args,
        num_machines=num_machines,
        num_processes=num_processes,
        mixed_precision=mixed_precision,
        dynamo_backend=dynamo_backend,
        eval_script_name=eval_script_name,
        eval_person=eval_person,
        eval_download_path=(
            str(eval_download_path)
            if eval_download_path is not None
            else None
        ),
        eval_split_file=(
            str(eval_split_file) if eval_split_file is not None else None
        ),
        eval_device=str(eval_device) if eval_device is not None else None,
        eval_only_fgd=eval_only_fgd,
        eval_style_metrics=eval_style_metrics,
        emage_style_metrics_enabled=emage_style_metrics_enabled,
        watchdog=watchdog,
        base_environment=os.environ.copy(),
    )

    print(
        f"Eval launcher config: backend={backend}, run_id={run_id}, num_machines={num_machines}, "
        f"parallel_workers={parallel_workers}, "
        f"num_processes_per_worker={num_processes}, "
        f"main_process_port={main_process_port}, "
        f"mixed_precision={mixed_precision}, "
        f"dynamo_backend={dynamo_backend}, primary_config={primary_config}, "
        f"style_metrics={emage_style_metrics_enabled}, "
        f"watchdog_enabled={watchdog.enabled}"
    )
    for slot in worker_slots:
        print(
            f"  worker {slot.index}: CUDA_VISIBLE_DEVICES="
            f"{slot.visible_devices}, port={slot.main_process_port}",
            flush=True,
        )

    if single_ckpt_mode:
        csv_path = ckpt_dir / f"pth_eval_metrics_{load_from_path.stem}_{backend}_{run_id}.csv"
    else:
        csv_path = ckpt_dir / f"pth_eval_metrics_{backend}_{run_id}.csv"
    _write_rows_csv(csv_path, ())
    rows = _run_checkpoint_tasks(
        checkpoints=checkpoints,
        worker_slots=worker_slots,
        context=context,
        csv_path=csv_path,
    )
    best_row = _best_successful_row(rows)
    if best_row is not None:
        _copy_best_effective_config(best_row=best_row, context=context)

    print(f"Saved metrics CSV to: {csv_path}")
    if best_row is None:
        print("No successful checkpoint evaluation. Please check CSV error column.")
    elif single_ckpt_mode:
        # 单 ckpt 模式：只跑单轮测试，跳过最优筛选与清理
        fgd_val = best_row.get("fgd")
        if fgd_val is not None and fgd_val != "":
            print(f"Single-checkpoint result: {best_row['checkpoint']} (FGD={float(fgd_val):.6f})")
    else:
        print(
            "Best checkpoint by FGD: "
            f"{best_row['checkpoint']} (FGD={float(best_row['fgd']):.6f})"
        )

        # --- cleanup non-best checkpoints ---
        # eval.cleanup_non_best: "no" (default) | "yes" | "ask"
        cleanup_mode = str(eval_cfg.get("cleanup_non_best", "ask")).lower()
        best_ckpt = Path(best_row["checkpoint"]).resolve()

        # 找到 epoch 最大的 checkpoint，也予以保留
        latest_ckpt: Path | None = None
        max_epoch = -1
        for c in checkpoints:
            ep = _parse_epoch(c)
            if ep is not None and ep > max_epoch:
                max_epoch = ep
                latest_ckpt = c.resolve()

        keep_set = {best_ckpt}
        if latest_ckpt is not None:
            keep_set.add(latest_ckpt)

        non_best = [c for c in checkpoints if c.resolve() not in keep_set]

        kept_names = [p.name for p in keep_set]
        print(f"Checkpoints to keep: {kept_names}")

        if non_best and cleanup_mode in ("yes", "ask"):
            if cleanup_mode == "ask":
                print(f"\nFound {len(non_best)} non-kept checkpoint(s):")
                for nb in non_best:
                    print(f"  - {nb.name}")
                answer = input("Delete all non-kept checkpoints? [y/N] ").strip().lower()
                do_delete = answer in ("y", "yes")
            else:
                do_delete = True

            if do_delete:
                for nb in non_best:
                    try:
                        nb.unlink()
                        print(f"Deleted: {nb}")
                    except OSError as e:
                        print(f"Failed to delete {nb}: {e}")
                print(f"Kept checkpoints: {kept_names}")
            else:
                print("Skipped cleanup.")

    completion_marker_raw = os.environ.get("SEED_SWEEP_COMPLETION_MARKER")
    defer_completion_marker = os.environ.get(
        "SEED_SWEEP_DEFER_COMPLETION_MARKER",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if completion_marker_raw and not defer_completion_marker:
        completion_run_id = os.environ.get("SEED_SWEEP_COMPLETION_RUN_ID", "")
        completion_token = os.environ.get("SEED_SWEEP_COMPLETION_TOKEN", "")
        if completion_run_id != run_id:
            raise RuntimeError(
                "Seed-sweep completion run_id does not match eval.run_id: "
                f"{completion_run_id!r} != {run_id!r}"
            )
        write_completion_marker(
            completion_marker_raw,
            producer="eval_pth_dir",
            run_id=completion_run_id,
            token=completion_token,
        )
        print(f"Published seed-sweep completion marker: {completion_marker_raw}")


if __name__ == "__main__":
    main()
