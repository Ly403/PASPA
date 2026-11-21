#!/usr/bin/env python3
"""Register a child process group before atomically handing it a workload.

This Linux-only trampoline closes the gap between ``Popen`` and publication of
the process-group record used by the seed-sweep watchdog.  Its parent must
launch it as a new session/process-group leader and pass a pipe write end in
``--ready-fd``.  The trampoline:

1. arms ``PR_SET_PDEATHSIG`` with ``SIGKILL`` and verifies the expected parent;
2. verifies that its PID is also its process-group ID;
3. atomically publishes the process-group record;
4. writes one ready byte to the inherited pipe; and
5. replaces itself with the requested command via ``execvpe``.

No workload is executed unless every registration and handshake step succeeds.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import signal
import sys
from pathlib import Path
from typing import NoReturn, Sequence

try:
    from .completion_marker import write_process_group_record
except ImportError:
    from completion_marker import write_process_group_record


_PR_SET_PDEATHSIG = 1
_READY_BYTE = b"\x01"


def _set_parent_death_signal(expected_parent_pid: int) -> None:
    """Arm SIGKILL-on-parent-death without leaving the usual fork/prctl race."""

    if not sys.platform.startswith("linux") or os.name != "posix":
        raise RuntimeError("registered_exec requires Linux")
    if expected_parent_pid <= 1:
        raise ValueError("expected parent PID must be greater than 1")

    # Checking on both sides of prctl closes the race in which the expected
    # parent exits just before the kernel starts watching it.
    if os.getppid() != expected_parent_pid:
        raise RuntimeError(
            "parent changed before PR_SET_PDEATHSIG was armed "
            f"(expected {expected_parent_pid}, got {os.getppid()})"
        )

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        prctl = libc.prctl
    except AttributeError as exc:
        raise RuntimeError("libc does not expose prctl") from exc
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), "prctl")

    current_parent_pid = os.getppid()
    if current_parent_pid != expected_parent_pid:
        raise RuntimeError(
            "parent changed while PR_SET_PDEATHSIG was being armed "
            f"(expected {expected_parent_pid}, got {current_parent_pid})"
        )


def _write_ready_byte(ready_fd: int) -> None:
    if ready_fd < 0:
        raise ValueError("ready FD must be non-negative")
    while True:
        try:
            written = os.write(ready_fd, _READY_BYTE)
            break
        except InterruptedError:
            continue
    if written != len(_READY_BYTE):
        raise OSError(errno.EIO, "short write to ready FD")


def registered_exec(
    *,
    record_path: Path,
    producer: str,
    run_id: str,
    token: str,
    expected_parent_pid: int,
    ready_fd: int,
    command: Sequence[str],
) -> NoReturn:
    """Publish this process group, complete the handshake, and exec *command*."""

    if not command or not command[0]:
        raise ValueError("a non-empty command is required")

    _set_parent_death_signal(expected_parent_pid)

    pid = os.getpid()
    pgid = os.getpgrp()
    if pid != pgid:
        raise RuntimeError(
            "registered_exec must be launched as a process-group leader "
            f"(pid={pid}, pgid={pgid})"
        )

    write_process_group_record(
        record_path,
        producer=producer,
        run_id=run_id,
        token=token,
        pid=pid,
        pgid=pgid,
    )

    # The workload must never inherit this descriptor.  A failed write is a
    # failed registration handshake and therefore intentionally blocks exec.
    _write_ready_byte(ready_fd)
    os.close(ready_fd)

    os.execvpe(command[0], list(command), os.environ.copy())
    raise AssertionError("os.execvpe unexpectedly returned")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-path", required=True, type=Path)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--expected-parent-pid", required=True, type=int)
    parser.add_argument("--ready-fd", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        separator_index = raw_arguments.index("--")
    except ValueError:
        if "-h" in raw_arguments or "--help" in raw_arguments:
            build_parser().parse_args(raw_arguments)
        print("registered_exec error: missing '--' before command", file=sys.stderr)
        return 2

    args = build_parser().parse_args(raw_arguments[:separator_index])
    command = raw_arguments[separator_index + 1 :]
    if not command or not command[0]:
        print("registered_exec error: command after '--' must not be empty", file=sys.stderr)
        return 2

    try:
        registered_exec(
            record_path=args.record_path,
            producer=args.producer,
            run_id=args.run_id,
            token=args.token,
            expected_parent_pid=args.expected_parent_pid,
            ready_fd=args.ready_fd,
            command=command,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"registered_exec error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
