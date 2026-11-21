#!/usr/bin/env python3
"""Write and validate per-attempt completion markers and nested-PGID records.

The token is an attempt-specific nonce supplied by the sweep runner.  Requiring
both it and the run ID prevents a marker left by an older launcher invocation
from being mistaken for completion of the current attempt.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


MARKER_VERSION = 1
_MARKER_KEYS = frozenset(
    {"version", "producer", "run_id", "token", "finished_at"}
)
_PROCESS_GROUP_KEYS = frozenset(
    {"version", "producer", "run_id", "token", "pid", "pgid"}
)
_PRODUCER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAX_MARKER_BYTES = 64 * 1024
_MAX_IDENTIFIER_LENGTH = 256
_MAX_TOKEN_LENGTH = 4096
_MAX_PID = 2**31 - 1


@dataclass(frozen=True)
class CompletionMarker:
    version: int
    producer: str
    run_id: str
    token: str
    finished_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "producer": self.producer,
            "run_id": self.run_id,
            "token": self.token,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True)
class ProcessGroupRecord:
    version: int
    producer: str
    run_id: str
    token: str
    pid: int
    pgid: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "producer": self.producer,
            "run_id": self.run_id,
            "token": self.token,
            "pid": self.pid,
            "pgid": self.pgid,
        }


def _valid_token(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_TOKEN_LENGTH
        and all(0x21 <= ord(character) <= 0x7E for character in value)
    )


def _valid_finished_at(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value) > 128:
        return False
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _marker_from_payload(payload: Any) -> CompletionMarker | None:
    if not isinstance(payload, dict) or frozenset(payload) != _MARKER_KEYS:
        return None
    version = payload.get("version")
    producer = payload.get("producer")
    run_id = payload.get("run_id")
    token = payload.get("token")
    finished_at = payload.get("finished_at")
    if type(version) is not int or version != MARKER_VERSION:
        return None
    if (
        not isinstance(producer, str)
        or len(producer) > _MAX_IDENTIFIER_LENGTH
        or _PRODUCER_PATTERN.fullmatch(producer) is None
    ):
        return None
    if (
        not isinstance(run_id, str)
        or len(run_id) > _MAX_IDENTIFIER_LENGTH
        or _RUN_ID_PATTERN.fullmatch(run_id) is None
    ):
        return None
    if not _valid_token(token) or not _valid_finished_at(finished_at):
        return None
    return CompletionMarker(
        version=version,
        producer=producer,
        run_id=run_id,
        token=token,
        finished_at=finished_at,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def _read_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = handle.read(_MAX_MARKER_BYTES + 1)
    except (OSError, UnicodeError):
        return None
    if len(content.encode("utf-8")) > _MAX_MARKER_BYTES:
        return None
    try:
        return json.loads(
            content,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except (json.JSONDecodeError, ValueError):
        return None


def read_completion_marker(
    path: str | os.PathLike[str],
    *,
    expected_run_id: str,
    expected_token: str,
    expected_producer: str | None = None,
) -> CompletionMarker | None:
    """Return a validated marker, or ``None`` when completion is unproven.

    Missing, unreadable, oversized, malformed, schema-incompatible, and stale
    markers are deliberately indistinguishable to watchdog callers.  A marker
    is stale when its run ID or attempt token differs from the expected value.
    """

    if (
        not isinstance(expected_run_id, str)
        or len(expected_run_id) > _MAX_IDENTIFIER_LENGTH
        or _RUN_ID_PATTERN.fullmatch(expected_run_id) is None
        or not _valid_token(expected_token)
    ):
        return None
    if expected_producer is not None and (
        not isinstance(expected_producer, str)
        or len(expected_producer) > _MAX_IDENTIFIER_LENGTH
        or _PRODUCER_PATTERN.fullmatch(expected_producer) is None
    ):
        return None

    marker = _marker_from_payload(_read_json(Path(path)))
    if marker is None:
        return None
    if not hmac.compare_digest(marker.run_id, expected_run_id):
        return None
    if not hmac.compare_digest(marker.token, expected_token):
        return None
    if expected_producer is not None and not hmac.compare_digest(
        marker.producer, expected_producer
    ):
        return None
    return marker


def _process_group_from_payload(payload: Any) -> ProcessGroupRecord | None:
    if not isinstance(payload, dict) or frozenset(payload) != _PROCESS_GROUP_KEYS:
        return None
    version = payload.get("version")
    producer = payload.get("producer")
    run_id = payload.get("run_id")
    token = payload.get("token")
    pid = payload.get("pid")
    pgid = payload.get("pgid")
    if type(version) is not int or version != MARKER_VERSION:
        return None
    if (
        not isinstance(producer, str)
        or len(producer) > _MAX_IDENTIFIER_LENGTH
        or _PRODUCER_PATTERN.fullmatch(producer) is None
        or not isinstance(run_id, str)
        or len(run_id) > _MAX_IDENTIFIER_LENGTH
        or _RUN_ID_PATTERN.fullmatch(run_id) is None
        or not _valid_token(token)
        or type(pid) is not int
        or type(pgid) is not int
        or pid <= 1
        or pgid <= 1
        or pid > _MAX_PID
        or pgid > _MAX_PID
        or pid != pgid
    ):
        return None
    return ProcessGroupRecord(
        version=version,
        producer=producer,
        run_id=run_id,
        token=token,
        pid=pid,
        pgid=pgid,
    )


def read_process_group_record(
    path: str | os.PathLike[str],
    *,
    expected_run_id: str,
    expected_token: str,
    expected_producer: str | None = None,
) -> ProcessGroupRecord | None:
    if (
        not isinstance(expected_run_id, str)
        or len(expected_run_id) > _MAX_IDENTIFIER_LENGTH
        or _RUN_ID_PATTERN.fullmatch(expected_run_id) is None
        or not _valid_token(expected_token)
    ):
        return None
    if expected_producer is not None and (
        not isinstance(expected_producer, str)
        or len(expected_producer) > _MAX_IDENTIFIER_LENGTH
        or _PRODUCER_PATTERN.fullmatch(expected_producer) is None
    ):
        return None
    record = _process_group_from_payload(_read_json(Path(path)))
    if record is None:
        return None
    if not hmac.compare_digest(record.run_id, expected_run_id):
        return None
    if not hmac.compare_digest(record.token, expected_token):
        return None
    if expected_producer is not None and not hmac.compare_digest(
        record.producer,
        expected_producer,
    ):
        return None
    return record


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _best_effort_fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _atomic_write_payload(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp-",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _best_effort_fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def write_completion_marker(
    path: str | os.PathLike[str],
    *,
    producer: str,
    run_id: str,
    token: str,
    finished_at: str | None = None,
) -> CompletionMarker:
    """Atomically replace *path* with one fully validated JSON marker."""

    marker = _marker_from_payload(
        {
            "version": MARKER_VERSION,
            "producer": producer,
            "run_id": run_id,
            "token": token,
            "finished_at": _utc_timestamp() if finished_at is None else finished_at,
        }
    )
    if marker is None:
        raise ValueError(
            "invalid completion marker fields: producer/run_id must be safe "
            "identifiers, token must be non-empty visible ASCII, and finished_at "
            "must be a timezone-aware ISO-8601 timestamp"
        )

    _atomic_write_payload(Path(path), marker.as_dict())
    return marker


def write_process_group_record(
    path: str | os.PathLike[str],
    *,
    producer: str,
    run_id: str,
    token: str,
    pid: int,
    pgid: int,
) -> ProcessGroupRecord:
    record = _process_group_from_payload(
        {
            "version": MARKER_VERSION,
            "producer": producer,
            "run_id": run_id,
            "token": token,
            "pid": pid,
            "pgid": pgid,
        }
    )
    if record is None:
        raise ValueError("invalid nested process-group record fields")
    _atomic_write_payload(Path(path), record.as_dict())
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_parser = subparsers.add_parser(
        "write",
        help="atomically write a completion marker",
    )
    write_parser.add_argument("path", type=Path, help="marker JSON destination")
    write_parser.add_argument("--producer", required=True)
    write_parser.add_argument("--run-id", required=True)
    write_parser.add_argument("--token", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "write":
        raise AssertionError(f"unhandled command: {args.command}")
    try:
        write_completion_marker(
            args.path,
            producer=args.producer,
            run_id=args.run_id,
            token=args.token,
        )
    except (OSError, ValueError) as exc:
        print(f"completion marker error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
