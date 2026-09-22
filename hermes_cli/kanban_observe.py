"""Dedicated no-init, no-migrate, read-only Kanban board observer."""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
import sqlite3
import stat
import time
from typing import Any
from urllib.parse import quote

from hermes_cli.kanban_contracts import ContractValidationError, canonical_json_sha256

ObserveRequestV1 = Mapping[str, Any]
ObserveReportV1 = dict[str, Any]

_TASK_ID_PREFIX = "t_"


class ObserveError(RuntimeError):
    """An observation failure that must never cause a retry via a writer path."""


def _validate_request(request: ObserveRequestV1) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ContractValidationError("WRONG_TYPE", "observe request must be an object")
    payload = dict(request)
    expected_keys = {
        "task_id",
        "format",
        "no_init",
        "no_migrate",
        "no_recompute",
        "expected_schema_version",
    }
    extras = set(payload) - expected_keys
    if extras:
        raise ContractValidationError("UNEXPECTED_PROPERTY", f"unexpected request fields: {sorted(extras)}")
    missing = expected_keys - set(payload)
    if missing:
        raise ContractValidationError("MISSING_REQUIRED", f"missing request fields: {sorted(missing)}")
    task_id = payload["task_id"]
    if not isinstance(task_id, str) or len(task_id) != 10 or not task_id.startswith(_TASK_ID_PREFIX):
        raise ContractValidationError("WRONG_TYPE", "task_id must match t_[0-9a-f]{8}")
    suffix = task_id[2:]
    if any(char not in "0123456789abcdef" for char in suffix):
        raise ContractValidationError("WRONG_TYPE", "task_id must match t_[0-9a-f]{8}")
    if payload["format"] != "json" or any(payload[name] is not True for name in ("no_init", "no_migrate", "no_recompute")):
        raise ContractValidationError("WRONG_TYPE", "observe requires json plus all three no-* flags")
    expected_schema_version = payload["expected_schema_version"]
    if isinstance(expected_schema_version, bool) or not isinstance(expected_schema_version, int) or expected_schema_version < 1:
        raise ContractValidationError("WRONG_TYPE", "expected_schema_version must be a positive integer")
    return payload


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
    except FileNotFoundError:
        return {
            "path": str(path), "exists": False, "device": None, "inode": None,
            "mode": None, "size_bytes": None, "sha256": None,
        }
    if not stat.S_ISREG(info.st_mode):
        raise ObserveError(f"OBSERVE_API_VIOLATION: target is not a regular file: {path}")
    digest = sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path), "exists": True, "device": info.st_dev, "inode": info.st_ino,
        "mode": info.st_mode, "size_bytes": info.st_size, "sha256": digest,
    }


def _snapshot(db_path: Path) -> list[dict[str, Any]]:
    return [_file_identity(candidate) for candidate in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm"))]


def _read_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row is not None else 0


def observe_board_read_only(db_path: Path, request: ObserveRequestV1) -> ObserveReportV1:
    """Observe one board through SQLite ``mode=ro`` without normal Kanban connect.

    This function deliberately does not import ``kanban_db``: its ``connect``
    path initializes, migrates, configures WAL, and can write.  A future schema
    is rejected immediately after the read-only version probe, before table
    queries execute.
    """
    request_payload = _validate_request(request)
    target = Path(db_path)
    if not target.is_absolute():
        raise ObserveError("OBSERVE_API_VIOLATION: --db must be an absolute path")
    if not target.is_file():
        raise ObserveError(f"OBSERVE_API_VIOLATION: database does not exist: {target}")
    resolved = target.resolve(strict=True)
    before_files = _snapshot(resolved)
    before_logical_hash = canonical_json_sha256(before_files)
    uri = f"file:{quote(str(resolved))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    try:
        connection.execute("PRAGMA query_only=ON")
        raw_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        # Existing pre-versioned boards encode their baseline schema as 0.  They
        # are treated as v1 for the observer contract; only a future value fails.
        schema_version = raw_version or 1
        if schema_version > request_payload["expected_schema_version"]:
            raise ObserveError(
                "OBSERVE_UNSUPPORTED_SCHEMA: "
                f"database schema {schema_version} exceeds expected {request_payload['expected_schema_version']}"
            )
        task_count = _read_count(connection, "tasks")
        run_count = _read_count(connection, "task_runs")
        event_count = _read_count(connection, "task_events")
        if connection.total_changes != 0:
            raise ObserveError("OBSERVE_WRITE_DETECTED: read-only connection changed rows")
    finally:
        connection.close()
    after_files = _snapshot(resolved)
    after_logical_hash = canonical_json_sha256(after_files)
    if before_files != after_files or before_logical_hash != after_logical_hash:
        raise ObserveError("OBSERVE_WRITE_DETECTED: database or SQLite sidecar identity changed")
    return {
        "database_identity": f"{resolved}:{before_logical_hash}",
        "task_id": request_payload["task_id"],
        "schema_version": schema_version,
        "query_only": True,
        "uri_mode": "ro",
        "before_files": before_files,
        "after_files": after_files,
        "before_logical_hash": before_logical_hash,
        "after_logical_hash": after_logical_hash,
        "task_count": task_count,
        "run_count": run_count,
        "event_count": event_count,
        "total_changes": 0,
        "target_write_syscalls": 0,
        "observed_at": int(time.time()),
    }
