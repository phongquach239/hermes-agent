"""Read-only observer contract: no false writes, no missed ones.

``observe_board_read_only`` is the surface that reads a board without
initialising, migrating or writing it. Two failure modes matter and pull in
opposite directions:

* **False positive** — SQLite touches its own shared-memory sidecar when a
  read-only connection attaches to a WAL database. Treating that as a write
  makes the observer unusable on every board that is actually in WAL mode.
* **False negative** — a concurrent writer commits between the "before" and
  "after" identity snapshots and the observer reports a clean read of state
  that no longer exists.

These tests pin both halves: the sidecar the observer ignores, the write it
must still catch, and the promise that observing never initialises or
migrates the database it looks at.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_observe as observe

REQUEST = {
    "task_id": "t_0123abcd",
    "format": "json",
    "no_init": True,
    "no_migrate": True,
    "no_recompute": True,
    "expected_schema_version": 1,
}


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A real board database on disk, created through the normal write path."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect(board="observe-board")
    try:
        kb.create_task(conn, title="observed task", assignee="worker")
    finally:
        conn.close()
    return kb.kanban_db_path(board="observe-board")


def _sidecars(db_path: Path) -> list[str]:
    present = []
    for suffix in ("-wal", "-shm"):
        if Path(f"{db_path}{suffix}").exists():
            present.append(suffix)
    return present


def _master_snapshot(db_path: Path) -> tuple[int, list[str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        names = sorted(
            row[0] for row in conn.execute("SELECT name FROM sqlite_master")
        )
    finally:
        conn.close()
    return version, names


def test_clean_read_reports_no_write(board):
    report = observe.observe_board_read_only(board, REQUEST)

    assert report["total_changes"] == 0
    assert report["target_write_syscalls"] == 0
    assert report["before_files"] == report["after_files"]
    assert report["before_logical_hash"] == report["after_logical_hash"]
    assert report["query_only"] is True
    assert report["uri_mode"] == "ro"


def test_identity_ignores_read_coordination_files_but_keeps_a_pending_wal(board):
    """The sidecars SQLite creates for readers must not look like writes.

    ``-shm`` and an empty ``-wal`` both appear simply because a connection
    attached. The write signal that has to survive is a ``-wal`` holding
    frames, which is where a concurrent commit actually lands.
    """
    report = observe.observe_board_read_only(board, REQUEST)
    watched = {Path(entry["path"]).name for entry in report["before_files"]}

    assert board.name in watched
    assert f"{board.name}-shm" not in watched
    assert f"{board.name}-wal" not in watched, "an empty WAL carries nothing to observe"
    assert report["before_files"] == report["after_files"]

    # Now park a committed transaction in the WAL by leaving a writer open.
    writer = kb.connect(board="observe-board")
    try:
        kb.create_task(writer, title="pending frame", assignee="worker")
        writer.commit()
        wal = Path(f"{board}-wal")
        assert wal.exists() and wal.stat().st_size > 0, (
            "the open writer should be holding frames in the WAL"
        )
        snapshot = {Path(entry["path"]).name for entry in observe._snapshot(board)}
        assert f"{board.name}-wal" in snapshot, (
            "a WAL with frames pending is the write signal and must count"
        )
    finally:
        writer.close()


def test_a_write_landing_during_the_read_is_detected(board, monkeypatch):
    """A concurrent commit must never be reported as a clean observation."""
    real_read_count = observe._read_count
    fired = {"count": 0}

    def read_count_then_write(conn, table):
        value = real_read_count(conn, table)
        if fired["count"] == 0:
            fired["count"] += 1
            writer = kb.connect(board="observe-board")
            try:
                kb.create_task(writer, title="concurrent", assignee="worker")
                writer.commit()
            finally:
                writer.close()
        return value

    monkeypatch.setattr(observe, "_read_count", read_count_then_write)

    with pytest.raises(observe.ObserveError, match="OBSERVE_WRITE_DETECTED"):
        observe.observe_board_read_only(board, REQUEST)

    assert fired["count"] == 1, "the injected writer must have run"


def test_observation_does_not_initialize_or_migrate(board):
    """Observing must not be a migration in disguise."""
    before = _master_snapshot(board)

    observe.observe_board_read_only(board, REQUEST)

    assert _master_snapshot(board) == before


def test_repeated_observation_is_stable(board):
    first = observe.observe_board_read_only(board, REQUEST)
    second = observe.observe_board_read_only(board, REQUEST)

    assert first["database_identity"] == second["database_identity"]
    assert first["task_count"] == second["task_count"]


def test_future_schema_version_is_refused(board):
    conn = sqlite3.connect(board)
    try:
        conn.execute("PRAGMA user_version = 9")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(observe.ObserveError, match="OBSERVE_UNSUPPORTED_SCHEMA"):
        observe.observe_board_read_only(board, REQUEST)


def test_relative_and_missing_paths_are_refused(board, tmp_path):
    with pytest.raises(observe.ObserveError, match="OBSERVE_API_VIOLATION"):
        observe.observe_board_read_only(Path("relative/kanban.db"), REQUEST)

    with pytest.raises(observe.ObserveError, match="OBSERVE_API_VIOLATION"):
        observe.observe_board_read_only(tmp_path / "nope.db", REQUEST)
