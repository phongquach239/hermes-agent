"""Regression tests for ``task_workspace_authority`` migration + capture path.

Covers:
- ``_migrate_v1_workspace_authority`` is idempotent and adds the table + journal
- ``capture_workspace_authority`` stamps a row with deterministic sha256
- ``claim_task`` populates ``task_runs.workspace_start_commit/tree/authority_sha256``
- ``set_workspace_path`` and ``set_branch_name`` capture authority automatically
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pytest


def _build_git_repo(tmp: str) -> str:
    """Initialise a one-commit git repo under ``tmp/repo`` and return its path."""
    repo_root = os.path.join(tmp, "repo")
    os.makedirs(repo_root)
    subprocess.run(["git", "-C", repo_root, "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", repo_root, "config", "user.email", "test@x"], check=True)
    subprocess.run(["git", "-C", repo_root, "config", "user.name", "test"], check=True)
    Path(repo_root, "f.txt").write_text("hi")
    subprocess.run(["git", "-C", repo_root, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_root, "commit", "-m", "init", "-q"], check=True)
    return repo_root


def _open_test_db(db_path: str) -> sqlite3.Connection:
    """Open a kanban DB at ``db_path`` with schema + migrations applied."""
    import hermes_cli.kanban_db as kb
    import hermes_cli.kanban_db_connect as kdc

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kdc._migrate_add_optional_columns(conn)
    conn.commit()
    return conn


def test_migration_idempotent_creates_table_and_journal(tmp_path: str) -> None:
    """Running migration twice still leaves one journal row and the table."""
    import hermes_cli.kanban_db_connect as kdc

    db_path = os.path.join(str(tmp_path), "test.db")
    conn = _open_test_db(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "task_workspace_authority" in tables
        assert "hm_kanban_schema_journal" in tables

        rows = list(
            conn.execute("SELECT migration_id FROM hm_kanban_schema_journal")
        )
        assert len(rows) == 1
        assert rows[0][0] == "v1_workspace_authority_20260928"

        # Re-run; the journal must still have exactly one row.
        kdc._migrate_v1_workspace_authority(conn)
        rows = list(
            conn.execute("SELECT migration_id FROM hm_kanban_schema_journal")
        )
        assert len(rows) == 1
    finally:
        conn.close()


def test_capture_workspace_authority_stamps_row_with_sha256(tmp_path: str) -> None:
    """A real git repo produces a row whose sha256 matches the deterministic formula."""
    import hermes_cli.kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = _build_git_repo(tmp)
        db_path = os.path.join(tmp, "test.db")
        conn = _open_test_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, workspace_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("task1", "test", "ready", "scratch", repo_root, 1234567890),
            )
            conn.commit()
            row = kdw.capture_workspace_authority(
                conn,
                task_id="task1",
                workspace=Path(repo_root),
                branch_name="main",
                source="test",
                captured_by="unit-test",
            )
            assert row is not None
            head_proc = subprocess.run(
                ["git", "-C", repo_root, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            tree_proc = subprocess.run(
                ["git", "-C", repo_root, "rev-parse", f"{head_proc.stdout.strip()}^{{tree}}"],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = (
                f"task1\n{head_proc.stdout.strip()}\n{tree_proc.stdout.strip()}\ntest\nmain"
            )
            expected_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            assert row["authority_sha256"] == expected_sha
            assert row["base_commit"] == head_proc.stdout.strip()
            assert row["branch_name"] == "main"
        finally:
            conn.close()


def test_claim_task_populates_workspace_start_columns(tmp_path: str) -> None:
    """claim_task writes workspace_start_commit/tree/authority_sha256 into task_runs."""
    import hermes_cli.kanban_db as kb
    import hermes_cli.kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = _build_git_repo(tmp)
        db_path = os.path.join(tmp, "test.db")
        conn = _open_test_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, workspace_path, created_at, assignee) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("task1", "test", "ready", "scratch", repo_root, 1234567890, "test-profile"),
            )
            conn.commit()
            kdw.set_workspace_path(conn, "task1", repo_root)
            kdw.set_branch_name(conn, "task1", "main")
            claimed = kb.claim_task(conn, "task1", ttl_seconds=300)
            assert claimed is not None

            run_row = conn.execute(
                "SELECT workspace_start_commit, workspace_start_tree, workspace_authority_sha256 "
                "FROM task_runs WHERE task_id = ?",
                ("task1",),
            ).fetchone()
            assert run_row is not None
            assert run_row["workspace_start_commit"] is not None
            assert run_row["workspace_start_tree"] is not None
            assert run_row["workspace_authority_sha256"] is not None
            assert len(run_row["workspace_authority_sha256"]) == 64
        finally:
            conn.close()


def test_set_workspace_path_captures_authority_best_effort(tmp_path: str) -> None:
    """set_workspace_path leaves a row in task_workspace_authority when inside a repo."""
    import hermes_cli.kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = _build_git_repo(tmp)
        db_path = os.path.join(tmp, "test.db")
        conn = _open_test_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("task1", "test", "ready", "scratch", 1234567890),
            )
            conn.commit()
            kdw.set_workspace_path(conn, "task1", repo_root)
            row = conn.execute(
                "SELECT base_commit, base_tree, authority_sha256 "
                "FROM task_workspace_authority WHERE task_id = ?",
                ("task1",),
            ).fetchone()
            assert row is not None
            assert row["base_commit"]
            assert row["base_tree"]
            assert row["authority_sha256"]
        finally:
            conn.close()


def test_non_git_workspace_does_not_capture_authority(tmp_path: str) -> None:
    """A workspace outside any git repo leaves task_workspace_authority empty."""
    import hermes_cli.kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as tmp:
        non_git_dir = os.path.join(tmp, "no-git")
        os.makedirs(non_git_dir)
        db_path = os.path.join(tmp, "test.db")
        conn = _open_test_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, workspace_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("task1", "test", "ready", "scratch", non_git_dir, 1234567890),
            )
            conn.commit()
            kdw.set_workspace_path(conn, "task1", non_git_dir)
            row = conn.execute(
                "SELECT 1 FROM task_workspace_authority WHERE task_id = ?",
                ("task1",),
            ).fetchone()
            assert row is None
        finally:
            conn.close()

def test_set_workspace_path_backfills_task_runs_workspace_start_columns(tmp_path: str) -> None:
    """When a task_runs row exists with NULL workspace_start_* (because claim_task
    ran before set_workspace_path), the next set_workspace_path call must
    backfill those columns from the freshly-captured authority row."""
    import hermes_cli.kanban_db as kb
    import hermes_cli.kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = _build_git_repo(tmp)
        db_path = os.path.join(tmp, "test.db")
        conn = _open_test_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, created_at, assignee) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("task1", "test", "ready", "scratch", 1234567890, "test-profile"),
            )
            conn.commit()
            # Simulate the dispatch hot-path order: claim_task inserts task_runs
            # BEFORE set_workspace_path runs.
            claimed = kb.claim_task(conn, "task1", ttl_seconds=300)
            assert claimed is not None
            run_row = conn.execute(
                "SELECT workspace_start_commit, workspace_start_tree, workspace_authority_sha256 "
                "FROM task_runs WHERE task_id = ?",
                ("task1",),
            ).fetchone()
            # At claim-time, set_workspace_path hasn't run yet, so the columns
            # stay NULL even though the migration populated them.
            assert run_row["workspace_start_commit"] is None
            # Now run the dispatcher hot-path step.
            kdw.set_workspace_path(conn, "task1", repo_root)
            run_row = conn.execute(
                "SELECT workspace_start_commit, workspace_start_tree, workspace_authority_sha256 "
                "FROM task_runs WHERE task_id = ?",
                ("task1",),
            ).fetchone()
            # set_workspace_path must have backfilled the columns from the
            # authority row it just wrote.
            assert run_row["workspace_start_commit"] is not None
            assert run_row["workspace_start_tree"] is not None
            assert run_row["workspace_authority_sha256"] is not None
            assert len(run_row["workspace_authority_sha256"]) == 64
        finally:
            conn.close()

@pytest.fixture
def claimed_workspace_attempt(tmp_path):
    import hermes_cli.kanban_db as kb

    repo = Path(_build_git_repo(str(tmp_path)))
    conn = _open_test_db(str(tmp_path / "attempts.db"))
    conn.execute(
        "INSERT INTO tasks(id,title,status,workspace_kind,created_at,assignee) "
        "VALUES ('task1','workspace claim','ready','worktree',1,'test-profile')"
    )
    conn.commit()
    assert kb.claim_task(conn, "task1", claimer="current-claim") is not None
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id='task1'"
    ).fetchone()[0]
    try:
        yield conn, repo, run_id
    finally:
        conn.close()


def _attempt_authority(conn, run_id):
    return tuple(conn.execute(
        "SELECT workspace_start_commit,workspace_start_tree,workspace_authority_sha256 "
        "FROM task_runs WHERE id=?", (run_id,),
    ).fetchone())


def test_workspace_backfill_preserves_history_and_replay(claimed_workspace_attempt):
    import hermes_cli.kanban_db_workspace as kdw

    conn, repo, current_id = claimed_workspace_attempt
    historical = []
    for status, outcome in [("done", "completed"), ("failed", "failed"),
                            ("failed", "rate_limited"), ("running", None)]:
        historical.append(conn.execute(
            "INSERT INTO task_runs(task_id,status,outcome,started_at,ended_at) "
            "VALUES ('task1',?,?,1,?)",
            (status, outcome, None if status == "running" else 2),
        ).lastrowid)
    conn.commit()
    before = {row_id: _attempt_authority(conn, row_id) for row_id in historical}
    kdw.set_workspace_path(conn, "task1", repo)
    current = _attempt_authority(conn, current_id)
    assert all(current), "the actual pre-spawn claim must receive full authority"
    assert {row_id: _attempt_authority(conn, row_id) for row_id in historical} == before

    (repo / "f.txt").write_text("a later commit must not restamp a started attempt")
    subprocess.run(["git", "-C", str(repo), "commit", "-am", "later", "-q"], check=True)
    kdw.set_workspace_path(conn, "task1", repo)
    assert _attempt_authority(conn, current_id) == current
    assert {row_id: _attempt_authority(conn, row_id) for row_id in historical} == before


@pytest.mark.parametrize("case", [
    "partial_tree", "partial_hash", "terminal_status", "terminal_outcome", "ended",
    "task_terminal", "task_spawned", "run_spawned", "claim_mismatch", "claim_expired",
    "capture_none", "capture_error", "claim_changed_during_capture", "stamp_sql_failure",
])
def test_workspace_backfill_refuses_ineligible_attempts(
    claimed_workspace_attempt, monkeypatch, case,
):
    import hermes_cli.kanban_db_workspace as kdw

    conn, repo, run_id = claimed_workspace_attempt
    mutations = {
        "partial_tree": "UPDATE task_runs SET workspace_start_tree='existing-tree' WHERE id=?",
        "partial_hash": "UPDATE task_runs SET workspace_authority_sha256='existing-hash' WHERE id=?",
        "terminal_status": "UPDATE task_runs SET status='done' WHERE id=?",
        "terminal_outcome": "UPDATE task_runs SET outcome='completed' WHERE id=?",
        "ended": "UPDATE task_runs SET ended_at=2 WHERE id=?",
        "run_spawned": "UPDATE task_runs SET worker_pid=12345 WHERE id=?",
        "claim_mismatch": "UPDATE task_runs SET claim_lock='different-claim' WHERE id=?",
        "claim_expired": "UPDATE task_runs SET claim_expires=0 WHERE id=?",
    }
    if case in mutations:
        conn.execute(mutations[case], (run_id,))
    if case == "task_terminal":
        conn.execute("UPDATE tasks SET status='done' WHERE id='task1'")
    if case == "task_spawned":
        conn.execute("UPDATE tasks SET worker_pid=12345 WHERE id='task1'")
    conn.commit()
    before = _attempt_authority(conn, run_id)
    if case == "stamp_sql_failure":
        conn.execute(
            "CREATE TRIGGER reject_start_stamp BEFORE UPDATE OF workspace_start_commit "
            "ON task_runs BEGIN SELECT RAISE(ABORT, 'injected start-authority write failure'); END"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="injected start-authority write failure"):
            kdw.set_workspace_path(conn, "task1", repo)
        assert _attempt_authority(conn, run_id) == before
        return
    extra_runs = []
    real_capture = kdw.capture_workspace_authority
    if case in {"capture_none", "capture_error"}:
        # A stale task-level row must not be re-used after this capture fails.
        assert real_capture(conn, task_id="task1", workspace=repo,
                            source="earlier", captured_by="fixture")
        def no_capture(*args, **kwargs):
            if case == "capture_error":
                raise sqlite3.OperationalError("injected capture failure")
            return None
        monkeypatch.setattr(kdw, "capture_workspace_authority", no_capture)
    elif case == "claim_changed_during_capture":
        def capture_then_replace(*args, **kwargs):
            result = real_capture(*args, **kwargs)
            new_id = conn.execute(
                "INSERT INTO task_runs(task_id,status,claim_lock,claim_expires,started_at) "
                "SELECT task_id,'running','replacement',claim_expires,started_at "
                "FROM task_runs WHERE id=?", (run_id,),
            ).lastrowid
            conn.execute(
                "UPDATE tasks SET current_run_id=?,claim_lock='replacement' WHERE id='task1'",
                (new_id,),
            )
            conn.commit()
            extra_runs.append(new_id)
            return result
        monkeypatch.setattr(kdw, "capture_workspace_authority", capture_then_replace)
    kdw.set_workspace_path(conn, "task1", repo)
    assert _attempt_authority(conn, run_id) == before
    for new_id in extra_runs:
        assert _attempt_authority(conn, new_id) == (None, None, None)
