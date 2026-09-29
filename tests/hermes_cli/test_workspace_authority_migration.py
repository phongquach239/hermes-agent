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
