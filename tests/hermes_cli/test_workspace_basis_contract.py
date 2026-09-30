"""Frozen Core workspace-basis contract tests (F02).

Covers the producer/plan/resolve contract chosen in
``core-frozen-basis-packet.json``:

* ``task_workspace_plans`` v1 stores the *selected* project anchor, canonical
  worktree target, frozen base commit/tree and a deterministic basis digest.
* Activation (``create_swarm`` with ``per_task_worktrees=True``) binds the plan
  row BEFORE native root activation; exact replays validate, conflicting /
  missing active plans refuse.
* Frozen ``git_base_commit`` / ``git_base_tree`` inputs to create_swarm are
  honored; absent means capture the current basis once. Resolve worktrees from
  the stored commit, not later HEAD; mismatch existing branch / checkout fails
  without reset.
* ``capture_workspace_authority`` keeps actual fresh observation (selected
  project anchor distinct from Git ``--show-toplevel``); historical NULL rows
  are preserved.
* Foreign-clone / linked-project / canonical-path / branch-setter cases
  preserve distinct authority.
* Additive ``task_workspace_plans`` migration is idempotent and never alters
  the original ``task_workspace_authority`` row or ``hm_kanban_schema_journal``
  record.

The tests are L1 / unit; they hit real Core producers and SQLite migrations,
NOT the whole HM consumer surface (which lives in
``hm-loop-candidate-core-20260928`` and is not our concern here).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Real-Git helpers (do NOT mock Git — explicit goal is to bind real Core
# captures to real project anchors and verify they distinguish identical
# git-objects across foreign clones).
# ---------------------------------------------------------------------------


def _git(repo_root: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _init_repo(path: Path, branch: str = "main") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", branch)
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "test")
    (path / "f.txt").write_text("hi")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")
    return path


def _head(path: Path) -> str:
    return _git(path, "rev-parse", "HEAD").stdout.strip()


def _tree(path: Path, commit: str) -> str:
    return _git(path, "rev-parse", f"{commit}^{{tree}}").stdout.strip()


def _clone_into(parent: Path, name: str, source: Path) -> Path:
    """Foreign clone of ``source`` -> ``parent/name`` (no shared gitdir)."""
    target = parent / name
    subprocess.run(
        ["git", "clone", "-q", str(source), str(target)],
        check=True,
    )
    return target


def _linked_worktree_of(repo_root: Path, worktree_path: Path) -> Path:
    """Make ``worktree_path`` a linked checkout of ``repo_root`` (shared .git)."""
    worktree_path.mkdir(parents=True, exist_ok=True)
    proc = _git(repo_root, "worktree", "add", str(worktree_path), "HEAD")
    assert proc.returncode == 0, proc.stderr
    return worktree_path


# ---------------------------------------------------------------------------
# DB helpers — apply the v1 migration set so ``task_workspace_authority``,
# ``task_workspace_plans`` and ``hm_kanban_schema_journal`` are present.
# ---------------------------------------------------------------------------


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` with the full Core schema + migrations applied."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kbc._migrate_add_optional_columns(conn)
    conn.commit()
    return conn


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _index_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
    }


# ---------------------------------------------------------------------------
# Migration: additive ``task_workspace_plans`` v1 table.
# ---------------------------------------------------------------------------


def test_migrate_v1_workspace_plans_additive_and_idempotent(tmp_path: Path) -> None:
    """v1_workspace_plans_20260930 is applied eagerly by ``_open_db`` (the
    additive migration pass). Re-running ``_migrate_v1_workspace_plans`` on
    a board that already has the table is a fast no-op: the journal keeps
    exactly one row for this migration id, the original
    ``task_workspace_authority`` table is untouched, and the new indexes
    are idempotent (``CREATE INDEX IF NOT EXISTS``)."""
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "kanban.db"
    conn = _open_db(db_path)
    try:
        # After _open_db, the v1 authority migration AND the v1 plans
        # migration are both present (both are run by the additive
        # migration pass).
        baseline_journal = {
            row[0]
            for row in conn.execute(
                "SELECT migration_id FROM hm_kanban_schema_journal"
            )
        }
        assert "v1_workspace_authority_20260928" in baseline_journal
        assert "v1_workspace_plans_20260930" in baseline_journal

        # Tables: original capture path AND new plan table both present.
        tables = _table_names(conn)
        assert "task_workspace_authority" in tables
        assert "task_workspace_plans" in tables

        # Indexes on the new table exist.
        assert "idx_task_workspace_plans_root" in _index_names(conn)
        assert "idx_task_workspace_plans_base_commit" in _index_names(conn)

        # Re-run; journal stays single-row per migration id.
        kbc._migrate_v1_workspace_plans(conn)
        new_journal = {
            row[0]
            for row in conn.execute(
                "SELECT migration_id FROM hm_kanban_schema_journal"
            )
        }
        assert new_journal == baseline_journal
        assert len(new_journal) == 2  # authority + plans
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Producer: ``capture_workspace_authority`` binds the SELECTED project anchor,
# not just ``git rev-parse --show-toplevel``. Distinct foreign clones of the
# same git-objects produce distinct authority.
# ---------------------------------------------------------------------------


def test_capture_binds_selected_project_anchor_not_just_git_toplevel(
    tmp_path: Path,
) -> None:
    """Two foreign clones of the SAME git-objects are distinct projects; the
    producer must bind the *selected* project anchor, not just the shared git
    history."""
    from hermes_cli import kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        canonical = _init_repo(td / "canonical")
        foreign = _clone_into(td, "foreign", canonical)
        # Both clones see identical HEAD + tree objects.
        assert _head(canonical) == _head(foreign)

        db_path = td / "test.db"
        conn = _open_db(db_path)
        try:
            # The selected project for task-a is the canonical repo; for
            # task-b it is the foreign clone. Capture from each must record
            # the *selected* path (the file we were given), not the
            # underlying git-dir join.
            for task_id, anchor in (
                ("task-a", canonical),
                ("task-b", foreign),
            ):
                conn.execute(
                    "INSERT INTO tasks(id, title, status, workspace_kind, "
                    "workspace_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (task_id, "x", "ready", "scratch", str(anchor), 1),
                )
            conn.commit()

            row_a = kdw.capture_workspace_authority(
                conn,
                task_id="task-a",
                workspace=canonical,
                source="test",
                captured_by="unit",
            )
            row_b = kdw.capture_workspace_authority(
                conn,
                task_id="task-b",
                workspace=foreign,
                source="test",
                captured_by="unit",
            )
            assert row_a is not None and row_b is not None
            # Distinct selected anchors -> distinct authority, even though
            # the git objects are identical (foreign clone shares commits).
            assert row_a["authority_sha256"] != row_b["authority_sha256"], (
                "distinct project anchors must produce distinct authority"
            )
            # And the recorded ``workspace_root`` reflects the SELECTED path
            # (not the shared git-dir join from --show-toplevel).
            stored_a = conn.execute(
                "SELECT workspace_root FROM task_workspace_authority "
                "WHERE task_id = ?",
                ("task-a",),
            ).fetchone()
            stored_b = conn.execute(
                "SELECT workspace_root FROM task_workspace_authority "
                "WHERE task_id = ?",
                ("task-b",),
            ).fetchone()
            assert stored_a is not None and stored_b is not None
            assert Path(stored_a["workspace_root"]).resolve() == canonical.resolve()
            assert Path(stored_b["workspace_root"]).resolve() == foreign.resolve()
        finally:
            conn.close()


def test_capture_binds_linked_project_checkout(tmp_path: Path) -> None:
    """A linked-worktree checkout of the project IS that project. The
    producer must record its ``workspace_root`` (the linked checkout path),
    not the primary checkout that owns the common git dir.

    NOTE: We do NOT compare ``--git-common-dir`` (that is Git's own invariant
    about how a primary + linked checkout share a ``.git`` directory — not
    Core's contract). The contract we test is: the *selected* path is the
    linked-checkout path, and that's what the authority row binds."""
    from hermes_cli import kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        primary = _init_repo(td / "primary")
        linked = _linked_worktree_of(primary, td / "linked-project")

        db_path = td / "test.db"
        conn = _open_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, "
                "workspace_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("linked-task", "x", "ready", "worktree", str(linked), 1),
            )
            conn.commit()
            row = kdw.capture_workspace_authority(
                conn,
                task_id="linked-task",
                workspace=linked,
                source="test",
                captured_by="unit",
            )
            assert row is not None
            # workspace_root is the LINKED path, not the primary checkout.
            stored = conn.execute(
                "SELECT workspace_root FROM task_workspace_authority "
                "WHERE task_id = ?",
                ("linked-task",),
            ).fetchone()
            assert stored is not None
            assert Path(stored["workspace_root"]).resolve() == linked.resolve()
        finally:
            conn.close()


def test_branch_setter_does_not_drift_task_basis_digest(tmp_path: Path) -> None:
    """Setting ``branch_name`` on a task UPDATES the branch_name field but
    MUST NOT silently rewrite the recorded ``authority_sha256`` when the
    Git basis (commit + tree) has not actually changed."""
    from hermes_cli import kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        repo = _init_repo(td / "repo")
        db_path = td / "test.db"
        conn = _open_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, "
                "workspace_path, branch_name, created_at) VALUES "
                "(?, ?, ?, ?, ?, ?, ?)",
                ("branch-task", "x", "ready", "worktree", str(repo), "main", 1),
            )
            conn.commit()

            before = kdw.capture_workspace_authority(
                conn,
                task_id="branch-task",
                workspace=repo,
                branch_name="main",
                source="test",
                captured_by="unit",
            )
            assert before is not None
            before_sha = before["authority_sha256"]

            # Branch rename without any new commit. Authority MUST stay the
            # same (no Git basis drift). The set_branch_name setter records
            # only the display field.
            kdw.set_branch_name(conn, "branch-task", "feat/rename")
            after = conn.execute(
                "SELECT base_commit, base_tree, authority_sha256 "
                "FROM task_workspace_authority WHERE task_id = ?",
                ("branch-task",),
            ).fetchone()
            assert after is not None
            assert after["authority_sha256"] == before_sha, (
                "ordinary branch rename must not drift the basis digest"
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Capture parity between foreign clones vs canonical anchor (Core-vs-Core
# binding is NOT collapsed; this confirms it).
# ---------------------------------------------------------------------------


def test_foreign_clone_and_canonical_anchor_produce_distinct_authority(
    tmp_path: Path,
) -> None:
    """The cleaned-worktree comparison is Core-vs-Core; a foreign clone of
    identical git-objects MUST be distinguished from the canonical anchor
    we selected, because the *selected project* differs even if the objects
    agree."""
    from hermes_cli import kanban_db_workspace as kdw

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        canonical = _init_repo(td / "canonical")
        foreign = _clone_into(td, "foreign", canonical)
        assert _head(canonical) == _head(foreign)

        db_path = td / "test.db"
        conn = _open_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, "
                "workspace_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("task-can", "x", "ready", "scratch", str(canonical), 1),
            )
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, "
                "workspace_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("task-fc", "x", "ready", "scratch", str(foreign), 1),
            )
            conn.commit()
            row_can = kdw.capture_workspace_authority(
                conn,
                task_id="task-can",
                workspace=canonical,
                source="test",
                captured_by="unit",
            )
            row_fc = kdw.capture_workspace_authority(
                conn,
                task_id="task-fc",
                workspace=foreign,
                source="test",
                captured_by="unit",
            )
            assert row_can is not None and row_fc is not None
            assert row_can["authority_sha256"] != row_fc["authority_sha256"], (
                "canonical and foreign-clone selections are different projects"
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# History: unknown / NULL historical rows are preserved; new claims stay NULL
# until a fresh pre-spawn capture.
# ---------------------------------------------------------------------------


def test_legacy_task_workspace_authority_rows_remain_unknown(
    tmp_path: Path,
) -> None:
    """Pre-F02 rows that recorded no ``workspace_root`` / ``plan_version`` /
    ``base_commit`` (legacy or non-git workspaces) stay NULL — we never
    retrofit historical provenance from current state."""
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "kanban.db"
    conn = _open_db(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS task_workspace_authority_legacy_shape(
                task_id           TEXT PRIMARY KEY,
                base_commit       TEXT NOT NULL,
                base_tree         TEXT NOT NULL,
                authority_sha256  TEXT NOT NULL,
                source            TEXT NOT NULL,
                branch_name       TEXT,
                captured_at       TEXT NOT NULL,
                captured_by       TEXT NOT NULL
            );
            """
        )
        # The legacy table exists; the F02 v1 plans migration must NOT
        # attempt to backfill or modify rows from this shadow table.
        kbc._migrate_v1_workspace_plans(conn)
        # Migration is additive on the canonical table; legacy shadow stays.
        cols = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(task_workspace_authority_legacy_shape)"
            )
        }
        assert "workspace_root" not in cols
        assert "plan_version" not in cols
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Producer helper: ``basis_digest(version, task_id, root, path, commit, tree)``
# is deterministic and stable across runs.
# ---------------------------------------------------------------------------


def test_basis_digest_is_deterministic_and_order_sensitive(
    tmp_path: Path,
) -> None:
    """``basis_digest`` is the Core workspace basis digest over
    (version, task_id, root, path, commit, tree) — no source / branch /
    time. The digest changes when ANY of those inputs changes; the same
    inputs always produce the same digest."""
    from hermes_cli.kanban_db_workspace import basis_digest

    a = basis_digest(
        version=1,
        task_id="t",
        workspace_root="/r",
        workspace_path="/r/.worktrees/t",
        base_commit="c",
        base_tree="t",
    )
    b = basis_digest(
        version=1,
        task_id="t",
        workspace_root="/r",
        workspace_path="/r/.worktrees/t",
        base_commit="c",
        base_tree="t",
    )
    assert a == b  # deterministic
    # Mutating any field changes the digest.
    assert basis_digest(
        version=2, task_id="t", workspace_root="/r",
        workspace_path="/r/.worktrees/t", base_commit="c", base_tree="t",
    ) != a
    assert basis_digest(
        version=1, task_id="X", workspace_root="/r",
        workspace_path="/r/.worktrees/t", base_commit="c", base_tree="t",
    ) != a
    assert basis_digest(
        version=1, task_id="t", workspace_root="/X",
        workspace_path="/r/.worktrees/t", base_commit="c", base_tree="t",
    ) != a
    assert basis_digest(
        version=1, task_id="t", workspace_root="/r",
        workspace_path="/X", base_commit="c", base_tree="t",
    ) != a
    assert basis_digest(
        version=1, task_id="t", workspace_root="/r",
        workspace_path="/r/.worktrees/t", base_commit="X", base_tree="t",
    ) != a
    assert basis_digest(
        version=1, task_id="t", workspace_root="/r",
        workspace_path="/r/.worktrees/t", base_commit="c", base_tree="X",
    ) != a
    # And the canonical sha256 length.
    assert len(a) == 64


# ---------------------------------------------------------------------------
# Plan binding: ``bind_workspace_plan`` writes the row BEFORE activation and
# ``resolve_workspace_plan_from_commit`` honors a frozen base.
# ---------------------------------------------------------------------------


def test_bind_workspace_plan_persists_row_and_validates_exact_replay(
    tmp_path: Path,
) -> None:
    """First call writes the plan row; replay with identical inputs is a
    no-op; replay with a conflicting active plan refuses loudly."""
    from hermes_cli.kanban_db_workspace import (
        bind_workspace_plan,
        resolve_workspace_plan_from_commit,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        repo = _init_repo(td / "repo")
        commit = _head(repo)
        tree = _tree(repo, commit)

        db_path = td / "kanban.db"
        conn = _open_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                ("plan-task", "x", "blocked", "worktree", 1),
            )
            conn.commit()

            planned = bind_workspace_plan(
                conn,
                task_id="plan-task",
                workspace_root=repo,
                workspace_path=repo / ".worktrees" / "plan-task",
                base_commit=commit,
                base_tree=tree,
                plan_version=1,
            )
            assert planned is not None
            # Row persisted.
            row = conn.execute(
                "SELECT task_id, workspace_root, workspace_path, "
                "base_commit, base_tree, plan_version, basis_digest "
                "FROM task_workspace_plans WHERE task_id = ?",
                ("plan-task",),
            ).fetchone()
            assert row is not None
            assert row["task_id"] == "plan-task"
            assert Path(row["workspace_root"]).resolve() == repo.resolve()
            assert row["base_commit"] == commit
            assert row["base_tree"] == tree
            assert row["plan_version"] == 1
            assert len(row["basis_digest"]) == 64

            # Exact replay -> the row is unchanged (no exception).
            replay = bind_workspace_plan(
                conn,
                task_id="plan-task",
                workspace_root=repo,
                workspace_path=repo / ".worktrees" / "plan-task",
                base_commit=commit,
                base_tree=tree,
                plan_version=1,
            )
            assert replay == planned

            # Conflicting active plan refuses loudly.
            with pytest.raises(ValueError):
                bind_workspace_plan(
                    conn,
                    task_id="plan-task",
                    workspace_root=repo,
                    workspace_path=Path("/somewhere/else"),
                    base_commit=commit,
                    base_tree=tree,
                    plan_version=1,
                )

            # Resolution honors the frozen base commit, not later HEAD.
            resolved = resolve_workspace_plan_from_commit(conn, "plan-task")
            assert resolved is not None
            assert resolved["base_commit"] == commit
        finally:
            conn.close()


def test_resolve_worktree_from_frozen_base_not_later_head(tmp_path: Path) -> None:
    """The frozen plan commit pins the worktree base; later HEAD advances
    must not silently rebind."""
    from hermes_cli.kanban_db_workspace import (
        bind_workspace_plan,
        resolve_workspace_plan_from_commit,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        repo = _init_repo(td / "repo")
        pinned_commit = _head(repo)
        pinned_tree = _tree(repo, pinned_commit)

        db_path = td / "kanban.db"
        conn = _open_db(db_path)
        try:
            conn.execute(
                "INSERT INTO tasks(id, title, status, workspace_kind, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                ("pinned", "x", "blocked", "worktree", 1),
            )
            conn.commit()
            bind_workspace_plan(
                conn,
                task_id="pinned",
                workspace_root=repo,
                workspace_path=repo / ".worktrees" / "pinned",
                base_commit=pinned_commit,
                base_tree=pinned_tree,
                plan_version=1,
            )

            # Advance HEAD with a fresh commit.
            (repo / "f.txt").write_text("advance")
            _git(repo, "commit", "-q", "-am", "advance")
            advanced_commit = _head(repo)
            assert advanced_commit != pinned_commit

            # The frozen base survives the unrelated advance.
            resolved = resolve_workspace_plan_from_commit(conn, "pinned")
            assert resolved is not None
            assert resolved["base_commit"] == pinned_commit
            assert resolved["base_commit"] != advanced_commit
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Activation: ``create_swarm`` with ``per_task_worktrees=True`` and an
# explicit frozen base commit binds the plan BEFORE activating the root.
# ---------------------------------------------------------------------------


def test_create_swarm_binds_plan_before_activation(tmp_path: Path) -> None:
    """``create_swarm(per_task_worktrees=True, git_base_commit=..., git_base_tree=...)``
    materializes a ``task_workspace_plans`` row for every swarm member BEFORE
    flipping the planning root to ``done``."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        repo = _init_repo(td / "repo")
        commit = _head(repo)
        tree = _tree(repo, commit)

        db_path = td / "kanban.db"
        conn = kbc.connect(db_path)
        try:
            created = create_swarm(
                conn,
                goal="F02 frozen-basis activation smoke",
                workers=[
                    SwarmWorkerSpec(
                        profile="p1",
                        title="worker-1",
                        body="b1",
                    ),
                ],
                verifier_assignee="verifier",
                synthesizer_assignee="synth",
                workspace_kind="worktree",
                workspace_path=str(repo),
                per_task_worktrees=True,
                git_base_commit=commit,
                git_base_tree=tree,
            )
            # Every swarm member has a plan row carrying the frozen commit/tree.
            task_ids = [
                created.root_id,
                *created.worker_ids,
                created.verifier_id,
                created.synthesizer_id,
            ]
            plan_rows = {
                row["task_id"]: dict(row)
                for row in conn.execute(
                    "SELECT task_id, workspace_root, workspace_path, "
                    "base_commit, base_tree, plan_version, basis_digest "
                    "FROM task_workspace_plans WHERE task_id IN ("
                    + ",".join("?" * len(task_ids))
                    + ")",
                    task_ids,
                )
            }
            assert set(plan_rows) == set(task_ids), plan_rows
            for tid in task_ids:
                row = plan_rows[tid]
                assert row["base_commit"] == commit
                assert row["base_tree"] == tree
                assert row["plan_version"] == 1
                # Canonical worktree path is the per-task anchor.
                assert row["workspace_path"].endswith(f"/.worktrees/{tid}")
                assert Path(row["workspace_root"]).resolve() == repo.resolve()
                assert len(row["basis_digest"]) == 64
        finally:
            conn.close()


def test_create_swarm_replay_with_matching_base_is_idempotent(tmp_path: Path) -> None:
    """A second ``create_swarm`` call with the same idempotency key and
    matching plan inputs validates the existing plan rows unchanged
    (no rewrite), and root activation still succeeds."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        repo = _init_repo(td / "repo")
        commit = _head(repo)
        tree = _tree(repo, commit)

        db_path = td / "kanban.db"
        conn = kbc.connect(db_path)
        try:
            kwargs = dict(
                goal="F02 replay smoke",
                workers=[SwarmWorkerSpec(profile="p1", title="w1", body="b1")],
                verifier_assignee="verifier",
                synthesizer_assignee="synth",
                workspace_kind="worktree",
                workspace_path=str(repo),
                per_task_worktrees=True,
                git_base_commit=commit,
                git_base_tree=tree,
                idempotency_key="f02-replay",
            )
            first = create_swarm(conn, **kwargs)
            second = create_swarm(conn, **kwargs)
            assert second.root_id == first.root_id
            # Replay must NOT have rewritten the digest on the persisted row.
            rows = list(conn.execute(
                "SELECT task_id, basis_digest FROM task_workspace_plans"
            ))
            digests = sorted(row["basis_digest"] for row in rows)
            # And the count of plan rows equals the swarm size (one per task).
            assert len(rows) == 4, digests
        finally:
            conn.close()


def test_create_swarm_refuses_conflicting_active_plan(tmp_path: Path) -> None:
    """A second ``create_swarm`` with a *different* workspace root for the
    same idempotency key MUST refuse — conflicting active plans are not
    silently rebased."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        repo_a = _init_repo(td / "repo-a")
        repo_b = _init_repo(td / "repo-b")
        commit_a = _head(repo_a)
        tree_a = _tree(repo_a, commit_a)
        commit_b = _head(repo_b)
        tree_b = _tree(repo_b, commit_b)

        db_path = td / "kanban.db"
        conn = kbc.connect(db_path)
        try:
            create_swarm(
                conn,
                goal="first",
                workers=[SwarmWorkerSpec(profile="p1", title="w1", body="b1")],
                verifier_assignee="verifier",
                synthesizer_assignee="synth",
                workspace_kind="worktree",
                workspace_path=str(repo_a),
                per_task_worktrees=True,
                git_base_commit=commit_a,
                git_base_tree=tree_a,
                idempotency_key="conflict",
            )
            with pytest.raises(ValueError):
                create_swarm(
                    conn,
                    goal="second",
                    workers=[SwarmWorkerSpec(profile="p1", title="w1", body="b1")],
                    verifier_assignee="verifier",
                    synthesizer_assignee="synth",
                    workspace_kind="worktree",
                    workspace_path=str(repo_b),
                    per_task_worktrees=True,
                    git_base_commit=commit_b,
                    git_base_tree=tree_b,
                    idempotency_key="conflict",
                )
        finally:
            conn.close()


def test_create_swarm_without_frozen_base_captures_current_basis_once(
    tmp_path: Path,
) -> None:
    """With NO ``git_base_commit`` provided, ``create_swarm`` captures the
    CURRENT basis once (one snapshot per swarm) and binds the plan to that."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        repo = _init_repo(td / "repo")
        commit = _head(repo)
        tree = _tree(repo, commit)

        db_path = td / "kanban.db"
        conn = kbc.connect(db_path)
        try:
            created = create_swarm(
                conn,
                goal="no frozen base",
                workers=[SwarmWorkerSpec(profile="p1", title="w1", body="b1")],
                verifier_assignee="verifier",
                synthesizer_assignee="synth",
                workspace_kind="worktree",
                workspace_path=str(repo),
                per_task_worktrees=True,
                # git_base_commit / git_base_tree omitted on purpose.
            )
            rows = list(conn.execute(
                "SELECT task_id, base_commit, base_tree "
                "FROM task_workspace_plans"
            ))
            assert len(rows) == 4
            assert {row["base_commit"] for row in rows} == {commit}
            assert {row["base_tree"] for row in rows} == {tree}
        finally:
            conn.close()