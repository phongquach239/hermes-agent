"""Fail-closed pre-spawn gate for ``kanban_db_dispatch._dispatch_lane_task``.

Reproduction for F02: a worktree row whose pre-spawn capture/preparation
fails (or whose current claim is no longer fresh) MUST NOT reach ``spawn_fn``,
and MUST NOT mutate any successor / historical run. Non-Git and non-worktree
flows stay valid (a missing authority row is not refusal for them).

These tests exercise the real Core claim/resolve/setters and Core DB schema;
the worker callback is a spy. No model calls, no live DB writes outside the
fresh SQLite file each test opens.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kdc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_db_workspace as workspace


# ---------------------------------------------------------------------------
# Shared fixtures: native real Git repo + native real Core schema.
# ---------------------------------------------------------------------------


def _make_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "offline-fixture"],
                   check=True, capture_output=True)
    (repo / "file.txt").write_text("isolated fixture\n")
    subprocess.run(["git", "-C", str(repo), "add", "file.txt"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True, capture_output=True)


def _open_board_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kdc._migrate_add_optional_columns(conn)
    conn.commit()
    return conn


def _run_columns(conn: sqlite3.Connection, run_id: int) -> tuple:
    row = conn.execute(
        "SELECT workspace_start_commit, workspace_start_tree, workspace_authority_sha256 "
        "FROM task_runs WHERE id=?",
        (int(run_id),),
    ).fetchone()
    return (row["workspace_start_commit"], row["workspace_start_tree"],
            row["workspace_authority_sha256"])


def _claim_state(conn: sqlite3.Connection, task_id: str) -> dict:
    row = conn.execute(
        "SELECT claim_lock, claim_expires, current_run_id, status, worker_pid "
        "FROM tasks WHERE id=?",
        (task_id,),
    ).fetchone()
    return dict(row) if row else {}


def _task_history_run_ids(conn: sqlite3.Connection, task_id: str) -> list[int]:
    return [int(r[0]) for r in conn.execute(
        "SELECT id FROM task_runs WHERE task_id=? ORDER BY id", (task_id,),
    ).fetchall()]


# ---------------------------------------------------------------------------
# Gate path: drive the real dispatcher with a spy spawn_fn.
# ---------------------------------------------------------------------------


def _drive_dispatch(conn: sqlite3.Connection, task_id: str, *,
                    lane: str, spy: dict, monkeypatch,
                    assignee: str = "fixture-worker") -> dispatch.DispatchResult:
    """Invoke the real ``_dispatch_lane_task`` with the spy's spawn_fn."""
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda _: True)
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    result = dispatch.DispatchResult()
    dispatch._dispatch_lane_task(
        conn, row, assignee, result,
        lane=lane, dry_run=False,
        ttl_seconds=300, board=None, failure_limit=3,
        spawn_fn=spy["fn"], per_profile_cap=None, per_profile_running={},
    )
    return result


def _spy_spawn(conn: sqlite3.Connection) -> dict:
    calls: list[dict] = []

    def fn(task, path, *args, **kwargs):
        started = conn.execute(
            "SELECT workspace_start_commit, workspace_start_tree, workspace_authority_sha256 "
            "FROM task_runs WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
            (task.id,),
        ).fetchone()
        calls.append({
            "task_id": task.id,
            "path": path,
            "authority": (
                started["workspace_start_commit"] if started else None,
                started["workspace_start_tree"] if started else None,
                started["workspace_authority_sha256"] if started else None,
            ),
        })
        return 0

    return {"fn": fn, "calls": calls}


# ---------------------------------------------------------------------------
# Positive control: capture succeeds -> the fresh run carries authority.
# ---------------------------------------------------------------------------


def test_worktree_spawn_with_fresh_capture_succeeds(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('capture-positive','fixture','ready','worktree',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        spy = _spy_spawn(conn)
        result = _drive_dispatch(conn, "capture-positive", lane="ready", spy=spy, monkeypatch=monkeypatch)
        assert len(spy["calls"]) == 1, "valid fresh capture must still admit the fixture worker"
        commit, tree, sha = spy["calls"][0]["authority"]
        assert commit and tree and sha, "positive control must really have captured start evidence"
        assert result.spawned and result.spawned[0][0] == "capture-positive"
    finally:
        conn.close()


def test_review_worktree_spawn_with_fresh_capture_succeeds(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('capture-positive-review','fixture','review','worktree',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        spy = _spy_spawn(conn)
        result = _drive_dispatch(conn, "capture-positive-review", lane="review", spy=spy, monkeypatch=monkeypatch)
        assert len(spy["calls"]) == 1, "review lane with fresh capture must spawn"
        commit, tree, sha = spy["calls"][0]["authority"]
        assert commit and tree and sha
        assert result.spawned and result.spawned[0][0] == "capture-positive-review"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Capture failure -> no spawn.
# ---------------------------------------------------------------------------


def test_worktree_capture_returning_none_blocks_spawn(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('capture-none','fixture','ready','worktree',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        monkeypatch.setattr(workspace, "capture_workspace_authority",
                            lambda *a, **kw: None)
        spy = _spy_spawn(conn)
        result = _drive_dispatch(conn, "capture-none", lane="ready", spy=spy, monkeypatch=monkeypatch)
        assert spy["calls"] == [], f"worktree worker reached spawn without fresh authority: {spy['calls']}"
        assert result.spawned == []
        # The single attempt we opened must have been closed with
        # ``spawn_failed`` — not left dangling, not stamping a successor.
        runs = _task_history_run_ids(conn, "capture-none")
        assert len(runs) == 1
        outcome = conn.execute(
            "SELECT outcome FROM task_runs WHERE id=?", (runs[0],),
        ).fetchone()["outcome"]
        assert outcome == "spawn_failed"
    finally:
        conn.close()


def test_worktree_capture_raising_blocks_spawn_and_records_failure(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('capture-err','fixture','ready','worktree',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        def _boom(*a, **kw):
            raise sqlite3.OperationalError("fixture-only capture write failure")
        monkeypatch.setattr(workspace, "capture_workspace_authority", _boom)
        spy = _spy_spawn(conn)
        result = _drive_dispatch(conn, "capture-err", lane="ready", spy=spy, monkeypatch=monkeypatch)
        assert spy["calls"] == [], "raised capture must not let the spy run"
        assert result.spawned == []
        runs = _task_history_run_ids(conn, "capture-err")
        assert len(runs) == 1
        outcome = conn.execute(
            "SELECT outcome FROM task_runs WHERE id=?", (runs[0],),
        ).fetchone()["outcome"]
        assert outcome == "spawn_failed"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Start-write failure inside set_workspace_path must not let the spy run.
# ---------------------------------------------------------------------------


def test_worktree_start_write_error_blocks_spawn(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('capture-stamp-err','fixture','ready','worktree',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        # Cause the run-row backfill UPDATE in set_workspace_path to fail after
        # capture_workspace_authority itself succeeds. This models a DB write
        # failure that the current product silently swallows.
        conn.execute(
            "CREATE TRIGGER reject_start_stamp BEFORE UPDATE OF workspace_start_commit "
            "ON task_runs BEGIN SELECT RAISE(ABORT, 'injected start-authority write failure'); END"
        )
        conn.commit()
        spy = _spy_spawn(conn)
        result = _drive_dispatch(conn, "capture-stamp-err", lane="ready", spy=spy, monkeypatch=monkeypatch)
        assert spy["calls"] == [], "start-write failure must not let the spy run"
        assert result.spawned == []
        runs = _task_history_run_ids(conn, "capture-stamp-err")
        assert len(runs) == 1
        outcome = conn.execute(
            "SELECT outcome FROM task_runs WHERE id=?", (runs[0],),
        ).fetchone()["outcome"]
        assert outcome == "spawn_failed"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Non-Git / scratch control: a missing authority row is NOT refusal.
# ---------------------------------------------------------------------------


def test_non_git_scratch_spawn_is_not_blocked(tmp_path, monkeypatch):
    repo = tmp_path / "no-git"
    repo.mkdir()
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('non-git','fixture','ready','scratch',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        spy = _spy_spawn(conn)
        result = _drive_dispatch(conn, "non-git", lane="ready", spy=spy, monkeypatch=monkeypatch)
        assert len(spy["calls"]) == 1, "non-Git scratch must still spawn; missing authority is not refusal"
        assert result.spawned and result.spawned[0][0] == "non-git"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Claim is no longer fresh -> no spawn, no successor / history mutation.
# ---------------------------------------------------------------------------


def _mutate_claim_then_set_workspace(monkeypatch, repo: Path, *, new_claim: Optional[str],
                                      new_expires: Optional[int]):
    """Monkeypatch set_workspace_path so that, mid-dispatch (after claim_task
    opened the run), the task's claim_lock / claim_expires is replaced before
    the gate runs. ``new_claim=None`` means "leave claim_lock alone" (only
    mutate expires); pass an explicit string to replace claim_lock too.
    """
    import hermes_cli.kanban_db_workspace as kdw
    real = kdw.set_workspace_path

    def _mut_then_set(conn, task_id, path):
        # 1. Mutate the claim on the LIVE tasks row to simulate a successor /
        #    expired-lease situation that arose AFTER claim_task returned.
        if new_claim is not None:
            conn.execute(
                "UPDATE tasks SET claim_lock=?, claim_expires=? WHERE id=?",
                (str(new_claim), int(__import__("time").time()) + 600, task_id),
            )
        if new_expires is not None:
            conn.execute(
                "UPDATE tasks SET claim_expires=? WHERE id=?",
                (int(new_expires), task_id),
            )
        conn.commit()
        # 2. Run the real setter so workspace_path / authority are stamped
        #    exactly as production would.
        return real(conn, task_id, path)

    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._kbw.set_workspace_path", _mut_then_set)


def test_worktree_with_changed_claim_blocks_spawn(tmp_path, monkeypatch):
    """If the task's claim_lock no longer matches the row the dispatcher is
    acting on (e.g. another dispatcher reclaimed it), the pre-spawn gate must
    refuse — without rewriting any successor or historical run row."""
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('claim-changed','fixture','ready','worktree',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        _mutate_claim_then_set_workspace(monkeypatch, repo, new_claim="successor-claim",
                                          new_expires=None)
        spy = _spy_spawn(conn)
        result = _drive_dispatch(conn, "claim-changed", lane="ready", spy=spy, monkeypatch=monkeypatch)
        assert spy["calls"] == [], "a changed claim must not let the spy run"
        assert result.spawned == []
        # The successor's claim_lock must remain exactly as it was when we
        # observed it (no rewrite from the refusing dispatcher).
        claim_row = conn.execute(
            "SELECT claim_lock, claim_expires FROM tasks WHERE id='claim-changed'",
        ).fetchone()
        assert claim_row["claim_lock"] == "successor-claim", \
            "must not rewrite the successor's claim_lock"
        # No fresh authority columns stamped on the run row.
        run_row = conn.execute(
            "SELECT id, workspace_start_commit, workspace_start_tree, "
            "workspace_authority_sha256 FROM task_runs WHERE task_id='claim-changed'"
        ).fetchone()
        assert run_row is not None
        assert run_row["workspace_start_commit"] is None
        assert run_row["workspace_start_tree"] is None
        assert run_row["workspace_authority_sha256"] is None
    finally:
        conn.close()


def test_worktree_with_expired_claim_blocks_spawn(tmp_path, monkeypatch):
    """An expired claim is no longer 'the one just claimed' — the gate must
    refuse without restarting the run or stamping authority."""
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('claim-expired','fixture','ready','worktree',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        _mutate_claim_then_set_workspace(monkeypatch, repo, new_claim=None,
                                          new_expires=1)
        spy = _spy_spawn(conn)
        result = _drive_dispatch(conn, "claim-expired", lane="ready", spy=spy, monkeypatch=monkeypatch)
        assert spy["calls"] == [], "an expired claim must not let the spy run"
        assert result.spawned == []
        # The expired claim must NOT have been extended or refreshed — leave it
        # exactly as the lease-expiry scenario left it.
        claim_row = conn.execute(
            "SELECT claim_lock, claim_expires FROM tasks WHERE id='claim-expired'",
        ).fetchone()
        assert claim_row["claim_expires"] == 1, "must not extend the expired claim"
        # No fresh authority columns stamped on the run row.
        run_row = conn.execute(
            "SELECT id, workspace_start_commit, workspace_start_tree, "
            "workspace_authority_sha256 FROM task_runs WHERE task_id='claim-expired'"
        ).fetchone()
        assert run_row is not None
        assert run_row["workspace_start_commit"] is None
        assert run_row["workspace_start_tree"] is None
        assert run_row["workspace_authority_sha256"] is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Gate state-drift coverage: every form of stale / terminal / already-spawned
# / lost-pointer refusal admitted by the gate. Each case must close the gate
# WITHOUT mutating tasks / task_runs / task_events (no history or successor
# rewrite). R1 missed these because the gate only checked
# tasks.claim_lock / tasks.claim_expires; the operator-side fixtures mutate
# task_runs and the tasks pointer as well.
# ---------------------------------------------------------------------------


def _state_snapshot(conn, task_id: str) -> dict:
    """Snapshot of tasks+task_runs+task_events for the task under test. The
    gate must leave this byte-for-byte unchanged on every form of state-
    drift refusal (silent refusal, no DB write)."""
    return {
        "tasks": [tuple(r) for r in conn.execute(
            "SELECT * FROM tasks WHERE id=? ORDER BY rowid", (task_id,),
        ).fetchall()],
        "task_runs": [tuple(r) for r in conn.execute(
            "SELECT * FROM task_runs WHERE task_id=? ORDER BY rowid", (task_id,),
        ).fetchall()],
        "task_events": [tuple(r) for r in conn.execute(
            "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,),
        ).fetchall()],
    }


def _fresh_worktree_task(conn, task_id: str, repo: Path) -> int:
    """Insert a ready worktree row whose current_run points at a fresh
    running claim/run. Returns the ``current_run_id`` the dispatcher will
    see; tests then mutate specific columns to inject a defect."""
    conn.execute(
        "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
        "VALUES (?, 'fixture', 'ready', 'worktree', ?, 1, 'fixture-worker')",
        (task_id, str(repo)),
    )
    conn.commit()
    claimed = kb.claim_task(conn, task_id, ttl_seconds=300)
    assert claimed is not None, "fixture bootstrap must claim the task"
    run_id = claimed.current_run_id
    workspace.set_workspace_path(conn, task_id, str(repo))
    conn.commit()
    return int(run_id)


@pytest.fixture
def drift_task(tmp_path, monkeypatch):
    """Bootstrap a worktree row whose current-claim run is fully stamped;
    the per-test mutator then corrupts ONE column to inject a defect.
    Yields ``(conn, run_id)``. Tests must ``conn.close()`` themselves in
    finally; the fixture does not own the connection because some
    independent-connection tests close it to exercise a sibling."""
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda _: True)
    run_id = _fresh_worktree_task(conn, "gate-review", repo)
    yield conn, run_id, repo


def _drive_dispatch_with_spy(conn, task_id: str, *, lane: str, monkeypatch,
                              spawn_calls: list) -> "dispatch.DispatchResult":
    """Drive ``_dispatch_lane_task`` with a spy that records every spawn
    call. State-drift refusals MUST result in zero spawn calls."""
    def spy(task, path, *args, **kwargs):
        spawn_calls.append(task.id)
        return 0
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda _: True)
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    result = dispatch.DispatchResult()
    dispatch._dispatch_lane_task(
        conn, row, "fixture-worker", result,
        lane=lane, dry_run=False,
        ttl_seconds=300, board=None, failure_limit=3,
        spawn_fn=spy, per_profile_cap=None, per_profile_running={},
    )
    return result


@pytest.mark.parametrize("defect", [
    "current_pointer_lost",
    "run_ended",
    "run_already_spawned",
    "task_already_spawned",
    "run_expired",
    "task_expired",
])
def test_worktree_state_drift_blocks_spawn_preserves_history(
        ready_task, monkeypatch, defect):
    """Inject drift after the dispatcher's claim/preparation, before admission."""
    conn, _repo = ready_task
    real_branch = workspace.set_branch_name
    real_gate = dispatch._check_pre_spawn_authority
    before, reasons = [], []

    def inject_after_prepare(c, task_id, branch):
        real_branch(c, task_id, branch)
        run_id = kb.get_task(c, task_id).current_run_id
        changes = {
            "current_pointer_lost": ("UPDATE tasks SET current_run_id=NULL WHERE id=?", task_id),
            "run_ended": ("UPDATE task_runs SET status='done', outcome='completed', ended_at=1 WHERE id=?", run_id),
            "run_already_spawned": ("UPDATE task_runs SET worker_pid=12345 WHERE id=?", run_id),
            "task_already_spawned": ("UPDATE tasks SET worker_pid=12345 WHERE id=?", task_id),
            "run_expired": ("UPDATE task_runs SET claim_expires=1 WHERE id=?", run_id),
            "task_expired": ("UPDATE tasks SET claim_expires=1 WHERE id=?", task_id),
        }
        sql, value = changes[defect]
        c.execute(sql, (value,))
        c.commit()
        before.append(_state_snapshot(c, task_id))

    def observe_gate(c, claimed):
        refusal = real_gate(c, claimed)
        reasons.append(refusal[0] if refusal else None)
        return refusal

    monkeypatch.setattr(workspace, "set_branch_name", inject_after_prepare)
    monkeypatch.setattr(dispatch, "_check_pre_spawn_authority", observe_gate)
    calls = []
    result = _drive_dispatch_with_spy(conn, "gate-review", lane="ready",
                                    monkeypatch=monkeypatch, spawn_calls=calls)
    assert len(before) == 1, "native claim/preparation and injection must run once"
    expected = "run_status_not_running" if defect == "run_ended" else defect
    assert reasons == [expected]
    assert calls == [] and result.spawned == [] and result.auto_blocked == []
    assert _state_snapshot(conn, "gate-review") == before[0]


def test_worktree_with_full_successor_swap_blocks_spawn(ready_task, monkeypatch):
    """A native successor replaces the original after preparation, before gate."""
    conn, _repo = ready_task
    real_branch = workspace.set_branch_name
    real_gate = dispatch._check_pre_spawn_authority
    before, identities, reasons = [], [], []

    def replace_after_prepare(c, task_id, branch):
        real_branch(c, task_id, branch)
        original = kb.get_task(c, task_id)
        dispatch._record_task_failure(
            c, task_id, "fixture predecessor failure", outcome="spawn_failed",
            failure_limit=10, release_claim=True, end_run=True,
        )
        successor = kb.claim_task(c, task_id, ttl_seconds=300, claimer="fixture-successor")
        assert successor is not None
        identities.append((original.current_run_id, original.claim_lock,
                           successor.current_run_id, successor.claim_lock))
        before.append(_state_snapshot(c, task_id))

    def observe_gate(c, claimed):
        refusal = real_gate(c, claimed)
        reasons.append(refusal[0] if refusal else None)
        return refusal

    monkeypatch.setattr(workspace, "set_branch_name", replace_after_prepare)
    monkeypatch.setattr(dispatch, "_check_pre_spawn_authority", observe_gate)
    calls = []
    result = _drive_dispatch_with_spy(conn, "gate-review", lane="ready",
                                    monkeypatch=monkeypatch, spawn_calls=calls)
    assert len(before) == len(identities) == 1
    old_run, old_lock, new_run, new_lock = identities[0]
    assert old_run != new_run and old_lock != new_lock
    assert reasons == ["current_pointer_lost"]
    assert calls == [] and result.spawned == [] and result.auto_blocked == []
    assert _state_snapshot(conn, "gate-review") == before[0]


def test_worktree_preparation_exception_preserves_native_successor(
        ready_task, monkeypatch):
    """The real bounded exception handler must preserve a native successor."""
    conn, _repo = ready_task
    real_set = workspace.set_workspace_path
    real_failure = dispatch._record_task_failure
    real_gate = dispatch._check_pre_spawn_authority
    before, originals, identities, failures, gates = [], [], [], [], []

    def observe_failure(c, task_id, error, **kwargs):
        failures.append((error, kwargs.get("expected_run_id"),
                         kwargs.get("expected_claim_lock")))
        return real_failure(c, task_id, error, **kwargs)

    def replace_then_raise(c, task_id, path):
        real_set(c, task_id, path)
        original = kb.get_task(c, task_id)
        originals.append((original.current_run_id, original.claim_lock))
        dispatch._record_task_failure(
            c, task_id, "fixture replacement", outcome="spawn_failed",
            failure_limit=10, release_claim=True, end_run=True,
        )
        successor = kb.claim_task(c, task_id, ttl_seconds=300, claimer="fixture-successor")
        assert successor is not None
        identities.append((successor.current_run_id, successor.claim_lock))
        before.append(_state_snapshot(c, task_id))
        raise sqlite3.OperationalError("offline preparation exception after replacement")

    def observe_gate(c, claimed):
        gates.append(claimed.current_run_id)
        return real_gate(c, claimed)

    monkeypatch.setattr(workspace, "set_workspace_path", replace_then_raise)
    monkeypatch.setattr(dispatch, "_record_task_failure", observe_failure)
    monkeypatch.setattr(dispatch, "_check_pre_spawn_authority", observe_gate)
    calls = []
    result = _drive_dispatch_with_spy(conn, "gate-review", lane="ready",
                                    monkeypatch=monkeypatch, spawn_calls=calls)
    assert len(before) == len(originals) == len(identities) == 1
    old_run, old_lock = originals[0]
    new_run, new_lock = identities[0]
    assert old_run != new_run and old_lock != new_lock
    assert failures == [
        ("fixture replacement", None, None),
        ("workspace_authority: offline preparation exception after replacement", old_run, old_lock),
    ]
    assert gates == []
    assert calls == [] and result.spawned == [] and result.auto_blocked == []
    assert _state_snapshot(conn, "gate-review") == before[0]


def test_worktree_gate_refusal_rechecks_owner_after_gate_read(
        ready_task, monkeypatch):
    """Replace ownership after actual missing-authority observation, before failure."""
    conn, _repo = ready_task
    real_gate = dispatch._check_pre_spawn_authority
    real_failure = dispatch._record_task_failure
    before, originals, identities, failures, reasons = [], [], [], [], []

    def observe_failure(c, task_id, error, **kwargs):
        failures.append((error, kwargs.get("expected_run_id"),
                         kwargs.get("expected_claim_lock")))
        return real_failure(c, task_id, error, **kwargs)

    def replace_after_observation(c, claimed):
        originals.append((claimed.current_run_id, claimed.claim_lock))
        c.execute(
            "UPDATE task_runs SET workspace_start_commit=NULL, workspace_start_tree=NULL, "
            "workspace_authority_sha256=NULL WHERE id=?", (claimed.current_run_id,),
        )
        c.commit()
        refusal = real_gate(c, claimed)
        reasons.append(refusal[0] if refusal else None)
        assert reasons == ["missing_authority_evidence"]
        dispatch._record_task_failure(
            c, claimed.id, "fixture replacement", outcome="spawn_failed",
            failure_limit=10, release_claim=True, end_run=True,
        )
        successor = kb.claim_task(c, claimed.id, ttl_seconds=300, claimer="fixture-successor")
        assert successor is not None
        identities.append((successor.current_run_id, successor.claim_lock))
        before.append(_state_snapshot(c, claimed.id))
        return refusal

    monkeypatch.setattr(dispatch, "_record_task_failure", observe_failure)
    monkeypatch.setattr(dispatch, "_check_pre_spawn_authority", replace_after_observation)
    calls = []
    result = _drive_dispatch_with_spy(conn, "gate-review", lane="ready",
                                    monkeypatch=monkeypatch, spawn_calls=calls)
    assert len(before) == len(originals) == len(identities) == 1
    old_run, old_lock = originals[0]
    new_run, new_lock = identities[0]
    assert old_run != new_run and old_lock != new_lock
    assert reasons == ["missing_authority_evidence"]
    assert failures == [
        ("fixture replacement", None, None),
        ("workspace_authority: missing_authority_evidence", old_run, old_lock),
    ]
    assert calls == [] and result.spawned == [] and result.auto_blocked == []
    assert _state_snapshot(conn, "gate-review") == before[0]


# ---------------------------------------------------------------------------
# Negative-control coverage: stamp SQL error closes the owned attempt
# through native failure (the helper's recheck passes because we still
# own the task). Independent of the state-drift set above; reuses the
# same bootstrap.
# ---------------------------------------------------------------------------


def test_worktree_review_state_drift_blocks_spawn(ready_task, monkeypatch):
    """Let native claim_review_task open the attempt before injecting expired run."""
    conn, _repo = ready_task
    conn.execute("UPDATE tasks SET status='review' WHERE id='gate-review'")
    conn.commit()
    initial = kb.get_task(conn, "gate-review")
    assert initial.claim_lock is None and initial.current_run_id is None
    real_branch = workspace.set_branch_name
    real_gate = dispatch._check_pre_spawn_authority
    before, reasons = [], []

    def expire_after_prepare(c, task_id, branch):
        real_branch(c, task_id, branch)
        claimed = kb.get_task(c, task_id)
        assert claimed.claim_lock and claimed.current_run_id
        c.execute("UPDATE task_runs SET claim_expires=1 WHERE id=?", (claimed.current_run_id,))
        c.commit()
        before.append(_state_snapshot(c, task_id))

    def observe_gate(c, claimed):
        refusal = real_gate(c, claimed)
        reasons.append(refusal[0] if refusal else None)
        return refusal

    monkeypatch.setattr(workspace, "set_branch_name", expire_after_prepare)
    monkeypatch.setattr(dispatch, "_check_pre_spawn_authority", observe_gate)
    calls = []
    result = _drive_dispatch_with_spy(conn, "gate-review", lane="review",
                                    monkeypatch=monkeypatch, spawn_calls=calls)
    assert len(before) == 1
    assert reasons == ["run_expired"]
    assert calls == [] and result.spawned == [] and result.auto_blocked == []
    assert _state_snapshot(conn, "gate-review") == before[0]


# ---------------------------------------------------------------------------
# R3-binding repair: ``task_runs.task_id`` must match ``claimed.id`` (gate)
# and ``task_id`` (failure helper); a partial expected-owner pair must
# refuse without mutation or TypeError. Migrated from the Main R3 binding
# spec probes; foreign-task_id mutations are explicit invalid-state
# fixtures, not production repair paths.
# ---------------------------------------------------------------------------


def _make_foreign_task_row(conn, task_id: str) -> None:
    """Insert a sibling ``tasks`` row under a different id so the foreign
    ``task_runs.task_id`` rewrite below names a REAL row, not a dangling
    reference. Operates on the same DB; the foreign row is never touched
    by the gate or the helper.
    """
    conn.execute(
        "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at) "
        "VALUES (?, 'foreign fixture', 'ready', 'worktree', '/tmp/foreign', 1)",
        (task_id,),
    )
    conn.commit()


def _make_foreign_run(conn, claimed) -> None:
    """Explicit foreign-state fixture: rewrite ``task_runs.task_id`` on
    the row the gate / helper is acting on so it names a DIFFERENT task.
    The sibling ``tasks`` row is inserted by ``_make_foreign_task_row``
    above (a real FK target, not a dangling reference). Real claim and
    capture are intact; only the ``task_id`` column is mutated.
    """
    foreign_id = "foreign-task"
    _make_foreign_task_row(conn, foreign_id)
    conn.execute(
        "UPDATE task_runs SET task_id=? WHERE id=?",
        (foreign_id, int(claimed.current_run_id)),
    )
    conn.commit()
    row = conn.execute(
        "SELECT task_id FROM task_runs WHERE id=?",
        (int(claimed.current_run_id),),
    ).fetchone()
    assert row["task_id"] == foreign_id, "foreign-state fixture must land"


@pytest.mark.parametrize("foreign", [False, True])
def test_gate_requires_run_task_binding(drift_task, foreign):
    """``task_runs.task_id`` must equal ``claimed.id``. A foreign rewrite
    (operator-side repair, manual SQL, DB restore) refuses silently —
    no DB write, no spawn."""
    conn, run_id, _repo = drift_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None and claimed.current_run_id == run_id
    if foreign:
        _make_foreign_run(conn, claimed)
    before = _state_snapshot(conn, "gate-review")
    refusal = dispatch._check_pre_spawn_authority(conn, claimed)
    if foreign:
        assert refusal is not None and refusal[0] == "run_owned_by_other_task", (
            f"gate accepted a run owned by another task (got {refusal!r})")
    else:
        assert refusal is None, (
            f"positive control must pass the gate (got {refusal!r})")
    assert _state_snapshot(conn, "gate-review") == before, (
        "gate refusal must not rewrite any successor or historical row")


@pytest.mark.parametrize("foreign", [False, True])
def test_failure_handler_preserves_foreign_task_run(drift_task, foreign):
    """``_record_task_failure`` must refuse when ``task_runs.task_id``
    names a foreign task; the helper's bounded owner recheck is the
    same check as the gate, applied under its own write txn so a
    successor / foreign run is never rewritten."""
    conn, run_id, _repo = drift_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    if foreign:
        _make_foreign_run(conn, claimed)
    before = _state_snapshot(conn, "gate-review")
    dispatch._record_task_failure(
        conn, claimed.id, "r3 binding fixture failure", outcome="spawn_failed",
        release_claim=True, end_run=True, failure_limit=3,
        expected_run_id=int(claimed.current_run_id),
        expected_claim_lock=claimed.claim_lock,
    )
    after_runs = _state_snapshot(conn, "gate-review")
    if foreign:
        assert after_runs == before, (
            "failure handler mutated a run owned by another task")
    else:
        # Positive control: the helper CLOSED the owned attempt through
        # the native failure lifecycle. The DB must have changed AND
        # the run row we still own must carry ``spawn_failed``.
        assert after_runs != before, (
            "positive control: helper must close the owned attempt")
        closed = conn.execute(
            "SELECT outcome, ended_at FROM task_runs WHERE id=?",
            (int(claimed.current_run_id),),
        ).fetchone()
        assert closed["outcome"] == "spawn_failed"
        assert closed["ended_at"] is not None


@pytest.mark.parametrize("missing", ["run_id", "claim_lock"])
def test_partial_owner_contract_refuses_without_writes(drift_task, missing):
    """Partial expected-owner pair (``expected_run_id`` only or
    ``expected_claim_lock`` only) is treated as refusal without
    mutation and without ``TypeError``. The bounded contract is
    all-or-nothing; a half-bound helper would coerce ``None`` through
    ``int(...)`` and either silently mutate (claim_lock check skipped)
    or raise — both regressions."""
    conn, run_id, _repo = drift_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    before = _state_snapshot(conn, "gate-review")
    kwargs = {
        "expected_run_id": int(claimed.current_run_id),
        "expected_claim_lock": claimed.claim_lock,
    }
    kwargs.pop("expected_" + missing)
    # Must not raise; must not mutate.
    dispatch._record_task_failure(
        conn, claimed.id, "r3 partial pair fixture", outcome="spawn_failed",
        release_claim=True, end_run=True, failure_limit=3, **kwargs,
    )
    assert _state_snapshot(conn, "gate-review") == before, (
        f"partial owner contract (missing expected_{missing}) "
        "was treated as authorization")


# ---------------------------------------------------------------------------
# Empty-field refusal: explicitly empty values (not just None) on the bounded
# owner pair must also refuse without writes. A half-bound contract whose
# supplied value is empty (``""`` claim_lock, ``0`` / negative / non-int
# run_id) is still malformed; the partial-pair guard refuses BEFORE opening
# any write_txn so the helper never mutates state, never emits an event.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_value", ["", "   "])
def test_empty_claim_lock_field_refuses_without_writes(drift_task, empty_value):
    """``expected_claim_lock`` is supplied but empty / whitespace-only — the
    bounded contract is non-empty on both supplied fields; refuse before any
    write. Mirrors the partial-pair guard, exercised on an explicit
    empty-string / whitespace fixture."""
    conn, run_id, _repo = drift_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    before = _state_snapshot(conn, "gate-review")
    dispatch._record_task_failure(
        conn, claimed.id, "r3 empty claim_lock fixture", outcome="spawn_failed",
        release_claim=True, end_run=True, failure_limit=3,
        expected_run_id=int(claimed.current_run_id),
        expected_claim_lock=empty_value,
    )
    assert _state_snapshot(conn, "gate-review") == before, (
        "empty expected_claim_lock was treated as authorization")


@pytest.mark.parametrize("empty_run_id", [0, -1])
def test_empty_run_id_field_refuses_without_writes(drift_task, empty_run_id):
    """``expected_run_id`` is supplied but empty / non-positive — refuse
    before any write. A zero or negative run_id can never be a valid run
    row; allowing it through would coerce ``int(...)`` to a comparison
    that the helper cannot satisfy."""
    conn, run_id, _repo = drift_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    before = _state_snapshot(conn, "gate-review")
    dispatch._record_task_failure(
        conn, claimed.id, "r3 empty run_id fixture", outcome="spawn_failed",
        release_claim=True, end_run=True, failure_limit=3,
        expected_run_id=empty_run_id,
        expected_claim_lock=claimed.claim_lock,
    )
    assert _state_snapshot(conn, "gate-review") == before, (
        f"empty expected_run_id={empty_run_id} was treated as authorization")


def test_unbound_failure_helper_keeps_legacy_behavior(drift_task):
    """Legacy-control: NEITHER ``expected_run_id`` nor ``expected_claim_lock``
    is supplied; the helper must take the same legacy mutation path it took
    before the bounded contract was introduced (close the owned attempt
    through the native failure lifecycle). Defaults preserved — every
    unrelated caller must be unaffected by the partial-pair guard."""
    conn, run_id, _repo = drift_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    before = _state_snapshot(conn, "gate-review")
    # No expected_* kwargs supplied — legacy path runs. The owned attempt
    # must be closed through native failure (spawn_failed outcome on the
    # run, claim released on tasks, event appended).
    dispatch._record_task_failure(
        conn, claimed.id, "r3 legacy unbound fixture", outcome="spawn_failed",
        release_claim=True, end_run=True, failure_limit=3,
    )
    after = _state_snapshot(conn, "gate-review")
    assert after != before, (
        "legacy unbound path must still close the owned attempt")
    closed = conn.execute(
        "SELECT outcome, ended_at FROM task_runs WHERE id=?",
        (int(claimed.current_run_id),),
    ).fetchone()
    assert closed["outcome"] == "spawn_failed"
    assert closed["ended_at"] is not None


# ===========================================================================
# R4-late-return repair: late-PID / no-PID late-return fences, the bounded
# owner contract on ``_set_worker_pid``, and the no-transaction-around-spawn
# invariant. Self-contained product tests, no Main evidence path imports.
#
# Defects repaired in this revision:
#   * dispatcher tests now drive a real READY row (not an already-claimed
#     ``drift_task``) so the dispatcher's native claim cycle runs exactly
#     once at the top of ``_dispatch_lane_task``;
#   * the future-clock fixture is bound BEFORE the time-monkeypatch so the
#     lambda closes over a fixed value (no recursive ``time.time()``);
#   * ``_process_fingerprint`` is spied on every test that exercises the
#     post-spawn path (full-match, unbound, late-return branches);
#   * every late-return test asserts the spawn callback executed exactly
#     once with the ORIGINAL run id, then asserts the dispatcher refuses
#     silently (no DB write, no hook, no ``result.spawned``);
#   * state-drift defects (pointer-lost, foreign run, terminal run,
#     expired lease) are injected INSIDE the slow callback — the gate
#     admits the task, the post-spawn fence catches the drift;
#   * cleanup is the existing ``_terminate_reclaimed_worker`` helper fed
#     the ``UNVERIFIED_WORKER_FINGERPRINT`` sentinel (R4-cleanup) — no
#     fabricated ``terminated=True`` from a spy.
# ===========================================================================


@pytest.fixture
def ready_task(tmp_path, monkeypatch):
    """Bootstrap a fresh READY worktree row whose native claim + workspace
    stamp is performed by ``_dispatch_lane_task`` itself (so the dispatcher
    is exercised end-to-end). Yields ``(conn, repo)``; tests may issue a
    native ``kb.claim_task`` + ``workspace.set_workspace_path`` themselves
    when they need to drive the post-spawn path with a pre-known run id.
    No model calls, no live DB writes outside the fresh SQLite file.
    """
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda _: True)
    monkeypatch.setattr(dispatch, "_process_fingerprint",
                        lambda value: "offline-fixture-fingerprint")
    conn.execute(
        "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
        "VALUES ('gate-review','fixture','ready','worktree',?,1,'fixture-worker')",
        (str(repo),),
    )
    conn.commit()
    yield conn, repo
    conn.close()


@pytest.fixture
def fresh_claimed_task(tmp_path, monkeypatch):
    """Bootstrap a fully claimed + workspace-stamped run row, ready for
    direct ``_set_worker_pid`` unit tests (no dispatcher claim cycle).
    Yields ``(conn, run_id, repo)``.
    """
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda _: True)
    monkeypatch.setattr(dispatch, "_process_fingerprint",
                        lambda value: "offline-fixture-fingerprint")
    conn.execute(
        "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
        "VALUES ('gate-review','fixture','ready','worktree',?,1,'fixture-worker')",
        (str(repo),),
    )
    conn.commit()
    claimed = kb.claim_task(conn, "gate-review", ttl_seconds=300)
    assert claimed is not None, "fixture bootstrap must claim the task"
    workspace.set_workspace_path(conn, "gate-review", str(repo))
    conn.commit()
    yield conn, int(claimed.current_run_id), repo
    conn.close()


def _drive_dispatch_with_slow_spawn(
    conn, task_id: str, *, monkeypatch, db_path: str,
    replace: bool, pid: Optional[int],
    cleanup_spy: Optional[Callable] = None,
    defect_in_cb: Optional[Callable] = None,
    hook_spy: Optional[Callable] = None,
) -> tuple:
    """Drive ``_dispatch_lane_task`` with a slow ``spawn_fn`` that optionally
    performs a native ``release_stale_claims`` + sibling ``claim_task`` round
    trip AND / OR injects a state-drift defect (foreign run, terminal run,
    expired lease, lost pointer) DURING the callback. The native claim at
    the top of ``_dispatch_lane_task`` runs first, so the dispatcher admits
    the gate; the post-spawn fence then catches the drift.

    Returns ``(result, originals, hooks, cleanup_calls, before, accepted)``.

    The future-clock value is captured BEFORE the time-monkeypatch so the
    lambda closes over a fixed value (no recursive ``time.time()``).
    """
    originals: list = []
    hooks: list = []
    cleanup_calls: list = []
    future = int(time.time()) + 400  # captured BEFORE any monkeypatch
    if hook_spy is None:
        hook_spy = lambda *a, **kw: hooks.append((a, kw))
    monkeypatch.setattr(dispatch, "_process_fingerprint",
                        lambda value: "offline-fixture-fingerprint")
    monkeypatch.setattr(kb, "_fire_worker_spawned_hook", hook_spy)
    if cleanup_spy is not None:
        monkeypatch.setattr(dispatch, "_terminate_reclaimed_worker", cleanup_spy)

    def slow_spawn(task, workspace, *args, **kwargs):
        originals.append(task.current_run_id)
        if replace:
            other = sqlite3.connect(db_path)
            other.row_factory = sqlite3.Row
            try:
                with monkeypatch.context() as clock:
                    clock.setattr(kb.time, "time", lambda: future)
                    assert kb.release_stale_claims(other, failure_limit=10) == 1
                    successor = kb.claim_task(
                        other, task.id, ttl_seconds=600,
                        claimer="fixture-successor-after-ttl",
                    )
                    assert successor is not None
                    assert successor.current_run_id != task.current_run_id
                    assert successor.claim_lock != task.claim_lock
            finally:
                other.close()
        if defect_in_cb is not None:
            defect_in_cb(conn, task.id, task.current_run_id)
        before = _state_snapshot(conn, task.id)
        slow_spawn._before = before
        return pid

    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda _: True)
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    result = dispatch.DispatchResult()
    accepted = dispatch._dispatch_lane_task(
        conn, row, "fixture-worker", result,
        lane="ready", dry_run=False,
        ttl_seconds=300, board=None, failure_limit=3,
        spawn_fn=slow_spawn, per_profile_cap=None, per_profile_running={},
    )
    return result, originals, hooks, cleanup_calls, slow_spawn._before, accepted


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("pid", [0, 54321])
def test_late_spawn_return_cannot_stamp_or_report_successor(
        ready_task, monkeypatch, replace, pid):
    """Migrated from the parent late-return repro: a slow ``spawn_fn``
    callback that returns AFTER ``release_stale_claims`` + a sibling
    ``claim_task`` must NOT stamp the returned PID, must NOT append
    ``result.spawned``, and must NOT fire the spawned hook. ``pid=0`` is
    the no-PID branch (no ``_terminate_reclaimed_worker`` call);
    ``pid=54321`` is the late-PID branch (the cleanup spy is wired so we
    can also assert it ran with the ORIGINAL claim identity, fed the
    ``UNVERIFIED_WORKER_FINGERPRINT`` sentinel so a fresh fingerprint
    read never authorises a kill).
    """
    conn, _repo = ready_task
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    cleanup_calls: list = []

    def cleanup_spy(pid_arg, lock, *args, **kwargs):
        cleanup_calls.append({"pid": pid_arg, "lock": lock,
                              "started_at": kwargs.get("started_at")})
        # Simulate the real helper's UNVERIFIED rule: a live unknown PID
        # is held, never signalled. The dispatcher must NOT claim
        # ``terminated`` was successful from this spy — see the
        # verified-spy control below for the legitimate-terminated case.
        return {
            "host_local": True,
            "termination_attempted": False,
            "terminated": not bool(pid_arg),
            "signal_refused": True,
        }

    spy = cleanup_spy if pid else None
    result, originals, hooks, cleanup, before, accepted = _drive_dispatch_with_slow_spawn(
        conn, "gate-review", monkeypatch=monkeypatch, db_path=db_path,
        replace=replace, pid=pid, cleanup_spy=spy,
    )
    assert len(originals) == 1, "spawn_fn must run exactly once"
    original_run = originals[0]
    if replace:
        # Late-return: ownership moved to a successor during the callback.
        # The dispatcher must NOT report this as a successful spawn.
        assert before is not None
        assert _state_snapshot(conn, "gate-review") == before, (
            "late spawn result mutated successor task/run/events")
        assert not accepted, "stale spawn result was reported as accepted"
        assert result.spawned == [], (
            "stale spawn result was appended to result.spawned")
        assert hooks == [], "spawn hook was emitted for a superseded owner"
        if pid:
            assert cleanup_calls, (
                "late PID return must invoke the existing termination "
                "helper with the UNVERIFIED sentinel")
            call = cleanup_calls[0]
            assert call["pid"] == pid
            assert call["started_at"] == "unverified", (
                "late-PID cleanup must NOT use a freshly-read fingerprint")
            assert isinstance(call["lock"], str) and call["lock"], (
                "cleanup must receive the ORIGINAL claim_lock, not the "
                "live successor's lock")
        else:
            assert cleanup_calls == [], (
                "no-PID late return must not invoke cleanup")
    else:
        # Positive control: the callback returned while we still owned
        # the task, so the dispatcher publishes the spawn.
        assert accepted, "fresh callback must be reported as accepted"
        assert len(result.spawned) == 1
        assert result.spawned[0][0] == "gate-review"
        assert len(hooks) == 1, "fresh spawn must fire the hook exactly once"
        if pid:
            task_row = conn.execute(
                "SELECT worker_pid, worker_started_at "
                "FROM tasks WHERE id='gate-review'",
            ).fetchone()
            assert int(task_row["worker_pid"]) == pid
            assert task_row["worker_started_at"] == "offline-fixture-fingerprint"
            run_row = conn.execute(
                "SELECT worker_pid, worker_started_at FROM task_runs "
                "WHERE id=?", (original_run,),
            ).fetchone()
            assert int(run_row["worker_pid"]) == pid
            assert cleanup_calls == [], (
                "clean positive control must NOT trigger late-return cleanup")


def test_no_pid_late_return_preserves_successor(ready_task, monkeypatch):
    """No-PID branch of the late-return fence, isolated so the no-PID
    semantics are explicit. ``spawn_fn`` returns ``0`` AFTER
    ``release_stale_claims`` + sibling ``claim_task``; the dispatcher must
    NOT append ``result.spawned``, must NOT fire the hook, and must NOT
    rewrite the successor.
    """
    conn, _repo = ready_task
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    result, originals, hooks, cleanup, before, accepted = _drive_dispatch_with_slow_spawn(
        conn, "gate-review", monkeypatch=monkeypatch, db_path=db_path,
        replace=True, pid=0,
    )
    assert len(originals) == 1, "spawn_fn must run exactly once"
    assert not accepted
    assert result.spawned == []
    assert hooks == []
    assert cleanup == []
    assert _state_snapshot(conn, "gate-review") == before, (
        "no-PID late return must leave successor task/run/events "
        "byte-for-byte")


def _inject_foreign_pointer(conn, task_id, run_id):
    """Inject a foreign run row + re-point the live ``current_run_id`` to
    it DURING the spawn callback. The gate has already admitted the
    owned run; the post-spawn fence is what catches the drift.
    """
    foreign_task = "foreign-pointer-task"
    conn.execute(
        "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at) "
        "VALUES (?, 'foreign', 'ready', 'worktree', '/tmp/foreign', 1)",
        (foreign_task,),
    )
    conn.execute(
        "INSERT INTO task_runs(task_id,claim_lock,status,outcome,claim_expires,started_at) "
        "VALUES (?, 'foreign-run-lock', 'running', NULL, ?, ?)",
        (foreign_task, int(time.time()) + 600, int(time.time())),
    )
    foreign_run_id = conn.execute(
        "SELECT id FROM task_runs WHERE claim_lock='foreign-run-lock'",
    ).fetchone()[0]
    conn.execute(
        "UPDATE tasks SET current_run_id=? WHERE id=?",
        (int(foreign_run_id), task_id),
    )
    conn.commit()


def test_changed_pointer_late_return_refuses_silently(ready_task, monkeypatch):
    """State-drift fence on the post-spawn path: between ``spawn_fn`` entry
    and return, ``tasks.current_run_id`` is rewritten to a FOREIGN run id.
    The dispatcher must refuse without writing PID, without firing hook,
    without appending ``result.spawned``. The defect is injected DURING
    the callback so the gate admits the task; the post-spawn fence is
    what catches it.
    """
    conn, _repo = ready_task
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    hooks: list = []
    result, originals, hooks, cleanup, before, accepted = _drive_dispatch_with_slow_spawn(
        conn, "gate-review", monkeypatch=monkeypatch, db_path=db_path,
        replace=False, pid=54321,
        defect_in_cb=_inject_foreign_pointer,
        hook_spy=lambda *a, **kw: hooks.append((a, kw)),
    )
    assert len(originals) == 1, "spawn_fn must run exactly once"
    assert not accepted
    assert result.spawned == []
    assert hooks == []
    assert cleanup == [], "no cleanup when only the pointer was swapped"
    assert _state_snapshot(conn, "gate-review") == before, (
        "changed-pointer late return must not rewrite any row")


def _inject_foreign_run_task_id(conn, task_id, run_id):
    """Rewrite ``task_runs.task_id`` on the live run row DURING the
    callback to name a different (foreign) task. Same fence as the
    R3-binding repair, exercised on the spawn-result path.
    """
    foreign_task = "foreign-run-task-id-task"
    conn.execute(
        "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at) "
        "VALUES (?, 'foreign', 'ready', 'worktree', '/tmp/foreign', 1)",
        (foreign_task,),
    )
    conn.execute(
        "UPDATE task_runs SET task_id=? WHERE id=?",
        (foreign_task, int(run_id)),
    )
    conn.commit()


def test_foreign_run_task_id_late_return_refuses_silently(ready_task, monkeypatch):
    """Live ``task_runs.task_id`` names a foreign task DURING the
    callback. ``_set_worker_pid`` refuses; the dispatcher must NOT publish
    the spawn.
    """
    conn, _repo = ready_task
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    hooks: list = []
    result, originals, hooks, cleanup, before, accepted = _drive_dispatch_with_slow_spawn(
        conn, "gate-review", monkeypatch=monkeypatch, db_path=db_path,
        replace=False, pid=54321,
        defect_in_cb=_inject_foreign_run_task_id,
        hook_spy=lambda *a, **kw: hooks.append((a, kw)),
    )
    assert len(originals) == 1, "spawn_fn must run exactly once"
    assert not accepted
    assert result.spawned == []
    assert hooks == []
    assert _state_snapshot(conn, "gate-review") == before


def _inject_terminal_run(conn, task_id, run_id):
    """Close the live run DURING the callback (``status='spawn_failed'``,
    ``outcome='spawn_failed'``, ``ended_at`` set). ``_set_worker_pid``
    refuses because the run is no longer ``running``.
    """
    conn.execute(
        "UPDATE task_runs SET status='spawn_failed', outcome='spawn_failed', "
        "ended_at=? WHERE id=?",
        (int(time.time()), int(run_id)),
    )
    conn.commit()


def test_terminal_run_late_return_refuses_silently(ready_task, monkeypatch):
    """Late-return race where the original run is closed DURING the
    callback (``status='spawn_failed'``, ``ended_at`` set). The bounded
    owner recheck sees the run's status is no longer ``running``; the
    dispatcher must refuse silently.
    """
    conn, _repo = ready_task
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    hooks: list = []
    result, originals, hooks, cleanup, before, accepted = _drive_dispatch_with_slow_spawn(
        conn, "gate-review", monkeypatch=monkeypatch, db_path=db_path,
        replace=False, pid=54321,
        defect_in_cb=_inject_terminal_run,
        hook_spy=lambda *a, **kw: hooks.append((a, kw)),
    )
    assert len(originals) == 1, "spawn_fn must run exactly once"
    assert not accepted
    assert result.spawned == []
    assert hooks == []
    assert _state_snapshot(conn, "gate-review") == before


def _inject_expired_lease(conn, task_id, run_id):
    """Force both the tasks row and the task_runs row to an expired
    lease DURING the callback. The bounded owner recheck sees
    ``claim_expires <= now`` and refuses silently.
    """
    conn.execute(
        "UPDATE tasks SET claim_expires=1 WHERE id=?",
        (task_id,),
    )
    conn.execute(
        "UPDATE task_runs SET claim_expires=1 WHERE id=?",
        (int(run_id),),
    )
    conn.commit()


def test_expired_lease_late_return_refuses_silently(ready_task, monkeypatch):
    """Lease elapsed DURING the callback. The bounded owner recheck sees
    ``claim_expires <= now`` and refuses silently.
    """
    conn, _repo = ready_task
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    hooks: list = []
    result, originals, hooks, cleanup, before, accepted = _drive_dispatch_with_slow_spawn(
        conn, "gate-review", monkeypatch=monkeypatch, db_path=db_path,
        replace=False, pid=54321,
        defect_in_cb=_inject_expired_lease,
        hook_spy=lambda *a, **kw: hooks.append((a, kw)),
    )
    assert len(originals) == 1, "spawn_fn must run exactly once"
    assert not accepted
    assert result.spawned == []
    assert hooks == []
    assert _state_snapshot(conn, "gate-review") == before


def test_unverified_cleanup_uses_unverified_sentinel(ready_task, monkeypatch):
    """Cleanup-spy control: when ``_set_worker_pid`` refuses on a late-PID
    return, ``_terminate_reclaimed_worker`` MUST be invoked with the
    ORIGINAL claim_lock captured before the sibling reclaim (not the live
    ``tasks.claim_lock``, which by then belongs to the successor) AND
    with ``started_at='unverified'`` — the R4-cleanup rule that prevents
    a freshly-read fingerprint from authorising a kill on the live PID
    occupant.
    """
    conn, _repo = ready_task
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    cleanup_calls: list = []

    def cleanup_spy(pid_arg, lock, *args, **kwargs):
        cleanup_calls.append({"pid": pid_arg, "lock": lock,
                              "started_at": kwargs.get("started_at")})
        return {
            "host_local": True,
            "termination_attempted": False,
            "terminated": False,
            "signal_refused": True,
        }

    result, originals, hooks, cleanup, before, accepted = _drive_dispatch_with_slow_spawn(
        conn, "gate-review", monkeypatch=monkeypatch, db_path=db_path,
        replace=True, pid=54321, cleanup_spy=cleanup_spy,
    )
    assert len(originals) == 1, "spawn_fn must run exactly once"
    assert cleanup_calls, "cleanup must be called for late PID return"
    call = cleanup_calls[0]
    assert call["pid"] == 54321, "cleanup must target the late-return PID"
    assert call["started_at"] == "unverified", (
        "cleanup must be invoked with UNVERIFIED_WORKER_FINGERPRINT so a "
        "fresh fingerprint read never authorises a kill")
    assert isinstance(call["lock"], str) and call["lock"], (
        "cleanup must receive a non-empty ORIGINAL claim_lock")
    assert not accepted
    assert result.spawned == []
    assert hooks == []
    assert _state_snapshot(conn, "gate-review") == before


def test_native_termination_helper_unchanged_on_genuine_prior_fingerprint(
        monkeypatch):
    """Control: the existing ``_terminate_reclaimed_worker`` keeps its
    legitimate behaviour for a row that ALREADY has a genuine PRIOR
    persisted fingerprint (``tasks.worker_started_at`` written by a
    previous spawn). The genuine-spawned control signals the worker; the
    recycled-PID control refuses. No fresh fingerprint is ever laundered
    into kill authority from a spy merely returning ``terminated=True``.
    """
    signals = []

    def make_spy(current):
        return lambda pid, lock, *args, **kwargs: signals.append(
            (pid, current)) or {
                    "host_local": True,
                    "termination_attempted": True,
                    "terminated": True,
                }

    # Genuine PRIOR persisted fingerprint — the control case.
    monkeypatch.setattr(dispatch, "_process_fingerprint",
                        lambda value: "fixture-boot|100")
    monkeypatch.setattr(kb, "_pid_alive", lambda _: True)
    monkeypatch.setattr(dispatch, "_poll_worker_exit",
                        lambda *a, **kw: True)
    info = dispatch._terminate_reclaimed_worker(
        54321, kb._host_prefix() + "fixture-lock", started_at="fixture-boot|100",
        signal_fn=make_spy("fixture-boot|100"),
    )
    assert info["terminated"] is True
    assert signals == [(54321, "fixture-boot|100")], (
        "genuine prior fingerprint must authorise the signal")
    signals.clear()
    # Recycled PID — never signalled.
    monkeypatch.setattr(dispatch, "_process_fingerprint",
                        lambda value: "fixture-boot|200")
    info = dispatch._terminate_reclaimed_worker(
        54321, kb._host_prefix() + "fixture-lock", started_at="fixture-boot|100",
        signal_fn=make_spy("fixture-boot|200"),
    )
    assert info["terminated"] is True
    assert info.get("pid_recycled") is True
    assert signals == [], "recycled PID must never be signalled"


def test_unbound_set_worker_pid_keeps_legacy_behavior(fresh_claimed_task):
    """Primitive control: when NO ``expected_*`` kwarg is supplied, the
    helper behaves exactly as before — every unrelated caller (the unit
    tests that call ``kbd._set_worker_pid(conn, tid, pid)`` directly,
    ``reap_terminal_workers`` cleanup, etc.) is unaffected. The
    fingerprint spy returns ``offline-fixture-fingerprint``; the legacy
    path must stamp it on both the tasks row and the task_runs row.
    """
    conn, run_id, _repo = fresh_claimed_task
    written = dispatch._set_worker_pid(conn, "gate-review", 98765)
    assert written is True, "unbound legacy call must return True"
    task = conn.execute(
        "SELECT worker_pid, worker_started_at FROM tasks WHERE id='gate-review'",
    ).fetchone()
    assert int(task["worker_pid"]) == 98765
    assert task["worker_started_at"] == "offline-fixture-fingerprint"
    run = conn.execute(
        "SELECT worker_pid, worker_started_at FROM task_runs "
        "WHERE id=?", (int(run_id),),
    ).fetchone()
    assert int(run["worker_pid"]) == 98765


@pytest.mark.parametrize("missing", ["task_id", "run_id", "claim_lock"])
def test_set_worker_pid_partial_pair_refuses_without_writes(
        fresh_claimed_task, missing):
    """Partial / malformed pair guard on the primitive: a half-bound call
    (any missing field) refuses WITHOUT writes and WITHOUT ``TypeError``.
    Same all-or-nothing contract as the failure helper's partial-pair
    guard. ``expected_claim_lock`` fixture is the genuine live lock so a
    full-bound call would otherwise stamp.
    """
    conn, run_id, _repo = fresh_claimed_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    before = _state_snapshot(conn, "gate-review")
    kwargs = {
        "expected_task_id": "gate-review",
        "expected_run_id": int(run_id),
        "expected_claim_lock": claimed.claim_lock,
    }
    kwargs.pop("expected_" + missing)
    accepted = dispatch._set_worker_pid(conn, "gate-review", 54321, **kwargs)
    assert accepted is False, (
        f"partial pair (missing expected_{missing}) must refuse")
    assert _state_snapshot(conn, "gate-review") == before, (
        f"partial pair (missing expected_{missing}) was treated as "
        "authorization")


@pytest.mark.parametrize("empty_run_id", [0, -1])
def test_set_worker_pid_empty_run_id_refuses_without_writes(
        fresh_claimed_task, empty_run_id):
    """Empty / non-positive ``expected_run_id`` refuses BEFORE any write
    even when the other two fields are fully bound (so the all-or-nothing
    check is satisfied). A zero or negative run_id can never be a valid
    run row.
    """
    conn, run_id, _repo = fresh_claimed_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    before = _state_snapshot(conn, "gate-review")
    accepted = dispatch._set_worker_pid(
        conn, "gate-review", 54321,
        expected_task_id="gate-review",
        expected_run_id=empty_run_id,
        expected_claim_lock=claimed.claim_lock,
    )
    assert accepted is False
    assert _state_snapshot(conn, "gate-review") == before


@pytest.mark.parametrize("empty_lock", ["", "   "])
def test_set_worker_pid_empty_claim_lock_refuses_without_writes(
        fresh_claimed_task, empty_lock):
    """Empty / whitespace ``expected_claim_lock`` refuses BEFORE any write
    even when the other two fields are fully bound.
    """
    conn, run_id, _repo = fresh_claimed_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    before = _state_snapshot(conn, "gate-review")
    accepted = dispatch._set_worker_pid(
        conn, "gate-review", 54321,
        expected_task_id="gate-review",
        expected_run_id=int(run_id),
        expected_claim_lock=empty_lock,
    )
    assert accepted is False
    assert _state_snapshot(conn, "gate-review") == before


def test_set_worker_pid_mismatched_task_id_refuses_without_writes(
        fresh_claimed_task):
    """R4-target repair on the primitive: ``expected_task_id`` and the
    positional ``task_id`` MUST match when the bounded contract is supplied.
    A caller that validates the gate-review row but writes to the
    other-task row is rejected — no DB mutation, no spawn, no event.
    """
    conn, run_id, _repo = fresh_claimed_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    before = _state_snapshot(conn, "gate-review")
    accepted = dispatch._set_worker_pid(
        conn, "other-task", 54321,
        expected_task_id="gate-review",
        expected_run_id=int(run_id),
        expected_claim_lock=claimed.claim_lock,
    )
    assert accepted is False, "mismatched task_id must refuse"
    assert _state_snapshot(conn, "gate-review") == before, (
        "mismatched task_id was treated as authorization")
    # No other-task row was created and the gate-review row is unchanged.
    other = conn.execute(
        "SELECT worker_pid FROM tasks WHERE id='other-task'",
    ).fetchone()
    assert other is None, "mismatched task_id must not write a foreign row"


def test_set_worker_pid_full_match_stamps(fresh_claimed_task):
    """Positive control on the primitive: a fully bound call whose live
    state still matches the expected identity stamps PID + fingerprint +
    spawned event AND returns True. The fingerprint spy returns
    ``offline-fixture-fingerprint`` so the row stamps the same value.
    """
    conn, run_id, _repo = fresh_claimed_task
    claimed = kb.get_task(conn, "gate-review")
    assert claimed is not None
    accepted = dispatch._set_worker_pid(
        conn, "gate-review", 55555,
        expected_task_id=claimed.id,
        expected_run_id=int(claimed.current_run_id),
        expected_claim_lock=claimed.claim_lock,
    )
    assert accepted is True
    task = conn.execute(
        "SELECT worker_pid, worker_started_at FROM tasks WHERE id='gate-review'",
    ).fetchone()
    assert int(task["worker_pid"]) == 55555
    assert task["worker_started_at"] == "offline-fixture-fingerprint"
    run = conn.execute(
        "SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?",
        (int(run_id),),
    ).fetchone()
    assert int(run["worker_pid"]) == 55555


def test_set_worker_pid_full_mismatch_refuses_silently(fresh_claimed_task):
    """Negative control on the primitive: the expected identity does not
    match the live state (a successor reclaimed) — refuse without writing.
    """
    conn, run_id, _repo = fresh_claimed_task
    before = _state_snapshot(conn, "gate-review")
    # Lie about the expected claim_lock so the live state never matches.
    accepted = dispatch._set_worker_pid(
        conn, "gate-review", 55555,
        expected_task_id="gate-review",
        expected_run_id=int(run_id),
        expected_claim_lock="deliberately-wrong-lock",
    )
    assert accepted is False
    assert _state_snapshot(conn, "gate-review") == before
    task = conn.execute(
        "SELECT worker_pid FROM tasks WHERE id='gate-review'",
    ).fetchone()
    assert task["worker_pid"] is None, (
        "no PID must be stamped on a mismatched fence")


def test_spawned_hook_keeps_original_run_id_after_publication(
        tmp_path, monkeypatch):
    """R4-hook repair — the committed historical spawn must retain its
    own run identity. ``_fire_worker_spawned_hook`` is called AFTER the
    PID is durably persisted; a sibling reclaim between the publication
    commit and the hook fire would otherwise re-attribute the spawn to
    the successor. The new ``expected_run_id`` kwarg preserves the
    ORIGINAL run id; legacy callers (no kwarg) still read the live
    ``_current_run_id`` so existing observers are unaffected.
    """
    repo = tmp_path / "repo"
    _make_repo(repo)
    conn = _open_board_db(tmp_path / "board.db")
    try:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workspace_kind,workspace_path,created_at,assignee) "
            "VALUES ('hook-original','fixture','ready','worktree',?,1,'fixture-worker')",
            (str(repo),),
        )
        conn.commit()
        notifications: list = []
        monkeypatch.setattr(kb, "_kanban_observer_consumed", lambda _: True)
        monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook",
                            lambda event, task_id, **kw: notifications.append(
                                (event, kw)))
        # First spawn: claim + write + fire hook with the ORIGINAL run id.
        claimed = kb.claim_task(conn, "hook-original", ttl_seconds=300)
        assert claimed is not None
        workspace.set_workspace_path(conn, "hook-original", str(repo))
        conn.commit()
        original_run_id = int(claimed.current_run_id)
        written = dispatch._set_worker_pid(
            conn, "hook-original", 77777,
            expected_task_id=claimed.id,
            expected_run_id=original_run_id,
            expected_claim_lock=claimed.claim_lock,
        )
        assert written is True
        # Fire the hook the way the dispatcher does — supplying the
        # ORIGINAL run id, NOT a fresh ``_current_run_id`` re-read.
        kb._fire_worker_spawned_hook(
            conn, claimed, str(repo), 77777,
            expected_run_id=original_run_id,
        )
        # Native failure + successor reclaim on a sibling connection.
        other = sqlite3.connect(conn.execute("PRAGMA database_list").fetchone()[2])
        other.row_factory = sqlite3.Row
        try:
            dispatch._record_task_failure(
                other, "hook-original", "fixture native close after publication",
                outcome="spawn_failed", release_claim=True, end_run=True,
                failure_limit=10,
            )
            successor = kb.claim_task(other, "hook-original", ttl_seconds=600,
                                        claimer="fixture-successor")
            assert successor is not None
            successor_run_id = int(successor.current_run_id)
            assert successor_run_id != original_run_id
        finally:
            other.close()
        spawned = [kw for event, kw in notifications
                    if event == "on_kanban_worker_spawned"]
        assert len(spawned) == 1, (
            "committed historical spawn should retain its own run identity")
        assert spawned[0]["run_id"] == original_run_id, (
            "hook re-attributed an old spawn to the successor")
    finally:
        conn.close()


def test_no_transaction_around_spawn_fn(ready_task, monkeypatch):
    """Invariant: no ``write_txn`` is held across the ``spawn_fn`` call.
    The dispatcher must close any prior txn (the ``set_workspace_path`` /
    ``set_branch_name`` writes) BEFORE invoking ``spawn_fn`` so the slow
    callback does not stall a writer.
    """
    conn, _repo = ready_task
    txn_open = {"held": False}
    real_write_txn = kb.write_txn

    @contextlib.contextmanager
    def watch_txn(c, *args, **kwargs):
        with real_write_txn(c, *args, **kwargs) as ctx:
            txn_open["held"] = True
            try:
                yield ctx
            finally:
                txn_open["held"] = False

    monkeypatch.setattr(kb, "write_txn", watch_txn)
    seen: list = []

    def spawn_fn(task, workspace, *args, **kwargs):
        seen.append(bool(txn_open["held"]))
        return 54321

    monkeypatch.setattr(dispatch, "_process_fingerprint",
                        lambda value: "offline-fixture-fingerprint")
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda _: True)
    row = conn.execute("SELECT * FROM tasks WHERE id='gate-review'").fetchone()
    result = dispatch.DispatchResult()
    accepted = dispatch._dispatch_lane_task(
        conn, row, "fixture-worker", result,
        lane="ready", dry_run=False,
        ttl_seconds=300, board=None, failure_limit=3,
        spawn_fn=spawn_fn, per_profile_cap=None, per_profile_running={},
    )
    assert accepted, "dispatcher must accept the fresh callback"
    assert len(result.spawned) == 1, (
        "launch reached: dispatcher must record exactly one spawned task")
    assert seen == [False], "no write_txn must be open during spawn_fn"
    assert not conn.in_transaction, (
        "callback must observe conn.in_transaction == False")
    assert len(seen) == 1, "spawn_fn must run exactly once"
