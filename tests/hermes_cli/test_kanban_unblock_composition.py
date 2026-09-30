"""Core-owned unblock composition; preserve public nesting refusal and rollback."""
import json
import sqlite3

import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kdc
from hermes_cli import kanban_db_transitions as graph


@pytest.fixture
def board(tmp_path):
    db = sqlite3.connect(tmp_path / "board.db")
    db.row_factory = sqlite3.Row
    db.executescript(kb.SCHEMA_SQL)
    for _ in range(2):
        kdc._migrate_add_optional_columns(db)
        db.commit()
    try:
        yield db
    finally:
        db.close()


@pytest.mark.parametrize("mode", ["root", "parent_pending", "resume_review", "scheduled", "already_ready", "dangling_run"])
def test_transactional_unblock_preserves_native_status_and_event_contract(board, mode):
    parent = kb.create_task(board, title="parent", initial_status="blocked")
    tid = kb.create_task(board, title="child", initial_status="blocked",
                         parents=[parent] if mode == "parent_pending" else [])
    if mode == "resume_review":
        kb._append_event(board, tid, "blocked", {"resume_status": "review"})
    elif mode == "scheduled":
        board.execute("UPDATE tasks SET status='scheduled' WHERE id=?", (tid,))
    elif mode in {"already_ready", "dangling_run"}:
        assert kb.unblock_task(board, tid)
        if mode == "dangling_run":
            assert kb.claim_task(board, tid, claimer="fixture-owner") is not None
            run_id = kb.get_task(board, tid).current_run_id
            # Simulate preexisting recovery damage, not a native block witness.
            board.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
    board.commit()
    before = list(board.iterdump())
    native = getattr(graph, "unblock_task_in_transaction", None)
    assert callable(native), "Core needs an explicit composition primitive, not a nesting bypass"
    with kdc.write_txn(board):
        changed = native(board, tid)
    expected = {"root": "ready", "parent_pending": "todo", "resume_review": "review",
                "scheduled": "ready", "already_ready": "ready", "dangling_run": "ready"}[mode]
    assert kb.get_task(board, tid).status == expected
    assert kb.get_task(board, parent).status == "blocked"
    assert changed is (mode != "already_ready")
    if mode == "dangling_run":
        run = board.execute("SELECT * FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["status"] == run["outcome"] == "reclaimed"
        assert run["ended_at"] is not None and run["claim_lock"] is None
        assert kb.get_task(board, tid).current_run_id is None
    if mode == "already_ready":
        assert list(board.iterdump()) == before
    else:
        event = board.execute("SELECT kind,payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        assert event["kind"] == "unblocked"
        if expected != "ready":
            assert json.loads(event["payload"])["status"] == expected
    # Neither surface may invent a successful transition on an already-ready task.
    after = list(board.iterdump())
    assert kb.unblock_task(board, tid) is False
    assert list(board.iterdump()) == after


@pytest.mark.parametrize("probe", ["no_outer_transaction", "public_nested", "rollback_after_two", "delegated_fence"])
def test_unblock_composition_requires_owner_transaction_and_preserves_rollback(board, monkeypatch, probe):
    ids = [kb.create_task(board, title=name, initial_status="blocked") for name in ("A", "B")]
    native = getattr(graph, "unblock_task_in_transaction", None)
    assert callable(native), "Core composition primitive missing"
    before = list(board.iterdump())
    if probe == "no_outer_transaction":
        with pytest.raises(RuntimeError, match="transaction"):
            native(board, ids[0])
    elif probe == "public_nested":
        with pytest.raises(RuntimeError, match="already inside a transaction"):
            with kdc.write_txn(board):
                kb.unblock_task(board, ids[0])
    elif probe == "delegated_fence":
        with kdc.write_txn(board):
            with monkeypatch.context() as patched:
                def denied(path):
                    raise PermissionError("fixture delegated fence")
                patched.setattr(kb, "_assert_not_delegated_child_mutation", denied)
                with pytest.raises(PermissionError, match="fixture delegated fence"):
                    native(board, ids[0])
    else:
        with pytest.raises(RuntimeError, match="fixture rollback"):
            with kdc.write_txn(board):
                for tid in ids:
                    assert native(board, tid)
                raise RuntimeError("fixture rollback")
    assert not board.in_transaction
    assert list(board.iterdump()) == before
    assert kb.unblock_task(board, ids[0])
