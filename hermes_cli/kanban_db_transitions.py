"""Native task transitions composable under an owning Core write transaction.

No managed graph, promotion or outbox APIs. The public single-task wrapper
keeps its own transaction; trusted graph callers retain one lock across all
identity checks and transitions. All writes use the same native state rules.
"""
from __future__ import annotations

import sqlite3
import time


def unblock_task_in_transaction(conn: sqlite3.Connection, task_id: str) -> bool:
    """Compose the native unblock transition under the caller's write transaction.

    The graph owner must enter write_txn before its authority reads and call
    this primitive without committing between members. No hooks, process work
    or external effects occur here; task/run/event writes roll back together.
    The public unblock_task retains its own non-nestable transaction boundary.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import _main_db_file

    if not conn.in_transaction:
        raise RuntimeError("unblock composition requires an owning write transaction")
    kb._assert_not_delegated_child_mutation(_main_db_file(conn))
    now = int(time.time())
    resume_status = (
        kb._resume_status_from_events(conn, task_id)
        if kb._task_status(conn, task_id) == "blocked"
        else "ready"
    )
    kb._reclaim_dangling_run(
        conn, task_id, statuses=("blocked", "scheduled"), now=now,
        note="invariant recovery on unblock",
    )
    # Re-gate on parent completion before restoring the source phase.
    landing_status = kb._landing_status_after_parents(conn, task_id)
    new_status = (
        "review"
        if landing_status == "ready" and resume_status == "review"
        else landing_status
    )
    # ``block_kind``/``block_recurrences`` deliberately survive the unblock:
    # resetting them is the amnesia that let cron-unblock <-> re-block loop
    # unbounded; only complete_task clears them. ``consecutive_failures``
    # (the dispatcher's spawn/crash counter) IS reset — a deliberate unblock
    # is a fresh start for the retry budget.
    cur = conn.execute(
        "UPDATE tasks SET status = ?, current_run_id = NULL, "
        "consecutive_failures = 0, last_failure_error = NULL "
        "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
    )
    if cur.rowcount != 1:
        return False
    kb._append_event(
        conn, task_id, "unblocked",
        (
            {"status": new_status, "resume_status": resume_status}
            if new_status != "ready" or resume_status != "ready"
            else None
        ),
    )
    return True
