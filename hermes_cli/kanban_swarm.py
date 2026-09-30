"""Kanban Swarm v1: thin swarm topology helpers on top of Kanban.

Deliberately no second scheduler — a small task graph written into the
existing Kanban kernel:

    planning root (completed immediately)
        ├─ parallel specialist workers (ready)
        └─ verifier (todo until all workers done)
             └─ synthesizer (todo until verifier done)

The shared blackboard is structured JSON comments on the root task, so all
state lives in existing task_comments/task_events rows and the dashboard,
notifier, slash command and dispatcher keep working without a new service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb

BLACKBOARD_PREFIX = "[swarm:blackboard] "


@dataclass(frozen=True)
class SwarmWorkerSpec:
    """A single parallel worker card in a swarm."""

    profile: str
    title: str
    body: str
    skills: list[str] = field(default_factory=list)
    priority: int = 0
    max_runtime_seconds: Optional[int] = None


@dataclass(frozen=True)
class SwarmCreated:
    """IDs produced by :func:`create_swarm`."""

    root_id: str
    worker_ids: list[str]
    verifier_id: str
    synthesizer_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_text(value: str, field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _swarm_context(root_id: str, goal: str) -> str:
    return (
        f"\n\n## Swarm protocol\n- Swarm root / shared blackboard: `{root_id}`.\n- Read "
        f"sibling/parent handoffs from Kanban context before working.\n- Put machine-readable "
        f"facts in completion metadata.\n- Put cross-worker notes on the root task using "
        f"structured comments.\n- Goal: {goal.strip()}\n"
    )


def _activate_root_inline(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    summary: str,
    metadata: dict[str, Any],
) -> bool:
    """Inline blocked→done CAS flip + event insert for the swarm root.

    Runs INSIDE create_swarm's write_txn, so it must not call
    ``kb.complete_task`` (own transaction + post-commit side effects that
    would run while the outer txn can still roll back). The caller runs
    ``recompute_ready`` after the outer commit.
    """
    cur = conn.execute(
        """
        UPDATE tasks
           SET status       = 'done',
               completed_at = ?,
               claim_lock   = NULL,
               claim_expires= NULL,
               worker_pid   = NULL
         WHERE id = ?
           AND status = 'blocked'
        """,
        (int(time.time()), root_id),
    )
    if cur.rowcount != 1:
        return False
    run_id = kb._synthesize_ended_run(conn, root_id, outcome="completed", summary=summary, metadata=metadata)
    kb._append_event(
        conn, root_id, "completed", {"result_len": 0, "summary": summary[:400] or None}, run_id=run_id,
    )
    return True


def create_swarm(
    conn: sqlite3.Connection,
    *,
    goal: str,
    workers: Iterable[SwarmWorkerSpec],
    verifier_assignee: str,
    synthesizer_assignee: str,
    root_title: Optional[str] = None,
    verifier_title: str = "Verify swarm outputs",
    synthesizer_title: str = "Synthesize swarm outputs",
    tenant: Optional[str] = None,
    created_by: str = "swarm-orchestrator",
    workspace_kind: Optional[str] = None,
    workspace_path: Optional[str] = None,
    priority: int = 0,
    idempotency_key: Optional[str] = None,
    per_task_worktrees: bool = False,
    git_base_commit: Optional[str] = None,
    git_base_tree: Optional[str] = None,
) -> SwarmCreated:
    """Create an atomic swarm, optionally planning distinct worktree targets.

    ``per_task_worktrees`` interprets ``workspace_path`` as the existing project
    root. Plans are persisted before activation, not rebound after dispatch;
    this does not create worktrees or certify their Git authority.

    F02 frozen-workspace-basis: when ``per_task_worktrees=True`` AND
    ``git_base_commit`` / ``git_base_tree`` are provided, ``create_swarm``
    binds an immutable plan row in ``task_workspace_plans`` for every swarm
    member BEFORE activating the planning root. When the frozen base is
    omitted, ``create_swarm`` captures the CURRENT basis once (HEAD /
    HEAD^{tree} at activation time) and binds that captured basis to every
    swarm member. The activation flow is ``bind plan -> activate root``;
    plan binding and activation share one transaction; the dispatcher's
    later ``resolve_workspace`` materializes the validated frozen plan.

    ``git_base_commit`` / ``git_base_tree`` are optional. Explicit objects
    are validated without normalizing their spelling. Otherwise activation
    captures the current basis once and exact replay reuses that stored basis.
    Missing plans on an active topology refuse rather than recertify history.
    """
    if type(per_task_worktrees) is not bool:
        raise ValueError("per_task_worktrees must be a boolean")
    project_root = None
    capture_basis = git_base_commit is None and git_base_tree is None
    if per_task_worktrees:
        if workspace_kind != "worktree" or not workspace_path or not Path(workspace_path).is_absolute():
            raise ValueError("per-task worktree planning requires an absolute workspace root")
        try:
            project_root = Path(workspace_path).resolve(strict=True)
        except OSError as exc:
            raise ValueError("worktree workspace root does not exist") from exc
        if not project_root.is_dir():
            raise ValueError("worktree workspace root must be a directory")
        # F02: validate frozen base inputs (commit + tree both present, or
        # both absent). Capture the current basis once when the caller
        # chose not to pin it.
        if (git_base_commit is None) != (git_base_tree is None):
            raise ValueError(
                "git_base_commit and git_base_tree must be provided together "
                "(both or neither)."
            )
    activation_summary = "Swarm topology planned; root remains the shared blackboard."
    activated = False
    with kb.write_txn(conn):
        created = _create_swarm_uncommitted(
            conn, goal=goal, workers=workers, verifier_assignee=verifier_assignee,
            synthesizer_assignee=synthesizer_assignee, root_title=root_title,
            verifier_title=verifier_title, synthesizer_title=synthesizer_title, tenant=tenant,
            created_by=created_by, workspace_kind=workspace_kind, workspace_path=workspace_path,
            priority=priority, idempotency_key=idempotency_key,
        )
        root = kb.get_task(conn, created.root_id)
        if project_root is not None:
            task_ids = [created.root_id, *created.worker_ids, created.verifier_id, created.synthesizer_id]
            expected_parents = {
                created.root_id: set(),
                **{task_id: {created.root_id} for task_id in created.worker_ids},
                created.verifier_id: set(created.worker_ids),
                created.synthesizer_id: {created.verifier_id},
            }
            if len(expected_parents) != len(task_ids):
                raise ValueError("swarm workspace plan contains repeated roles")
            # Bind the immutable plan rows BEFORE activating the root.
            # ``bind_workspace_plan`` uses allow_nested=True composition so
            # the plan INSERT participates in this outer transaction and
            # rolls back atomically with anything that fails between here
            # and root activation. Direct writes outside an outer
            # transaction still use BEGIN IMMEDIATE / COMMIT.
            from hermes_cli import kanban_db_workspace as kdw
            if root is None:
                raise ValueError("swarm workspace plan has no root")
            if root.status != "blocked":
                # Replay may validate existing authority, never recreate it.
                existing_plans = {
                    task_id: kdw.resolve_workspace_plan_from_commit(conn, task_id)
                    for task_id in task_ids
                }
                if any(plan is None for plan in existing_plans.values()):
                    raise ValueError("active swarm workspace plan is missing")
                if capture_basis:
                    # Omitted inputs mean capture ONCE, not refresh on replay.
                    root_plan = existing_plans[created.root_id]
                    git_base_commit = root_plan["base_commit"]
                    git_base_tree = root_plan["base_tree"]
            elif capture_basis:
                # A new planned topology needs an actual frozen basis. A
                # non-Git or unborn project cannot become ready with NULLs.
                # Replays above reuse the stored basis without reading HEAD.
                head_proc = subprocess.run(
                    ["git", "-C", str(project_root), "rev-parse", "HEAD"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=10, check=False,
                )
                if head_proc.returncode != 0 or not head_proc.stdout.strip():
                    raise ValueError("worktree planning requires a frozen Git basis: no committed HEAD")
                git_base_commit = head_proc.stdout.strip()
                tree_proc = subprocess.run(
                    ["git", "-C", str(project_root), "rev-parse", f"{git_base_commit}^{{tree}}"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=10, check=False,
                )
                if tree_proc.returncode != 0 or not tree_proc.stdout.strip():
                    raise ValueError("worktree planning requires a frozen Git basis: commit tree unavailable")
                git_base_tree = tree_proc.stdout.strip()
            for task_id in task_ids:
                if set(kb.parent_ids(conn, task_id)) != expected_parents[task_id]:
                    raise ValueError("swarm workspace plan has foreign graph members")
                planned_path = project_root / ".worktrees" / task_id
                if planned_path.resolve() != planned_path:
                    raise ValueError("swarm workspace plan resolves outside its canonical target")
                plan = kdw.bind_workspace_plan(
                    conn,
                    task_id=task_id,
                    workspace_root=project_root,
                    workspace_path=planned_path,
                    base_commit=git_base_commit,
                    base_tree=git_base_tree,
                )
                # Persist the canonical worktree path on the task row, but
                # ONLY for an unactivated topology and ONLY when the row is
                # still pristine (no run has touched it). Active replays
                # validate unchanged rows, never rewrite them.
                if root is not None and root.status == "blocked":
                    updated = conn.execute(
                        """UPDATE tasks SET workspace_path=?
                           WHERE id=? AND status IN ('blocked','todo')
                             AND current_run_id IS NULL AND worker_pid IS NULL
                             AND workspace_kind='worktree' AND workspace_path=?
                             AND NOT EXISTS (SELECT 1 FROM task_runs WHERE task_id=tasks.id)""",
                        (str(planned_path), task_id, workspace_path),
                    )
                    if updated.rowcount != 1:
                        raise ValueError("swarm workspace plan cannot be bound before activation")
                task = kb.get_task(conn, task_id)
                if task is None or task.workspace_kind != "worktree" or task.workspace_path != str(planned_path):
                    raise ValueError("swarm workspace plan differs on replay")
                # A planned topology always requires a complete validated
                # basis, including active replay of older NULL-basis plans.
                kdw._validate_plan_for_materialization(
                    plan, task, expected_repo_root=project_root,
                )
        if root is not None and root.status == "blocked":
            if not _activate_root_inline(
                conn,
                created.root_id,
                summary=activation_summary,
                metadata={
                    "kind": "kanban_swarm_v1",
                    "goal": goal.strip(),
                    "worker_count": len(created.worker_ids),
                },
            ):
                raise RuntimeError("could not activate the completed swarm topology")
            activated = True
    if activated:
        # After commit: recompute_ready opens its own txn and must never run
        # under an open write_txn.
        kb.recompute_ready(conn)
        root = kb.get_task(conn, created.root_id)
        run = kb.latest_run(conn, created.root_id)
        kb._fire_kanban_lifecycle_hook(
            "kanban_task_completed",
            created.root_id,
            board=kb.get_current_board(),
            assignee=root.assignee if root else None,
            run_id=run.id if run else None,
            summary=activation_summary,
        )
    return created


def _create_swarm_uncommitted(
    conn: sqlite3.Connection, *, goal: str, workers: Iterable[SwarmWorkerSpec],
    verifier_assignee: str, synthesizer_assignee: str, root_title: Optional[str],
    verifier_title: str, synthesizer_title: str, tenant: Optional[str], created_by: str,
    workspace_kind: Optional[str], workspace_path: Optional[str], priority: int, idempotency_key: Optional[str],
) -> SwarmCreated:
    """Create the swarm graph inside the caller's transaction: planning root
    (``blocked`` until the caller activates it), parallel workers, a verifier
    waiting on every worker, and a synthesizer waiting on the verifier."""
    goal = _require_text(goal, "goal")
    verifier_assignee = _require_text(verifier_assignee, "verifier_assignee")
    synthesizer_assignee = _require_text(synthesizer_assignee, "synthesizer_assignee")
    worker_specs = list(workers)
    if not worker_specs:
        raise ValueError("at least one worker is required")
    for i, spec in enumerate(worker_specs, start=1):
        _require_text(spec.profile, f"workers[{i}].profile")
        _require_text(spec.title, f"workers[{i}].title")

    common = dict(
        created_by=created_by, tenant=tenant,
        workspace_kind=workspace_kind, workspace_path=workspace_path,
    )
    root = kb.create_task(
        conn,
        title=root_title or f"Swarm: {goal.splitlines()[0][:80]}",
        body="Kanban Swarm v1 planning/root card. This card is completed "
             "immediately so parallel workers can start while it remains the "
             f"shared blackboard and audit anchor.\n\nGoal:\n{goal}",
        assignee=created_by,
        priority=priority,
        idempotency_key=idempotency_key,
        initial_status="blocked",
        **common,
    )

    # Idempotency may return an existing root: recover its topology from the
    # blackboard instead of duplicating the graph.
    existing = latest_blackboard(conn, root).get("topology")
    if isinstance(existing, dict):
        worker_ids = [str(x) for x in existing.get("worker_ids", []) if x]
        verifier_id = existing.get("verifier_id")
        synthesizer_id = existing.get("synthesizer_id")
        if worker_ids and verifier_id and synthesizer_id:
            return SwarmCreated(root, worker_ids, str(verifier_id), str(synthesizer_id))

    context_suffix = _swarm_context(root, goal)
    worker_ids = [
        kb.create_task(
            conn,
            title=spec.title,
            body=(spec.body or "") + context_suffix,
            assignee=spec.profile,
            parents=[root],
            priority=spec.priority or priority,
            skills=spec.skills or None,
            max_runtime_seconds=spec.max_runtime_seconds,
            **common,
        )
        for spec in worker_specs
    ]
    verifier = kb.create_task(
        conn,
        title=verifier_title,
        body=(
            "Review every worker handoff and blackboard update. Gate the swarm: "
            "complete only with metadata {\"gate\": \"pass\"} when evidence is "
            "sufficient; otherwise block with exact missing work."
            + context_suffix
        ),
        assignee=verifier_assignee,
        parents=worker_ids,
        priority=priority,
        skills=["requesting-code-review"],
        **common,
    )
    synthesizer = kb.create_task(
        conn,
        title=synthesizer_title,
        body=(
            "Synthesize the verified worker outputs into the final deliverable. "
            "Do not start until the verifier has passed the gate."
            + context_suffix
        ),
        assignee=synthesizer_assignee,
        parents=[verifier],
        priority=priority,
        skills=["humanizer"],
        **common,
    )

    created = SwarmCreated(root, worker_ids, verifier, synthesizer)
    post_blackboard_update(conn, root, author=created_by, key="topology", value=created.as_dict() | {"goal": goal})
    return created


def post_blackboard_update(conn: sqlite3.Connection, root_id: str, *, author: str, key: str, value: Any) -> int:
    """Append one structured update to the swarm root blackboard."""
    _require_text(root_id, "root_id")
    author = _require_text(author, "author")
    key = _require_text(key, "key")
    payload = json.dumps({"key": key, "value": value}, ensure_ascii=False, sort_keys=True)
    return kb.add_comment(conn, root_id, author=author, body=BLACKBOARD_PREFIX + payload)


def latest_blackboard(conn: sqlite3.Connection, root_id: str) -> dict[str, Any]:
    """Merge structured blackboard comments on a root card. Later comments
    replace earlier values for the same key; ``_authors`` records the author
    of the winning value for traceability."""
    merged: dict[str, Any] = {}
    authors: dict[str, str] = {}
    for comment in kb.list_comments(conn, root_id):
        body = comment.body or ""
        if not body.startswith(BLACKBOARD_PREFIX):
            continue
        try:
            payload = json.loads(body[len(BLACKBOARD_PREFIX):])
        except json.JSONDecodeError:
            continue
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            continue
        merged[key] = payload.get("value")
        authors[key] = comment.author
    if authors:
        merged["_authors"] = authors
    return merged


def parse_worker_arg(raw: str) -> SwarmWorkerSpec:
    """Parse CLI ``--worker profile:title[:skill,skill]`` values."""
    parts = [p.strip() for p in raw.split(":", 2)]
    if len(parts) < 2:
        raise ValueError("worker must be profile:title or profile:title:skill,skill")
    skills = [s.strip() for s in parts[2].split(",") if s.strip()] if len(parts) == 3 and parts[2] else []
    return SwarmWorkerSpec(profile=parts[0], title=parts[1], body=parts[1], skills=skills)
