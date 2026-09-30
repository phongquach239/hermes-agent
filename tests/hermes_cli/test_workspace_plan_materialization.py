"""F02 frozen-workspace-basis materialization tests (R2 repair + R3
literal-basis + real-checkout-membership completion).

Real Core producer + disposable SQLite + real Git. The tests target the
``_resolve_worktree_workspace`` / ``_ensure_git_worktree`` seam when
``conn`` is passed and a ``task_workspace_plans`` row exists for the
task. The repair contract:

* Validate the plan end-to-end (version, paired commit/tree, recomputed
  digest, foreign-root guard, plan-vs-task path alignment, Git
  commit→tree relationship) BEFORE any directory creation or
  ``worktree add``. Plan failure is refusal, NOT a legacy fallback.
* Use the stored selected project (``plan.workspace_root``) as the
  authoritative repo. Linked checkouts are accepted only when they share
  ``--git-common-dir`` with the canonical root — a foreign clone of the
  same upstream with identical objects is still foreign.
* The selected project MUST be an actual checkout top-level — a nested
  folder that inherits a parent's ``--git-common-dir`` is NOT a
  checkout root and refuses before any worktree add.
* The frozen ``base_commit`` MUST be the literal full object id
  (``rev-parse --verify <base_commit>^{commit}`` returns the exact same
  string). Symbolic refs, branch names, abbreviated SHAs refuse
  before any worktree-add.
* If the target checkout already exists with the same common dir,
  require correct REGISTERED worktree path (per ``git worktree list
  --porcelain -z``), expected branch, and exact frozen HEAD/tree. A
  plain directory inheriting the parent's git state, or a copied
  ``.git`` file pointing at a sibling worktree's metadata, refuses
  BEFORE any mutation — ``--git-common-dir`` walks up to the parent so
  those paths pass looser branch/HEAD probes.
* If the branch already exists but the target does not, attach the
  worktree WITHOUT resetting (``worktree add <target> <branch>``).
  Never use ``-B``, ``--force``, or ``reset`` on the planned path.
* Use raw stored identity strings in the basis digest and the path
  equality checks; whitespace around ``workspace_root`` /
  ``workspace_path`` MUST break digest recomputation rather than be
  silently repaired into a different byte sequence.
* Preserve all existing product tests (legacy ``resolve_workspace``
  without ``conn`` is unchanged). No modified parent probes.

These tests are L1 unit / integration. They hit real Core producers and
SQLite migrations, NOT the whole HM consumer surface (which lives in
``hm-loop-candidate-core-20260928`` and is not our concern here).
"""

from __future__ import annotations

import inspect
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kdc
from hermes_cli import kanban_db_workspace as ws


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


def _commit_file(repo_root: Path, content: str) -> tuple[str, str]:
    (repo_root / "basis.txt").write_text(content)
    _git(repo_root, "add", "basis.txt")
    _git(repo_root, "commit", "-q", "-m", "offline fixture basis")
    return (
        _git(repo_root, "rev-parse", "HEAD").stdout.strip(),
        _git(repo_root, "rev-parse", "HEAD^{tree}").stdout.strip(),
    )


def _foreign_clone(source: Path, parent: Path, name: str) -> Path:
    """Foreign clone of ``source`` → ``parent/name`` (no shared gitdir)."""
    target = parent / name
    subprocess.run(
        ["git", "clone", "-q", str(source), str(target)],
        check=True,
    )
    return target


def _linked_worktree(repo_root: Path, branch: str, path: Path) -> Path:
    """Linked checkout of ``repo_root`` (shared ``.git``)."""
    path.mkdir(parents=True, exist_ok=True)
    proc = _git(repo_root, "worktree", "add", "-q", "-b", branch, str(path))
    assert proc.returncode == 0, proc.stderr
    return path


# ---------------------------------------------------------------------------
# DB helpers — apply the v1 migration set so ``task_workspace_authority``,
# ``task_workspace_plans`` and ``hm_kanban_schema_journal`` are present.
# ---------------------------------------------------------------------------


def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kdc._migrate_add_optional_columns(conn)
    conn.commit()
    return conn


@pytest.fixture
def workspace_db(tmp_path):
    db = tmp_path / "fixture.db"
    conn = _open_db(str(db))
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Plan-binding helper — write a canonical ``task_workspace_plans`` row
# straight from the producer (``bind_workspace_plan``) so the digest is
# always consistent with the actual Core formula. Tests then mutate the
# row in place to inject each fault.
# ---------------------------------------------------------------------------


def _bind_plan(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    workspace_root: Path,
    base_commit: str,
    base_tree: str,
) -> dict:
    plan = ws.bind_workspace_plan(
        conn,
        task_id=task_id,
        workspace_root=workspace_root,
        workspace_path=workspace_root / ".worktrees" / task_id,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    return plan


def _insert_task(
    conn: sqlite3.Connection,
    task_id: str,
    workspace_path: str | None,
    workspace_kind: str = "worktree",
) -> None:
    conn.execute(
        "INSERT INTO tasks(id, title, status, workspace_kind, workspace_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, task_id, "blocked", workspace_kind, workspace_path, 1),
    )
    conn.commit()


def _snapshot(conn: sqlite3.Connection) -> dict:
    """Snapshot every mutable kanban table — used to prove that a refused
    resolution did NOT silently rewrite any row."""
    tables = [
        "tasks", "task_runs", "task_events",
        "task_workspace_plans", "task_workspace_authority",
    ]
    return {
        table: [tuple(row) for row in conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid",
        )]
        for table in tables
    }


def _replace_plan(
    conn: sqlite3.Connection,
    task,
    *,
    root: str,
    target: str,
    base: str,
    tree: str,
) -> None:
    """Overwrite a bound plan with a deliberate fault (literal identity,
    padded root, etc). Recomputes the digest so the tamper check is
    consistent with the (possibly invalid) stored fields; the test then
    expects the resolver to refuse BEFORE any side effect."""
    digest = ws.basis_digest(
        version=1, task_id=task.id,
        workspace_root=root, workspace_path=target,
        base_commit=base, base_tree=tree,
    )
    conn.execute(
        "UPDATE task_workspace_plans SET "
        "workspace_root=?, workspace_path=?, base_commit=?, base_tree=?, "
        "basis_digest=? WHERE task_id=?",
        (root, target, base, tree, digest, task.id),
    )
    conn.execute(
        "UPDATE tasks SET workspace_path=? WHERE id=?",
        (target, task.id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# POSITIVE CONTROLS — primary + linked selected roots are honored and the
# frozen commit/tree are observed at branch creation. This matches the
# ``head-drift controls`` set Main reported PASS at T1316.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("linked", [False, True])
def test_planned_materialization_uses_frozen_commit_when_branch_does_not_exist(
    workspace_db, tmp_path, linked
):
    """Primary (or linked-checkout) selected root, fresh worktree path,
    branch absent ⇒ ``worktree add -b <branch> <target> <commit>``;
    HEAD on the resulting checkout is the frozen commit."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    selected = primary
    if linked:
        selected = _linked_worktree(primary, "selected-project",
                                     tmp_path / "linked-selected-project")
    # Sanity: linked checkout shares --git-common-dir with primary.
    assert ws._git_common_dir(selected) == ws._git_common_dir(primary)

    _insert_task(workspace_db, "t1", str(selected / ".worktrees" / "t1"))
    _bind_plan(
        workspace_db,
        task_id="t1",
        workspace_root=selected,
        base_commit=base_commit,
        base_tree=base_tree,
    )

    task = kb.get_task(workspace_db, "t1")
    parameters = inspect.signature(ws.resolve_workspace).parameters
    assert "conn" in parameters
    checkout = ws.resolve_workspace(task, conn=workspace_db)
    assert checkout == selected / ".worktrees" / "t1"
    head = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    tree = _git(checkout, "rev-parse", "HEAD^{tree}").stdout.strip()
    assert head == base_commit
    assert tree == base_tree


def test_planned_materialization_attaches_existing_branch_without_reset(
    workspace_db, tmp_path
):
    """Branch exists at the frozen base_commit (a SAME-commit control);
    target absent ⇒ attach VERBATIM, do NOT ``-B``/``--force``/
    ``reset``. The branch tip is preserved."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    # Pre-create the branch AT the frozen base_commit. This is the
    # valid same-commit control: attach verbatim without moving the
    # branch tip, never ``-B``/``--force``/``reset``.
    branch = "wt/t2"
    _git(primary, "branch", branch, base_commit)
    tip_before = _git(primary, "rev-parse", branch).stdout.strip()
    assert tip_before == base_commit

    _insert_task(workspace_db, "t2", str(primary / ".worktrees" / "t2"))
    _bind_plan(
        workspace_db,
        task_id="t2",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    task = kb.get_task(workspace_db, "t2")
    checkout = ws.resolve_workspace(task, conn=workspace_db)
    assert checkout == primary / ".worktrees" / "t2"
    # The branch tip is preserved at base_commit (no reset).
    tip_after = _git(primary, "rev-parse", branch).stdout.strip()
    assert tip_after == tip_before == base_commit, (
        "planner reset the existing branch tip away from the frozen commit"
    )
    # The resulting worktree's HEAD equals the frozen base_commit.
    head_after = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    tree_after = _git(checkout, "rev-parse", "HEAD^{tree}").stdout.strip()
    assert head_after == base_commit
    assert tree_after == base_tree


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS — refused BEFORE any side effect, no ``-B``/``--force``
# /reset. Each test sets up the fault, snapshots, asserts refusal + clean
# state, then asserts the relevant pre-existing artifact survived.
# ---------------------------------------------------------------------------


def test_plan_mismatch_wrong_branch_refuses_without_reset(workspace_db, tmp_path):
    """Pre-existing branch points at a non-frozen commit; the planned
    path is empty ⇒ REFUSE without side effect. Branch tip and tree
    remain at the unrelated commit; the worktree target is never
    created; no ``-B``/``--force``/reset is issued."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    other_commit, other_tree = _commit_file(primary, "later unrelated\n")
    assert other_commit != base_commit
    branch = "wt/t_branch"
    _git(primary, "branch", branch, other_commit)
    before_branch = _git(primary, "rev-parse", branch).stdout.strip()
    before_tree = _git(primary, "rev-parse", f"{before_branch}^{{tree}}").stdout.strip()
    target = primary / ".worktrees" / "t_branch"
    # Target absent; the only pre-existing state is the wrong branch tip.
    assert not target.exists()

    _insert_task(workspace_db, "t_branch", str(target))
    _bind_plan(
        workspace_db,
        task_id="t_branch",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    task = kb.get_task(workspace_db, "t_branch")
    with pytest.raises(ValueError):
        ws.resolve_workspace(task, conn=workspace_db)
    # Branch tip + tree unchanged after the refused resolution.
    after_branch = _git(primary, "rev-parse", branch).stdout.strip()
    after_tree = _git(primary, "rev-parse", f"{after_branch}^{{tree}}").stdout.strip()
    assert after_branch == before_branch == other_commit
    assert after_tree == before_tree == other_tree
    # Target directory was never created.
    assert not target.exists()


def test_plan_mismatch_wrong_checkout_head_refuses_without_reset(
    workspace_db, tmp_path
):
    """The worktree path already exists as a linked checkout on the
    expected branch but at the WRONG commit ⇒ refuse, do NOT reset."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    later_commit, later_tree = _commit_file(primary, "wrong later\n")
    assert later_commit != base_commit
    target = primary / ".worktrees" / "t_wrong"
    # Pre-create the canonical worktree on the expected branch, but at
    # the wrong HEAD — simulates an in-flight worker that drifted.
    _git(primary, "worktree", "add", "-q", "-b", "wt/t_wrong", str(target), later_commit)
    before_head = _git(target, "rev-parse", "HEAD").stdout.strip()
    assert before_head == later_commit

    _insert_task(workspace_db, "t_wrong", str(target))
    _bind_plan(
        workspace_db,
        task_id="t_wrong",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    task = kb.get_task(workspace_db, "t_wrong")
    with pytest.raises(ValueError):
        ws.resolve_workspace(task, conn=workspace_db)
    # The worktree's HEAD is preserved (no reset on refusal).
    after_head = _git(target, "rev-parse", "HEAD").stdout.strip()
    assert after_head == before_head


def test_foreign_task_path_refuses_before_creation(workspace_db, tmp_path):
    """A foreign clone of the selected repo shares objects but a
    different ``--git-common-dir``. ``task.workspace_path`` set to that
    foreign clone refuses BEFORE any ``worktree add`` runs."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    foreign = _foreign_clone(primary, tmp_path, "foreign-clone")
    # ``foreign`` shares the commit/tree but its git-common-dir is distinct.
    assert ws._git_common_dir(foreign) != ws._git_common_dir(primary)

    _insert_task(workspace_db, "t_foreign", str(foreign / ".worktrees" / "t_foreign"))
    _bind_plan(
        workspace_db,
        task_id="t_foreign",
        workspace_root=primary,  # plan still anchors the primary
        base_commit=base_commit,
        base_tree=base_tree,
    )
    task = kb.get_task(workspace_db, "t_foreign")
    foreign_target = foreign / ".worktrees" / "t_foreign"
    assert not foreign_target.exists()
    with pytest.raises(ValueError):
        ws.resolve_workspace(task, conn=workspace_db)
    # No directory created on the foreign clone.
    assert not foreign_target.exists()
    assert not (foreign / ".worktrees").exists()


def test_tampered_digest_refuses_without_side_effects(workspace_db, tmp_path):
    """A plan row whose stored digest disagrees with the recomputed
    digest refuses BEFORE any side effect."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    _insert_task(workspace_db, "t_tamper", str(primary / ".worktrees" / "t_tamper"))
    plan = _bind_plan(
        workspace_db,
        task_id="t_tamper",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    # Tamper: rewrite the digest to garbage.
    workspace_db.execute(
        "UPDATE task_workspace_plans SET basis_digest=? WHERE task_id=?",
        ("deadbeef" * 8, "t_tamper"),
    )
    workspace_db.commit()
    task = kb.get_task(workspace_db, "t_tamper")
    target = primary / ".worktrees" / "t_tamper"
    assert not target.exists()
    with pytest.raises(ValueError):
        ws.resolve_workspace(task, conn=workspace_db)
    assert not target.exists()


def test_tampered_tree_refuses_without_side_effects(workspace_db, tmp_path):
    """A plan row whose stored ``base_tree`` does not match the tree of
    the stored ``base_commit`` refuses BEFORE any side effect. The stored
    digest is RECOMPUTED over the deliberately wrong but REAL tree so the
    digest check passes and the failure isolates the Git commit→tree
    pair validation step (not merely a digest mismatch)."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    _, wrong_tree = _commit_file(primary, "deliberately different tree\n")
    assert wrong_tree != base_tree
    _insert_task(workspace_db, "t_tree", str(primary / ".worktrees" / "t_tree"))
    plan = _bind_plan(
        workspace_db,
        task_id="t_tree",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    # Tamper: rewrite the tree to a real, deliberately wrong tree, AND
    # recompute the digest so the digest check still passes. This isolates
    # the Git pair (commit→tree) validation.
    new_digest = ws.basis_digest(
        version=plan["plan_version"],
        task_id="t_tree",
        workspace_root=plan["workspace_root"],
        workspace_path=plan["workspace_path"],
        base_commit=base_commit,
        base_tree=wrong_tree,
    )
    workspace_db.execute(
        "UPDATE task_workspace_plans SET base_tree=?, basis_digest=? WHERE task_id=?",
        (wrong_tree, new_digest, "t_tree"),
    )
    workspace_db.commit()
    task = kb.get_task(workspace_db, "t_tree")
    target = primary / ".worktrees" / "t_tree"
    assert not target.exists()
    with pytest.raises(ValueError):
        ws.resolve_workspace(task, conn=workspace_db)
    assert not target.exists()


def test_unknown_plan_version_refuses(workspace_db, tmp_path):
    """A plan row with an unsupported ``plan_version`` is refused
    immediately, no Git worktree add, no recursion into the legacy
    fallback path."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    _insert_task(workspace_db, "t_ver", str(primary / ".worktrees" / "t_ver"))
    plan = _bind_plan(
        workspace_db,
        task_id="t_ver",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    workspace_db.execute(
        "UPDATE task_workspace_plans SET plan_version=? WHERE task_id=?",
        (99, "t_ver"),
    )
    workspace_db.commit()
    task = kb.get_task(workspace_db, "t_ver")
    target = primary / ".worktrees" / "t_ver"
    assert not target.exists()
    with pytest.raises(ValueError):
        ws.resolve_workspace(task, conn=workspace_db)
    assert not target.exists()


def test_legacy_unplanned_path_still_works(workspace_db, tmp_path):
    """Without a plan row the resolver keeps its legacy unplanned
    behavior (no DB read for the frozen basis)."""
    primary = _init_repo(tmp_path / "primary")
    target = primary / ".worktrees" / "t_legacy"
    _insert_task(workspace_db, "t_legacy", str(target))
    # NOTE: we explicitly do NOT bind a plan.
    task = kb.get_task(workspace_db, "t_legacy")
    # The signature still exposes ``conn``; passing ``None`` keeps the
    # legacy fallback path (no plan read, no frozen-commit binding).
    checkout = ws.resolve_workspace(task, conn=None)
    assert checkout == target
    head = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    # HEAD is the repo's current tip (legacy behavior), not pinned.
    repo_head = _git(primary, "rev-parse", "HEAD").stdout.strip()
    assert head == repo_head


# ---------------------------------------------------------------------------
# Main L1 mirror — frozen-basis replay + path preservation.
#
# These mirror the parent probes in
# ``test_materialization_replay_and_paths.py`` so the same acceptance lives
# INSIDE the product test file. They use only the Core producer + SQLite +
# real Git, with explicit fault injection on disposable fixtures. They MUST
# agree with the parent probes on every assertion.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("existing", ["none", "branch", "checkout"])
def test_exact_frozen_basis_materialization_and_replay_succeed(
    workspace_db, tmp_path, existing,
):
    """Valid same-commit paths through the resolver:

    * ``none``    – fresh branch + worktree, frozen commit is HEAD.
    * ``branch``  – pre-existing branch already at the frozen commit;
                    attach verbatim, no ``-B``/``--force``/reset.
    * ``checkout`` – a fully materialized worktree on the expected branch
                    at the frozen commit; the resolver accepts it.

    Repeated resolution MUST remain usable.
    """
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    target = primary / ".worktrees" / "replay"
    branch = "wt/replay"

    _insert_task(workspace_db, "replay", str(target))
    _bind_plan(
        workspace_db,
        task_id="replay",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )

    if existing == "branch":
        # Pre-create the branch AT the frozen base_commit (the valid
        # same-commit attach path).
        _git(primary, "branch", branch, base_commit)
    elif existing == "checkout":
        # Pre-create the full linked worktree at the frozen base_commit.
        _git(
            primary, "worktree", "add", "-q", "-b", branch,
            str(target), base_commit,
        )

    task = kb.get_task(workspace_db, "replay")
    checkout = ws.resolve_workspace(task, conn=workspace_db)
    assert checkout == target
    head = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    tree = _git(checkout, "rev-parse", "HEAD^{tree}").stdout.strip()
    assert head == base_commit
    assert tree == base_tree
    # Repeated resolution MUST remain usable (no destructive side effect).
    again = ws.resolve_workspace(task, conn=workspace_db)
    assert again == checkout


def test_frozen_target_parent_symlink_is_rejected_before_creation(
    workspace_db, tmp_path,
):
    """A symlink in the target's parent directory must refuse BEFORE
    the resolver follows it and writes outside the frozen project."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    target = primary / ".worktrees" / "symlink_rejected"
    _insert_task(workspace_db, "symlink_rejected", str(target))
    _bind_plan(
        workspace_db,
        task_id="symlink_rejected",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )

    outside = tmp_path / "outside-selected-project"
    outside.mkdir()
    assert not target.parent.exists()
    # The parent directory becomes a symlink pointing outside the
    # project; a naive resolve(strict=False) would silently follow it.
    target.parent.symlink_to(outside, target_is_directory=True)
    before_refs = _git(primary, "show-ref").stdout
    task = kb.get_task(workspace_db, "symlink_rejected")
    with pytest.raises((ValueError, RuntimeError)):
        ws.resolve_workspace(task, conn=workspace_db)
    # No new refs created and no files written outside the project.
    assert _git(primary, "show-ref").stdout == before_refs
    assert list(outside.iterdir()) == [], (
        "symlink escape created files outside the frozen project"
    )


def test_empty_task_workspace_does_not_qualify_as_matching_plan(
    workspace_db, tmp_path,
):
    """An empty ``task.workspace_path`` is a refusal: the schema binds
    workspace_path at activation, and a missing field is not a license
    to invent one from the plan row."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    target = primary / ".worktrees" / "empty_ws"
    _insert_task(workspace_db, "empty_ws", str(target))
    _bind_plan(
        workspace_db,
        task_id="empty_ws",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    workspace_db.execute(
        "UPDATE tasks SET workspace_path=NULL WHERE id=?", ("empty_ws",),
    )
    workspace_db.commit()
    task = kb.get_task(workspace_db, "empty_ws")
    with pytest.raises((ValueError, RuntimeError)):
        ws.resolve_workspace(task, conn=workspace_db)
    assert not target.exists()


def test_self_consistent_noncanonical_plan_path_is_rejected(
    workspace_db, tmp_path,
):
    """Plan and task both agree on a noncanonical path that differs from
    ``<plan_root>/.worktrees/<task_id>``; refuse, do not silently
    materialize there. The frozen canonical form is the producer-derived
    anchor, not a value the resolver may rewrite."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    canonical_target = primary / ".worktrees" / "noncanonical"
    outside = tmp_path / "not-canonical-worktree"
    # Bind the plan with the CANONICAL target first so the row exists
    # (digest is consistent with the stored fields).
    _insert_task(workspace_db, "noncanonical", str(canonical_target))
    _bind_plan(
        workspace_db,
        task_id="noncanonical",
        workspace_root=primary,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    # Then mutate task AND plan to a self-consistent NON-canonical path,
    # recomputing the digest so the tamper detection accepts the row.
    digest = ws.basis_digest(
        version=1, task_id="noncanonical",
        workspace_root=str(primary),
        workspace_path=str(outside),
        base_commit=base_commit, base_tree=base_tree,
    )
    workspace_db.execute(
        "UPDATE tasks SET workspace_path=? WHERE id=?",
        (str(outside), "noncanonical"),
    )
    workspace_db.execute(
        "UPDATE task_workspace_plans SET workspace_path=?, basis_digest=? "
        "WHERE task_id=?",
        (str(outside), digest, "noncanonical"),
    )
    workspace_db.commit()
    task = kb.get_task(workspace_db, "noncanonical")
    with pytest.raises((ValueError, RuntimeError)):
        ws.resolve_workspace(task, conn=workspace_db)
    assert not outside.exists()


# ---------------------------------------------------------------------------
# R3 literal-basis + real-checkout-membership guards.
#
# Migrated from the parent L1 probe
# ``test_materialization_literal_basis_and_membership.py`` so the
# acceptance lives inside the product test file. Self-contained — no
# private-evidence imports. Each fault refuses BEFORE any worktree add
# (no ref, no branch, no target directory mutation). Positive controls
# keep the existing branch-name and replay guarantees.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["full", "symbolic", "abbreviated"])
def test_plan_requires_literal_full_commit_identity(workspace_db, tmp_path, kind):
    """``base_commit`` MUST be the literal full object id. Symbolic
    (HEAD) and abbreviated (short SHA) refs both resolve to a real
    commit but are NOT the immutable identity the producer bound.

    Positive control (``full``): resolver creates the worktree and
    pins HEAD to ``base``. Negative cases refuse WITHOUT side effects
    (no refs, no target directory, no DB mutation).
    """
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    target = primary / ".worktrees" / "literal"

    _insert_task(workspace_db, "literal", str(target))
    _bind_plan(
        workspace_db, task_id="literal",
        workspace_root=primary, base_commit=base_commit, base_tree=base_tree,
    )

    frozen = {"full": base_commit, "symbolic": "HEAD", "abbreviated": base_commit[:12]}[kind]
    # ``_replace_plan`` recomputes the digest over the (possibly invalid)
    # stored fields so the digest check is consistent and the failure
    # isolates the literal-identity check.
    task = kb.get_task(workspace_db, "literal")
    _replace_plan(
        workspace_db, task, root=str(primary),
        target=str(target), base=frozen, tree=base_tree,
    )
    task = kb.get_task(workspace_db, "literal")

    before_db = _snapshot(workspace_db)
    before_refs = _git(primary, "show-ref").stdout

    if kind == "full":
        checkout = ws.resolve_workspace(task, conn=workspace_db)
        assert checkout == target
        assert _git(target, "rev-parse", "HEAD").stdout.strip() == base_commit
    else:
        with pytest.raises((ValueError, RuntimeError)):
            ws.resolve_workspace(task, conn=workspace_db)
        assert not target.parent.exists(), "invalid identity made filesystem changes"
        assert _git(primary, "show-ref").stdout == before_refs
        assert _snapshot(workspace_db) == before_db


def test_selected_root_must_be_checkout_root_not_an_arbitrary_subdir(
    workspace_db, tmp_path,
):
    """``plan.workspace_root`` MUST be an actual git checkout top-level
    (primary or a genuine linked worktree). A plain subdirectory that
    merely inherits its parent's ``--git-common-dir`` is NOT a checkout
    root and refuses BEFORE any worktree add."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    nested = primary / "ordinary-folder"
    nested.mkdir()
    target = nested / ".worktrees" / "nested"

    _insert_task(workspace_db, "nested", str(target))
    _bind_plan(
        workspace_db, task_id="nested",
        workspace_root=primary, base_commit=base_commit, base_tree=base_tree,
    )
    # Rewriting the plan root to the nested folder shares
    # ``--git-common-dir`` with primary (the foreign-clone guard alone
    # is insufficient) and triggers the new checkout-top-level guard.
    task = kb.get_task(workspace_db, "nested")
    _replace_plan(
        workspace_db, task, root=str(nested),
        target=str(target), base=base_commit, tree=base_tree,
    )
    task = kb.get_task(workspace_db, "nested")

    before_refs = _git(primary, "show-ref").stdout
    with pytest.raises((ValueError, RuntimeError)):
        ws.resolve_workspace(task, conn=workspace_db)
    assert not target.parent.exists()
    assert _git(primary, "show-ref").stdout == before_refs


@pytest.mark.parametrize("kind", ["registered", "plain-directory", "copied-git-link"])
def test_existing_target_is_registered_checkout_not_git_discovery(
    workspace_db, tmp_path, kind,
):
    """When the target already exists, it MUST be a registered worktree
    of the selected repo. A plain directory inside ``.worktrees`` and a
    copied ``.git`` file pointing at a sibling worktree's gitdir both
    pass looser branch/HEAD probes (their branch HEAD walks up to the
    parent) but are not registered in ``git worktree list`` and refuse.

    Positive control (``registered``): a real ``worktree add`` on a
    CUSTOM branch name (``feature/custom-valid-name``, NOT ``wt/<id>``)
    is accepted and replayable.
    """
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    target = primary / ".worktrees" / "register"
    branch = "feature/custom-valid-name"

    _insert_task(workspace_db, "register", str(target))
    _bind_plan(
        workspace_db, task_id="register",
        workspace_root=primary, base_commit=base_commit, base_tree=base_tree,
    )
    workspace_db.execute(
        "UPDATE tasks SET branch_name=? WHERE id=?",
        (branch, "register"),
    )
    workspace_db.commit()
    task = kb.get_task(workspace_db, "register")

    if kind == "registered":
        _git(primary, "worktree", "add", "-qb", branch, str(target), base_commit)
    elif kind == "plain-directory":
        _git(primary, "checkout", "-qb", branch, base_commit)
        target.mkdir(parents=True)
    else:  # copied-git-link
        sibling = primary / ".worktrees" / "different-registered-checkout"
        _git(primary, "worktree", "add", "-qb", branch, str(sibling), base_commit)
        target.mkdir()
        (target / ".git").write_bytes((sibling / ".git").read_bytes())

    before_list = _git(primary, "worktree", "list", "--porcelain").stdout
    before_refs = _git(primary, "show-ref").stdout

    if kind == "registered":
        checkout = ws.resolve_workspace(task, conn=workspace_db)
        assert checkout == target
        # Replay remains usable: a registered checkout at the frozen
        # commit is the same idempotent anchor.
        again = ws.resolve_workspace(task, conn=workspace_db)
        assert again == target
    else:
        with pytest.raises((ValueError, RuntimeError)):
            ws.resolve_workspace(task, conn=workspace_db)
    assert _git(primary, "worktree", "list", "--porcelain").stdout == before_list
    assert _git(primary, "show-ref").stdout == before_refs


def test_padding_stored_project_identity_is_not_repaired_by_strip(
    workspace_db, tmp_path,
):
    """Whitespace around the stored ``workspace_root`` MUST break the
    digest recomputation. The producer bound the exact byte sequence;
    the validator MUST NOT silently strip a different spelling into
    the same identity."""
    primary = _init_repo(tmp_path / "primary")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    target = primary / ".worktrees" / "padded"

    _insert_task(workspace_db, "padded", str(target))
    plan = _bind_plan(
        workspace_db, task_id="padded",
        workspace_root=primary, base_commit=base_commit, base_tree=base_tree,
    )
    # Pad the stored root WITHOUT recomputing the digest: this isolates
    # the raw-stored-identity contract (the digest must change because
    # the stored root changed; ``strip`` would let it slip through).
    padded = " " + str(primary) + " "
    assert padded != plan["workspace_root"]
    workspace_db.execute(
        "UPDATE task_workspace_plans SET workspace_root=? WHERE task_id=?",
        (padded, "padded"),
    )
    workspace_db.commit()

    task = kb.get_task(workspace_db, "padded")
    before_db = _snapshot(workspace_db)
    with pytest.raises((ValueError, RuntimeError)):
        ws.resolve_workspace(task, conn=workspace_db)
    assert not target.parent.exists()
    assert _snapshot(workspace_db) == before_db


@pytest.mark.parametrize("suffix", ["", "/.", "/"])
def test_raw_root_spelling_is_canonical_before_path_parsing(
    workspace_db, tmp_path, suffix,
):
    """A matching digest cannot authorize an aliased stored root spelling."""
    primary = _init_repo(tmp_path / "project with spaces")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    target = primary / ".worktrees" / "literal-root"
    _insert_task(workspace_db, "literal-root", str(target))
    _bind_plan(
        workspace_db, task_id="literal-root", workspace_root=primary,
        base_commit=base_commit, base_tree=base_tree,
    )
    task = kb.get_task(workspace_db, "literal-root")
    _replace_plan(
        workspace_db, task, root=str(primary) + suffix,
        target=str(target), base=base_commit, tree=base_tree,
    )
    task = kb.get_task(workspace_db, "literal-root")
    before_db = _snapshot(workspace_db)
    before_refs = _git(primary, "show-ref").stdout
    if suffix:
        with pytest.raises(ValueError, match="literal canonical absolute spelling"):
            ws.resolve_workspace(task, conn=workspace_db)
        assert not target.parent.exists()
        assert _git(primary, "show-ref").stdout == before_refs
    else:
        assert ws.resolve_workspace(task, conn=workspace_db) == target
        assert ws.resolve_workspace(task, conn=workspace_db) == target
        assert _git(target, "rev-parse", "HEAD").stdout.strip() == base_commit
    assert _snapshot(workspace_db) == before_db


@pytest.mark.parametrize("copied_pointer", [False, True])
def test_registered_target_owns_its_git_metadata(
    workspace_db, tmp_path, copied_pointer,
):
    """Both paths registered: a copied sibling pointer must still refuse."""
    primary = _init_repo(tmp_path / "project with spaces")
    base_commit, base_tree = _commit_file(primary, "frozen\n")
    target = primary / ".worktrees" / "metadata-owner"
    branch = "feature/requested-branch"
    _insert_task(workspace_db, "metadata-owner", str(target))
    _bind_plan(
        workspace_db, task_id="metadata-owner", workspace_root=primary,
        base_commit=base_commit, base_tree=base_tree,
    )
    workspace_db.execute(
        "UPDATE tasks SET branch_name=? WHERE id=?", (branch, "metadata-owner"),
    )
    workspace_db.commit()
    task = kb.get_task(workspace_db, "metadata-owner")
    target_branch = "feature/actual-target" if copied_pointer else branch
    created = _git(primary, "worktree", "add", "-qb", target_branch, str(target), base_commit)
    assert created.returncode == 0, created.stderr
    if copied_pointer:
        sibling = target.parent / "registered sibling"
        created = _git(primary, "worktree", "add", "-qb", branch, str(sibling), base_commit)
        assert created.returncode == 0, created.stderr
        (target / ".git").write_bytes((sibling / ".git").read_bytes())
    # Isolate metadata ownership: branch, objects, toplevel and list membership pass.
    assert _git(target, "branch", "--show-current").stdout.strip() == branch
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == base_commit
    assert _git(target, "rev-parse", "--show-toplevel").stdout.strip() == str(target)
    before_list = _git(primary, "worktree", "list", "--porcelain").stdout
    assert f"worktree {target}\n" in before_list
    before_refs = _git(primary, "show-ref").stdout
    before_pointer = (target / ".git").read_bytes()
    before_db = _snapshot(workspace_db)
    if copied_pointer:
        with pytest.raises(ValueError, match="not a registered worktree"):
            ws.resolve_workspace(task, conn=workspace_db)
    else:
        assert ws.resolve_workspace(task, conn=workspace_db) == target
        assert ws.resolve_workspace(task, conn=workspace_db) == target
    assert _snapshot(workspace_db) == before_db
    assert (target / ".git").read_bytes() == before_pointer
    assert _git(primary, "worktree", "list", "--porcelain").stdout == before_list
    assert _git(primary, "show-ref").stdout == before_refs
