"""Task workspace lifecycle: scratch/dir/worktree resolution (incl. git worktree creation), post-completion cleanup with containment guards, worker tmux teardown and the first-use scratch-workspace tip.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import sqlite3
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING

from hermes_cli.worktree_ops import release_lsp_clients

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

_REMOVABLE_KINDS = ("scratch", "worktree")


def _path_key(path: Path | str | None) -> str:
    """Unicode-form-insensitive identity for a filesystem path.

    macOS hands back DECOMPOSED path strings (NFD: ``o`` + U+0308) for names the
    user typed in composed form (NFC: ``ö``) — a OneDrive/FileProvider path like
    ``OneDrive-Persönlich`` round-trips through ``git rev-parse --show-toplevel``
    as NFD while the DB row holds NFC. Raw ``Path`` equality then reports a real
    repo root as "not a repo" purely on Unicode form, so every path identity
    check here goes through this key.
    """
    return unicodedata.normalize("NFC", str(path)) if path is not None else ""

# Statuses after which a child no longer needs its parent's workspace artifacts.
_ACTIVE_CHILDREN_SQL = (
    "SELECT 1 FROM task_links l "
    "JOIN tasks t ON t.id = l.child_id "
    "WHERE l.parent_id = ? AND t.status NOT IN ('done', 'archived', 'failed', 'cancelled') "
    "LIMIT 1"
)

_WORKSPACE_ROW_SQL = "SELECT workspace_kind, workspace_path, branch_name FROM tasks WHERE id = ?"


def _git(repo_root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
    """``git -C repo_root args``; never raises on a non-zero exit."""
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
        check=False,
    )


def _has_active_children(conn: sqlite3.Connection, task_id: str) -> bool:
    return conn.execute(_ACTIVE_CHILDREN_SQL, (task_id,)).fetchone() is not None


def _managed_scratch_path_info(p: Path) -> tuple[bool, Optional[str]]:
    """Return whether *p* is managed scratch storage and the matching board."""
    try:
        p_abs = p.resolve(strict=False)
    except OSError:
        return False, None
    roots: list[tuple[Path, Optional[str]]] = []
    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        with contextlib.suppress(OSError):
            roots.append((Path(override).expanduser().resolve(strict=False), None))
    try:
        home = _kb.kanban_home()
    except OSError:
        home = None
    if home is not None:
        with contextlib.suppress(OSError):
            roots.append(((home / "kanban" / "workspaces").resolve(strict=False), _kb.DEFAULT_BOARD))
        entries: list[Path] = []
        with contextlib.suppress(OSError):
            entries = list((home / "kanban" / "boards").resolve(strict=False).iterdir())
        for entry in entries:
            with contextlib.suppress(OSError):
                if entry.is_dir():
                    roots.append(((entry / "workspaces").resolve(strict=False), entry.name))
    for root, board in roots:
        if p_abs == root:
            continue
        try:
            if p_abs.is_relative_to(root):
                return True, board
        except ValueError:
            continue
    return False, None


def _scratch_workspace(conn: sqlite3.Connection, task_id: str) -> Optional[Path]:
    """Expanded ``workspace_path`` when the task uses a scratch workspace, else ``None``."""
    row = conn.execute(
        "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row or row["workspace_kind"] != "scratch" or not row["workspace_path"]:
        return None
    return Path(row["workspace_path"]).expanduser()


def _is_managed_scratch_path(p: Path) -> bool:
    """True iff *p* is a STRICT descendant of a kanban-managed ``workspaces/``
    root (``HERMES_KANBAN_WORKSPACES_ROOT``, ``<kanban_home>/kanban/workspaces``,
    or ``<kanban_home>/kanban/boards/<slug>/workspaces``). A path equal to a
    root is not managed (deleting it would wipe every task's scratch dir);
    ``<kanban_home>/kanban``, ``.../logs`` and ``.../boards/<slug>`` hold
    Hermes' own DB and metadata. :func:`_cleanup_workspace` refuses
    ``rmtree`` outside managed storage — a board ``default_workdir`` on a real
    source tree paired with ``workspace_kind='scratch'`` would otherwise make
    task completion delete user data.

    See #28818.
    """
    return _managed_scratch_path_info(p)[0]


def _cleanup_workspace(conn: sqlite3.Connection, task_id: str) -> None:
    """Remove a task's scratch workspace dir and kill its stale tmux session.
    Called from :func:`complete_task` after the transaction commits; best-effort
    so cleanup never blocks completion. ``scratch`` is removed; ``worktree``
    only when provably free of work (clean tree, every commit reachable from a
    remote-tracking ref); ``dir`` is intentionally preserved."""
    try:
        row = conn.execute(_WORKSPACE_ROW_SQL, (task_id,)).fetchone()
        if not row:
            return
        kind: Optional[str] = row["workspace_kind"]
        path: Optional[str] = row["workspace_path"]
        if kind not in _REMOVABLE_KINDS or not path:
            # Not removable itself, but completing may still unblock a deferred
            # parent scratch cleanup (e.g. a 'dir' child of a scratch parent).
            # See #33774.
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        # Defer while any child is not yet terminal so it can still read
        # handoff artifacts from this workspace.
        if _has_active_children(conn, task_id):
            _kb._log.debug(
                "Deferring %s workspace cleanup for task %s: "
                "active children still need workspace at %s",
                kind, task_id, path,
            )
            return
        # Kill the (dead) tmux worker session BEFORE removing a worktree so a
        # lingering worker never has its cwd deleted from under it.
        if kind == "worktree":
            _cleanup_worker_tmux(conn, task_id)
            _cleanup_worktree_workspace(task_id, path, row["branch_name"])
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        wp = Path(path)
        if wp.is_dir():
            # Containment guard: a board's ``default_workdir`` can pair
            # ``workspace_kind='scratch'`` with a user path pointing at a real
            # source tree; without this, completion would rmtree the user's data.
            # See #28818.
            if _is_managed_scratch_path(wp):
                release_lsp_clients(str(wp))
                shutil.rmtree(wp, ignore_errors=True)
                _kb._log.debug("Removed scratch workspace: %s", wp)
            else:
                _kb._log.warning(
                    "Refusing to remove out-of-scratch workspace for task %s: %s "
                    "(workspace_kind='scratch' but path is outside any "
                    "kanban-managed workspaces root)",
                    task_id, wp,
                )
        # Kill the owning worker's tmux session if it is now dead, then let any
        # parent whose children are all done run its deferred cleanup.
        _cleanup_worker_tmux(conn, task_id)
        # After cleaning up this task's workspace, check if any parent tasks now have all children done —
        # their deferred cleanup can proceed (#33774).
        _try_cleanup_parent_workspaces(conn, task_id)
    except Exception:
        pass  # best-effort — never block completion


def _cleanup_worktree_workspace(
    task_id: str, path: str, branch_name: Optional[str] = None
) -> None:
    """Remove a finished task's linked git worktree when it holds no work.
    Mirrors the CLI startup pruner (``cli._prune_stale_worktrees``): removal
    requires a clean tree AND every commit reachable from a remote-tracking
    ref; any doubt (dirty, unpushed, unresolvable repo, failing git) preserves
    it. The auto-generated ``wt/<task-id>`` branch is deleted with it; custom
    branches are kept. Best-effort."""
    try:
        from hermes_cli.worktree_ops import _worktree_has_unpushed_commits, _worktree_is_dirty
    except Exception:
        return  # CLI safety predicates unavailable — preserve
    try:
        wp = Path(path).expanduser()
        if not wp.is_dir():
            return
        common = _git_common_dir(wp)
        if common is None or common.name != ".git":
            return  # not a linked worktree of a normal repo — never guess
        repo_root = common.parent
        if _path_key(wp.resolve(strict=False)) == _path_key(repo_root.resolve(strict=False)):
            return  # never remove the main checkout
        if _worktree_is_dirty(str(wp)) or _worktree_has_unpushed_commits(str(wp)):
            _kb._log.info(
                "Preserving worktree for task %s: dirty or unpushed work at %s",
                task_id, wp,
            )
            return
        # Windows cannot delete a directory while this process has its current
        # directory inside it. Completed workers normally run from their own
        # linked worktree, so move this process back to the main checkout
        # before asking Git to remove the worktree.
        worktree_path = wp.resolve(strict=False)
        try:
            cwd = Path.cwd().resolve(strict=False)
        except OSError:
            # cwd was already deleted (a scratch-kind child's own workspace is
            # rmtree'd before this deferred parent cleanup runs, #33774). A
            # dead cwd cannot hold the worktree open, so leaving it is safe.
            cwd = None
        if cwd is None or cwd == worktree_path or cwd.is_relative_to(worktree_path):
            try:
                os.chdir(repo_root)
            except OSError as exc:
                _kb._log.warning(
                    "Preserving worktree for task %s: cannot leave %s for %s: %s",
                    task_id, cwd or "<deleted cwd>", repo_root, exc,
                )
                return
        # No --force: git's own dirty guard re-verifies at removal time, so if
        # the tree became dirty since our check (TOCTOU) removal fails safe.
        release_lsp_clients(str(worktree_path))
        result = _git(repo_root, "worktree", "remove", str(wp), timeout=60)
        if result.returncode != 0:
            # Windows can retain a directory handle briefly after cwd changes.
            # Retry once without --force; Git still enforces its dirty guard.
            time.sleep(0.1)
            result = _git(repo_root, "worktree", "remove", str(wp), timeout=60)
        if result.returncode != 0:
            _kb._log.warning(
                "git worktree remove failed for task %s at %s: %s",
                task_id, wp, (result.stderr or result.stdout or "").strip(),
            )
            return
        _kb._log.debug("Removed worktree workspace: %s", wp)
        branch = (branch_name or "").strip() or f"wt/{task_id}"
        if branch.startswith("wt/"):
            _git(repo_root, "branch", "-D", branch, timeout=30)
    except Exception:
        pass  # best-effort — never block completion


def _try_cleanup_parent_workspaces(conn: sqlite3.Connection, task_id: str) -> None:
    """Run the deferred cleanup of any parent scratch/worktree workspace whose
    children are now all done/archived/failed/cancelled (called after each
    child completes).

    See #33774.
    """
    try:
        parents = conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        for (parent_id,) in parents:
            row = conn.execute(_WORKSPACE_ROW_SQL, (parent_id,)).fetchone()
            if (
                not row
                or row["workspace_kind"] not in _REMOVABLE_KINDS
                or not row["workspace_path"]
                or _has_active_children(conn, parent_id)
            ):
                continue
            if row["workspace_kind"] == "worktree":
                _cleanup_worktree_workspace(parent_id, row["workspace_path"], row["branch_name"])
                continue
            wp = Path(row["workspace_path"])
            if wp.is_dir() and _is_managed_scratch_path(wp):
                release_lsp_clients(str(wp))
                shutil.rmtree(wp, ignore_errors=True)
                _kb._log.debug("Deferred cleanup: removed parent %s scratch workspace: %s", parent_id, wp)
    except Exception:
        pass  # best-effort


def _cleanup_worker_tmux(conn: sqlite3.Connection, task_id: str) -> None:
    """Kill the tmux session associated with a task's assignee, if dead."""
    try:
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row or not row["assignee"]:
            return
        # Workers named swarm1-12 use tmux sessions named swarm-swarm1 etc.
        session = f"swarm-{row['assignee']}"
        out = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
        )
        if out.stdout.strip() == "1":
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=5)
            _kb._log.debug("Killed stale tmux session: %s", session)
    except Exception:
        pass  # best-effort — never block completion


_SCRATCH_TIP_SENTINEL_NAME = ".scratch_tip_shown"


_SCRATCH_TIP_MESSAGE = (
    "scratch workspaces are ephemeral — they're deleted when the task "
    "completes. Use --workspace worktree: (git worktree) or "
    "--workspace dir:/abs/path (existing dir) to preserve worker output."
)


def _scratch_tip_sentinel_path() -> Path:
    """Path to the per-install scratch-workspace-tip sentinel file."""
    return _kb.kanban_home() / _SCRATCH_TIP_SENTINEL_NAME


def _scratch_tip_shown() -> bool:
    """True iff the scratch-workspace tip was already emitted on this install.
    Best-effort — any error re-emits, the safer failure mode for a help message."""
    try:
        return _scratch_tip_sentinel_path().exists()
    except OSError:
        return False


def _mark_scratch_tip_shown() -> None:
    """Touch the sentinel so future scratch workspaces stay silent. Best-effort:
    a failure means the tip may appear once more, preferable to crashing dispatch."""
    try:
        path = _scratch_tip_sentinel_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def _maybe_emit_scratch_tip(
    conn: sqlite3.Connection,
    task_id: str,
    workspace_kind: Optional[str],
) -> None:
    """Emit the first-use scratch-workspace tip once per install, right after a
    scratch workspace is materialized. No-op for ``worktree``/``dir`` (preserved
    by design) and once the sentinel exists."""
    if (workspace_kind or "scratch") != "scratch" or _scratch_tip_shown():
        return
    try:
        _kb._log.warning("kanban: %s (task %s)", _SCRATCH_TIP_MESSAGE, task_id)
        with _kb.write_txn(conn):
            _kb._append_event(
                conn, task_id, "tip_scratch_workspace",
                {"message": _SCRATCH_TIP_MESSAGE},
            )
    except Exception:
        # Best-effort — never block the spawn loop over a help message.
        pass
    finally:
        _mark_scratch_tip_shown()


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def _git_toplevel(path: Path) -> Optional[Path]:
    """Return the git toplevel containing ``path``, or ``None`` if not in a repo."""
    out = _kb._git_out(path, "rev-parse", "--show-toplevel")
    if out is None:
        return None
    try:
        return Path(out).expanduser().resolve()
    except Exception:
        return Path(out).expanduser()


def _git_branch_exists(repo_root: Path, branch_name: str) -> bool:
    try:
        result = _git(repo_root, "show-ref", "--verify", f"refs/heads/{branch_name}", timeout=30)
    except Exception:
        return False
    return result.returncode == 0


def _git_abs_path(path: Path, flag: str) -> Optional[Path]:
    out = _kb._git_out(path, "rev-parse", "--path-format=absolute", flag)
    return Path(out).expanduser().resolve(strict=False) if out else None


def _git_common_dir(path: Path) -> Optional[Path]:
    return _git_abs_path(path, "--git-common-dir")


def _git_dir(path: Path) -> Optional[Path]:
    return _git_abs_path(path, "--git-dir")


def _git_current_branch(path: Path) -> Optional[str]:
    return _kb._git_out(path, "branch", "--show-current")


def _is_linked_worktree_checkout(path: Path) -> bool:
    git_dir = _git_dir(path)
    common_dir = _git_common_dir(path)
    return git_dir is not None and common_dir is not None and git_dir != common_dir


def _is_registered_worktree(repo_root: Path, target: Path) -> bool:
    """Require one registered checkout and its own unaliased Git backlink."""
    try:
        result = _git(
            repo_root, "worktree", "list", "--porcelain", "-z", timeout=10,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    target_key = _path_key(target.resolve(strict=False))
    # Git derives each registration path from its metadata's backlink.
    # A forged second backlink must not let one record vouch for another.
    registrations = [record.split("\0", 1)[0] for record in result.stdout.split("\0\0") if record]
    matches = [entry for entry in registrations if entry.startswith("worktree ")
               and _path_key(Path(entry[len("worktree "):]).expanduser()) == target_key]
    if len(matches) != 1:
        return False
    try:
        actual_top = _git_toplevel(target)
        git_dir = _git_dir(target)
        common_dir = _git_common_dir(target)
        if (actual_top != target or git_dir is None or common_dir is None
                or common_dir != _git_common_dir(repo_root)):
            return False
        if git_dir == common_dir:
            return True
        raw_backlink = (git_dir / "gitdir").read_text(encoding="utf-8").removesuffix("\n")
        backlink = Path(raw_backlink)
        if not backlink.is_absolute():
            backlink = git_dir / backlink
        expected = target / ".git"
        # Preserve legitimate relative Git metadata lexically, but never let
        # a symlink alias associate another registration with this checkout.
        if Path(os.path.abspath(backlink)) != expected:
            return False
        # Do not resolve the RHS: a target .git symlink to a sibling must fail.
        return backlink.resolve(strict=False) == expected
    except (OSError, ValueError):
        return False


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _repo_root_for_worktree_target(path: Path) -> Optional[Path]:
    current = _nearest_existing_path(path).resolve(strict=False)
    while True:
        repo_root = _git_toplevel(current)
        if repo_root is not None:
            return repo_root
        if current == current.parent:
            return None
        current = current.parent


def _validate_plan_for_materialization(
    plan: dict,
    task: Task,
    *,
    expected_repo_root: Path,
) -> tuple[Path, str, str]:
    """Validate a stored ``task_workspace_plans`` row before any Git or
    filesystem side effect.

    Returns ``(repo_root, base_commit, base_tree)`` when every assertion
    holds. Refuses (raises ``ValueError``) on:

    * Unsupported plan version
    * Missing/empty ``base_commit`` / ``base_tree`` (treat as refusal, NOT
      a legacy fallback to later HEAD)
    * Digest mismatch (recomputed from stored fields disagrees with stored
      ``basis_digest``) — tampered plan row; raw stored identity strings
      are fed into the digest and the path-equality checks so padding or
      whitespace around ``workspace_root`` / ``workspace_path`` is not
      silently repaired into a different byte sequence
    * ``plan.workspace_root`` does not share ``--git-common-dir`` with
      ``expected_repo_root`` — foreign-clone selection (a path in another
      clone with identical objects is still foreign)
    * ``plan.workspace_root`` is not an actual checkout top-level —
      ``git rev-parse --show-toplevel`` from it must equal the canonical
      spelling; an arbitrary nested folder inheriting a parent's git state
      is a refusal
    * ``task.workspace_path`` set and disagrees with ``plan.workspace_path``
      — task row was rewritten against a different anchor
    * The stored ``base_commit`` is not the literal full object id
      (``rev-parse --verify <base_commit>^{commit}`` must return the
      exact same string): HEAD / branch / abbreviated / symbolic refs
      refuse before any worktree-add; trees of the literal commit must
      equal ``plan.base_tree``

    The caller MUST pass an explicit ``expected_repo_root``; we never
    fall back to ``git rev-parse --show-toplevel`` from a guessed path
    because that is exactly the foreign-clone aliasing F02 forbids.
    """
    if plan.get("plan_version") != _PLAN_VERSION:
        raise ValueError(
            f"resolve_workspace: unsupported plan_version={plan.get('plan_version')!r}; "
            f"this Core resolver only honors plan_version={_PLAN_VERSION}"
        )
    # Raw stored identity — do NOT strip/expanduser before digest or
    # canonical checks; whitespace or relative segments around the
    # stored root/path MUST break the digest recomputation (a producer
    # bound the exact spelling and any other byte sequence is foreign).
    base_commit = (plan.get("base_commit") or "")
    base_tree = (plan.get("base_tree") or "")
    if not base_commit or not base_tree:
        raise ValueError(
            f"resolve_workspace: stored plan for task {task.id!r} has empty "
            f"base_commit/base_tree; refusing to materialize a worktree "
            f"without a frozen basis"
        )
    plan_root_str = (plan.get("workspace_root") or "")
    plan_path_str = (plan.get("workspace_path") or "")
    if not plan_root_str or not plan_path_str:
        raise ValueError(
            f"resolve_workspace: stored plan for task {task.id!r} has empty "
            f"workspace_root/workspace_path; refusing without a frozen anchor"
        )
    # Compare raw strings BEFORE Path parsing can discard trailing / or /.
    # Canonical absolute names may contain spaces; do not strip/expand them.
    for field, raw in (("workspace_root", plan_root_str),
                       ("workspace_path", plan_path_str)):
        try:
            parsed = Path(raw)
            canonical = str(parsed.resolve(strict=False))
        except OSError as exc:
            raise ValueError(f"resolve_workspace: cannot resolve {field}: {exc}") from exc
        if not parsed.is_absolute() or raw != canonical:
            raise ValueError(
                f"resolve_workspace: stored {field} {raw!r} is not its literal "
                f"canonical absolute spelling {canonical!r}; refusing"
            )
    # Recompute the basis digest from raw stored fields and compare against
    # what is in the row — a tampered digest, root, path, or
    # commit/tree refuses immediately, no Git worktree add. The producer
    # bound the exact byte sequence; ``strip`` here would let whitespace
    # around ``workspace_root`` repair into a different identity.
    recomputed = basis_digest(
        version=_PLAN_VERSION,
        task_id=task.id,
        workspace_root=plan_root_str,
        workspace_path=plan_path_str,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    stored_digest = (plan.get("basis_digest") or "")
    if not stored_digest or stored_digest != recomputed:
        raise ValueError(
            f"resolve_workspace: stored plan digest for task {task.id!r} does not "
            f"match recomputed digest (stored={stored_digest!r}, "
            f"recomputed={recomputed!r}); refusing without side effects"
        )
    try:
        plan_root = Path(plan_root_str).expanduser().resolve(strict=False)
    except OSError as exc:
        raise ValueError(
            f"resolve_workspace: cannot resolve plan workspace_root {plan_root_str!r}: {exc}"
        ) from exc
    # Foreign-clone guard: shared objects -- same git_common_dir. Two
    # clones of the same upstream share NO gitdir, so a path in a foreign
    # clone with identical objects still has a different ``--git-common-dir``
    # than the expected (selected) root.
    expected_common = _git_common_dir(expected_repo_root)
    plan_common = _git_common_dir(plan_root)
    if expected_common is None or plan_common is None:
        raise ValueError(
            f"resolve_workspace: plan workspace_root {plan_root_str!r} is not "
            f"inside a git repo; refusing without side effects"
        )
    if _path_key(expected_common) != _path_key(plan_common):
        raise ValueError(
            f"resolve_workspace: plan workspace_root {plan_root_str!r} "
            f"belongs to a different git project than the selected "
            f"repo {expected_repo_root!r}; refusing to create a worktree "
            f"in a foreign project"
        )
    # Checkout-top-level guard: ``plan_root`` must be an actual git
    # checkout root (primary or a genuine linked worktree), not an
    # arbitrary subdirectory inheriting a parent's git state. A plain
    # folder that happens to share ``--git-common-dir`` with the
    # selected repo still has the parent's ``--show-toplevel`` and so
    # fails this guard BEFORE we trust the foreign-root comparison alone.
    actual_toplevel = _git_toplevel(plan_root)
    if actual_toplevel is None:
        raise ValueError(
            f"resolve_workspace: plan workspace_root {plan_root_str!r} is not "
            f"inside a git repo; refusing without side effects"
        )
    if _path_key(actual_toplevel) != _path_key(plan_root):
        raise ValueError(
            f"resolve_workspace: plan workspace_root {plan_root_str!r} is "
            f"not an actual checkout top-level (git --show-toplevel resolves "
            f"to {actual_toplevel!r}); refusing without side effects"
        )
    # ``task.workspace_path`` (when set) MUST equal the canonical
    # ``<plan_root>/.worktrees/<task.id>`` from the plan. Any disagreement
    # means the task row was rewritten against a different anchor
    # AFTER activation, and we refuse rather than create the redirect.
    if task.workspace_path:
        try:
            requested_abs = Path(task.workspace_path).expanduser().resolve(strict=False)
        except OSError:
            requested_abs = Path(task.workspace_path).expanduser()
        try:
            plan_path_abs = Path(plan_path_str).expanduser().resolve(strict=False)
        except OSError:
            plan_path_abs = Path(plan_path_str).expanduser()
        if _path_key(requested_abs) != _path_key(plan_path_abs):
            raise ValueError(
                f"resolve_workspace: task {task.id!r} workspace_path "
                f"{task.workspace_path!r} disagrees with frozen plan "
                f"workspace_path {plan_path_str!r}; refusing to redirect "
                "into a foreign checkout"
            )
    # Plan content: require the frozen commit to be the literal full
    # object id. ``rev-parse --verify <REV>^{commit}`` resolves
    # symbolic/abbreviated refs to their full SHA AND confirms the
    # object type is a commit; we refuse if the resolved SHA does not
    # equal the raw stored base_commit byte-for-byte.
    verify_proc = _git(
        expected_repo_root, "rev-parse", "--verify",
        f"{base_commit}^{{commit}}", timeout=10,
    )
    if verify_proc.returncode != 0:
        raise ValueError(
            f"resolve_workspace: frozen base_commit {base_commit!r} is not "
            f"a resolvable commit object in the selected repo "
            f"{expected_repo_root!r}; refusing without side effects"
        )
    resolved_commit = verify_proc.stdout.strip()
    if resolved_commit != base_commit:
        raise ValueError(
            f"resolve_workspace: stored base_commit {base_commit!r} is "
            f"not the literal full object id (rev-parse resolves it to "
            f"{resolved_commit!r}); refusing without side effects — "
            "the frozen basis must be the exact immutable SHA, not a "
            "HEAD/branch/abbreviated reference"
        )
    actual_tree_proc = _git(
        expected_repo_root, "rev-parse", f"{resolved_commit}^{{tree}}", timeout=10
    )
    if actual_tree_proc.returncode != 0:
        raise ValueError(
            f"resolve_workspace: cannot resolve tree of frozen commit "
            f"{resolved_commit!r} in {expected_repo_root!r}; refusing"
        )
    actual_tree = actual_tree_proc.stdout.strip()
    if actual_tree != base_tree:
        raise ValueError(
            f"resolve_workspace: frozen commit {base_commit!r} tree "
            f"{actual_tree!r} does not match stored base_tree "
            f"{base_tree!r}; refusing without side effects"
        )
    return plan_root, base_commit, base_tree


def _ensure_git_worktree(
    repo_root: Path,
    target: Path,
    branch_name: str,
    *,
    base_commit: Optional[str] = None,
    expected_branch: Optional[str] = None,
    expected_tree: Optional[str] = None,
) -> None:
    """Materialize ``target`` as a linked git worktree under ``repo_root``.

    ``base_commit`` (optional, opt-in): when provided, the worktree is
    pinned to that commit. This is the F02 frozen-basis path used by the
    adapter when ``resolve_workspace`` is called with an explicit ``conn``
    AND the task has a ``task_workspace_plans`` row. Native dispatch
    wiring through this code path is still PENDING (Main will sequence
    the caller wiring after R2 hands off); existing ad-hoc callers are
    not affected because the default (``base_commit=None``) keeps the
    legacy behavior.

    Frozen-basis guarantees (when ``base_commit`` is provided):

    * If the target already exists and shares ``--git-common-dir`` with
      ``repo_root``, it is ACCEPTED only when it is a registered worktree,
      on the expected branch, and its HEAD/tree matches the frozen commit.
      A mismatch refuses (no reset, no rewrite, no ``-B``/``--force``).
    * If the branch already exists but the target does not, the branch is
      attached VERBATIM (``worktree add <target> <branch>``). We never
      ``-B`` reset the branch and never supply ``--force`` — an existing
      branch carries in-flight worker history that must be preserved.
    * If neither exists, the worktree is created at the frozen commit on
      a fresh branch (``worktree add -b <branch> <target> <commit>``).

    Without ``base_commit`` (legacy unplanned path), behavior is
    unchanged: a matching existing checkout is reused; otherwise the
    branch is reused if it exists or a fresh ``HEAD``-based worktree is
    created.
    """
    target = target.expanduser()
    repo_common = _git_common_dir(repo_root)
    if target.exists():
        target_common = _git_common_dir(target)
        if repo_common is not None and target_common is not None and _path_key(target_common) == _path_key(repo_common):
            # Existing checkout of the same project. The frozen-basis
            # path requires the canonical branch and HEAD; the legacy
            # unplanned path preserves the original reuse shortcut
            # (any branch accepted, no reset).
            if base_commit is not None:
                actual_branch = _git_current_branch(target)
                wanted_branch = expected_branch or branch_name
                if actual_branch != wanted_branch:
                    raise ValueError(
                        f"resolve_workspace: target {target} exists and is checked "
                        f"out on branch {actual_branch!r}, not the expected "
                        f"{wanted_branch!r}; refusing to repoint an existing "
                        "worktree (no -B / --force on the planned path)"
                    )
                # Query HEAD FROM the target itself. ``git -C repo_root
                # rev-parse HEAD <target>`` returns TWO lines (commit +
                # resolved path) with exit 0, which would compare unequal
                # against ``base_commit`` and refuse every valid existing
                # checkout. Run ``rev-parse HEAD`` from inside the target
                # directory so it returns exactly the frozen HEAD.
                head_proc = _git(target, "rev-parse", "HEAD", timeout=10)
                if head_proc.returncode != 0:
                    raise ValueError(
                        f"resolve_workspace: target {target} exists on branch "
                        f"{wanted_branch!r} but cannot resolve its HEAD; "
                        "refusing without side effects"
                    )
                actual_head = head_proc.stdout.strip()
                if actual_head != base_commit:
                    raise ValueError(
                        f"resolve_workspace: target {target} exists and is at "
                        f"HEAD {actual_head!r}, not the frozen base_commit "
                        f"{base_commit!r}; refusing to reset an in-flight "
                        "worktree (no -B / --force on the planned path)"
                    )
                tree_proc = _git(target, "rev-parse", "HEAD^{tree}", timeout=10)
                if tree_proc.returncode != 0:
                    raise ValueError(
                        f"resolve_workspace: target {target} exists on branch "
                        f"{wanted_branch!r} but cannot resolve its tree; "
                        "refusing without side effects"
                    )
                actual_tree = tree_proc.stdout.strip()
                if actual_tree and actual_tree != expected_tree:
                    raise ValueError(
                        f"resolve_workspace: target {target} exists and is on "
                        f"tree {actual_tree!r}, not the frozen base_tree "
                        f"{expected_tree!r}; refusing without side effects"
                    )
                # Registered-worktree guard: a plain subdirectory that
                # inherits the parent's git state, or a copied ``.git``
                # file pointing at a sibling worktree, both pass the
                # branch/HEAD/tree checks above without actually being
                # registered in ``git worktree list``. Refuse before any
                # filesystem effect — the existing check above only
                # confirms the path is INSIDE the same git project, not
                # that git recognizes it as a worktree anchor.
                if not _is_registered_worktree(repo_root, target):
                    raise ValueError(
                        f"resolve_workspace: target {target} exists but is "
                        f"not a registered worktree of the selected repo "
                        f"{repo_root!r} (plain directory or sibling "
                        f"gitdir copy); refusing to materialize against "
                        "an unregistered checkout"
                    )
            return
        if base_commit is not None:
            # A path that exists but is NOT a checkout of the selected
            # repo is a foreign target — refuse without side effect.
            raise ValueError(
                f"resolve_workspace: target {target} exists but is not a "
                f"checkout of the selected repo {repo_root!r}; refusing "
                "to redirect into a foreign path"
            )
    if base_commit is not None:
        # Frozen basis path. When the branch already exists, its tip MUST
        # equal the frozen ``base_commit``; otherwise the in-flight history
        # would be silently attached on top of a different anchor. Refuse
        # BEFORE any mkdir/worktree-add so neither ref nor target (or
        # their parents) is mutated.
        if _git_branch_exists(repo_root, branch_name):
            tip_proc = _git(
                repo_root, "rev-parse", "--verify",
                f"refs/heads/{branch_name}^{{commit}}", timeout=10,
            )
            if tip_proc.returncode != 0:
                # Fall back to rev-parse on the branch ref directly.
                tip_proc = _git(
                    repo_root, "rev-parse", "--verify",
                    f"refs/heads/{branch_name}", timeout=10,
                )
            existing_tip = tip_proc.stdout.strip() if tip_proc.returncode == 0 else ""
            if existing_tip != base_commit:
                raise ValueError(
                    f"resolve_workspace: branch {branch_name!r} already "
                    f"exists at {existing_tip!r}, not the frozen "
                    f"base_commit {base_commit!r}; refusing to attach an "
                    "existing branch whose tip does not match the frozen basis"
                )
        # All pre-mkdir validations pass; create the parent and the
        # worktree. The fresh-branch path uses ``-b <branch>`` to bind
        # the worktree to ``base_commit``; the existing-branch path uses
        # ``<branch>`` VERBATIM (``no -B / --force / reset``).
        target.parent.mkdir(parents=True, exist_ok=True)
        if _git_branch_exists(repo_root, branch_name):
            args = ["worktree", "add", str(target), branch_name]
        else:
            args = ["worktree", "add", "-b", branch_name, str(target), base_commit]
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if _git_branch_exists(repo_root, branch_name):
            args = ["worktree", "add", str(target), branch_name]
        else:
            args = ["worktree", "add", "-b", branch_name, str(target), "HEAD"]
    result = _git(repo_root, *args, timeout=60)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"git worktree add failed for {target} on branch {branch_name}: {stderr}"
        )


def _anchored_worktree(
    repo_root: Path,
    task_id: str,
    branch_name: str,
    *,
    base_commit: Optional[str] = None,
) -> tuple[Path, str]:
    """Materialize the canonical ``<repo>/.worktrees/<task-id>`` worktree.

    When ``base_commit`` is provided, the worktree is pinned to that commit
    (F02 frozen-basis path). Otherwise the legacy behavior (HEAD at creation
    time) applies.
    """
    target = repo_root / ".worktrees" / task_id
    _ensure_git_worktree(
        repo_root, target, branch_name,
        base_commit=base_commit, expected_branch=branch_name,
    )
    return target, branch_name


def _resolve_worktree_workspace(
    task: Task,
    *,
    board: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> tuple[Path, str]:
    """Resolve + materialize a linked git worktree for ``task``. With no
    ``task.workspace_path`` the anchor is the board's ``default_workdir`` so
    every worktree lands under a board-owned repo (``<repo>/.worktrees/<id>``)
    instead of the dispatcher's incidental CWD (whatever dir the gateway was
    launched from); with no anchor configured we fail loudly rather than guess.

    F02 frozen-basis adapter seam: when ``conn`` is provided AND the task
    has a ``task_workspace_plans`` row, the plan is validated end-to-end
    (version, nonempty paired commit/tree, recomputed digest, foreign-root
    guard, plan-vs-task path alignment, Git commit→tree relationship) BEFORE
    any directory creation or ``worktree add``. The selected repo is the
    ``plan.workspace_root`` (the stored project selection, including linked
    checkouts whose ``--git-common-dir`` is the same as the canonical
    project) — we never guess from a common-dir parent of the requested
    path, because that is exactly the foreign-clone aliasing F02 forbids.

    Without a plan row, the legacy behavior is preserved (no DB read for
    callers that omit ``conn``).
    """
    branch_name = (task.branch_name or "").strip() or f"wt/{task.id}"
    plan: Optional[dict] = None
    if conn is not None:
        plan = resolve_workspace_plan_from_commit(conn, task.id)

    # -------- Plan-driven path (F02 frozen basis) --------
    # When a plan row exists, the selected repo is the stored
    # ``plan.workspace_root`` (which may be a linked checkout of a larger
    # project). Validation here refuses BEFORE any mkdir / worktree add.
    if plan is not None:
        # Stored plan fields must be absolute and non-empty BEFORE we
        # attempt any canonical comparison. Anything else is a refusal,
        # never a legacy fallback.
        plan_root_str = (plan.get("workspace_root") or "")
        plan_path_str = (plan.get("workspace_path") or "")
        if not plan_root_str or not plan_path_str:
            raise ValueError(
                f"resolve_workspace: stored plan for task {task.id!r} has empty "
                f"workspace_root/workspace_path; refusing without a frozen anchor"
            )
        plan_root = Path(plan_root_str)
        plan_path = Path(plan_path_str)
        if not plan_root.is_absolute() or not plan_path.is_absolute():
            raise ValueError(
                f"resolve_workspace: stored plan for task {task.id!r} has "
                f"non-absolute workspace_root/workspace_path; refusing"
            )
        # task.workspace_path is REQUIRED and MUST equal the stored plan
        # workspace_path. An empty task workspace_path is a refusal — the
        # schema binds workspace_path at activation, and a missing field
        # is not a license to invent one from the plan.
        task_ws_str = (task.workspace_path or "")
        if not task_ws_str:
            raise ValueError(
                f"resolve_workspace: task {task.id!r} has empty workspace_path; "
                "refusing to materialize against a frozen plan without an "
                "explicit task-level workspace_path anchor"
            )
        if not Path(task_ws_str).is_absolute():
            raise ValueError(
                f"resolve_workspace: task {task.id!r} has non-absolute "
                f"workspace_path {task.workspace_path!r}; refusing"
            )
        # Reject any path manipulation that would redirect creation into
        # a foreign project. The selected root must satisfy the plan
        # against an expected repo_root. The plan row IS the expected
        # root — validate the row first so we never call git with a
        # wrong anchor.
        _validate_plan_for_materialization(
            plan, task,
            expected_repo_root=plan_root,
        )
        # Canonical-path guards on the stored plan fields. ``resolve(strict=False)``
        # is the soft form (does not require the path to exist) and is used
        # only to DETECT symlink components: if the resolved spelling differs
        # from the stored spelling, the path contains a symlink or ``..``
        # segment that would silently redirect outside the frozen project.
        # We never replace the stored spelling with the resolved one — the
        # canonical form is the exact ``<plan_root>/.worktrees/<task.id>``
        # string the producer bound at activation.
        canonical_target = plan_root / ".worktrees" / task.id
        canonical_target_str = str(canonical_target)
        if plan_path_str != canonical_target_str:
            raise ValueError(
                f"resolve_workspace: stored plan workspace_path "
                f"{plan_path_str!r} is not the canonical "
                f"{canonical_target_str!r} derived from plan_root and "
                f"task id {task.id!r}; refusing"
            )
        if task_ws_str != canonical_target_str:
            raise ValueError(
                f"resolve_workspace: task {task.id!r} workspace_path "
                f"{task_ws_str!r} does not equal the canonical "
                f"{canonical_target_str!r}; refusing to redirect into a "
                "foreign path"
            )
        try:
            plan_path_resolved = plan_path.resolve(strict=False)
            plan_root_resolved = plan_root.resolve(strict=False)
        except OSError as exc:
            raise ValueError(
                f"resolve_workspace: cannot resolve plan paths: {exc}"
            ) from exc
        if _path_key(plan_path_resolved) != _path_key(plan_path):
            raise ValueError(
                f"resolve_workspace: stored plan workspace_path "
                f"{plan_path_str!r} resolves to {plan_path_resolved!r}; "
                "symlink or traversal components would redirect outside "
                "the frozen project; refusing without side effects"
            )
        if _path_key(plan_root_resolved) != _path_key(plan_root):
            raise ValueError(
                f"resolve_workspace: stored plan workspace_root "
                f"{plan_root_str!r} resolves to {plan_root_resolved!r}; "
                "symlink or traversal components would redirect outside "
                "the frozen project; refusing without side effects"
            )
        # Validate that the target plan_path either does not exist yet,
        # or already exists as the correct checkout on the expected
        # branch at the frozen commit. _ensure_git_worktree enforces
        # both: wrong HEAD, wrong branch, or a foreign path all refuse.
        _ensure_git_worktree(
            plan_root, plan_path, branch_name,
            base_commit=plan["base_commit"], expected_branch=branch_name,
            expected_tree=plan["base_tree"],
        )
        return plan_path, branch_name

    # -------- Legacy unplanned path (no plan row) --------
    base_commit = None
    if not task.workspace_path:
        board_slug = board if board else _kb.get_current_board()
        board_default = (_kb.read_board_metadata(board_slug).get("default_workdir") or "").strip()
        if not board_default:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but no workspace_path, "
                f"and board {board_slug!r} has no default_workdir set. Set a board "
                "default workdir (a git repo) or create the task with "
                "--workspace worktree:<absolute-repo-path>."
            )
        anchor = Path(board_default).expanduser()
        if not anchor.is_absolute():
            raise ValueError(
                f"board {board_slug!r} default_workdir {board_default!r} is not "
                "absolute; use an absolute path to a git repo"
            )
        repo_root = _git_toplevel(anchor)
        if repo_root is None:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but board "
                f"{board_slug!r} default_workdir {board_default!r} is not inside a git repo"
            )
        return _anchored_worktree(repo_root, task.id, branch_name, base_commit=base_commit)

    requested = Path(task.workspace_path).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            f"task {task.id} has non-absolute worktree path "
            f"{task.workspace_path!r}; use an absolute path"
        )
    requested_resolved = requested.resolve(strict=False)

    if requested.exists() and _is_linked_worktree_checkout(requested):
        actual_branch = _git_current_branch(requested)
        if actual_branch == branch_name:
            return requested_resolved, actual_branch
        # The requested path is an existing checkout of a DIFFERENT task's
        # branch (decompose children inherit the root's workspace_path
        # verbatim, so siblings all point here). Reusing it would run this task
        # on the other task's branch — silent cross-task provenance corruption,
        # unsafe under concurrency — so fall back to our own worktree.
        fallback_root = _repo_root_for_worktree_target(requested.parent)
        if fallback_root is not None:
            fallback = fallback_root / ".worktrees" / task.id
            if _path_key(fallback.resolve(strict=False)) != _path_key(requested_resolved):
                _ensure_git_worktree(fallback_root, fallback, branch_name, base_commit=base_commit)
                return fallback.resolve(strict=False), branch_name
        # No repo to anchor a fallback on (or the occupied path IS this task's
        # own canonical worktree): keep the legacy reuse rather than fail dispatch.
        return requested_resolved, actual_branch or branch_name

    repo_root = _git_toplevel(requested)
    if repo_root is not None and _path_key(requested_resolved) == _path_key(repo_root):
        return _anchored_worktree(repo_root, task.id, branch_name, base_commit=base_commit)

    repo_root = _repo_root_for_worktree_target(requested.parent)
    if repo_root is None:
        raise ValueError(
            f"task {task.id} worktree path {task.workspace_path!r} is not inside a git repo "
            "and does not point at a git repo root"
        )
    _ensure_git_worktree(repo_root, requested, branch_name, base_commit=base_commit)
    return requested, branch_name


def resolve_workspace(
    task: Task,
    *,
    board: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Path:
    """Resolve (and create if needed) the workspace for a task.

    ``scratch``: ``<board-root>/workspaces/<id>/`` — path-stable across the
    dispatcher and every profile worker. ``dir``: ``workspace_path``, created
    if missing; MUST be absolute (relative paths would resolve against the
    dispatcher's CWD — confused-deputy traversal). ``worktree``: a linked git
    worktree; a repo-root ``workspace_path`` anchors ``<repo>/.worktrees/<id>``,
    a concrete path is created/reused, none -> the board's ``default_workdir``
    (raises if unset rather than guessing). Persist via ``set_workspace_path``.

    F02 frozen-basis adapter seam: ``conn`` is an OPTIONAL, EXPLICIT-only
    argument. When passed (and the task has a ``task_workspace_plans`` row),
    the worktree is pinned to the stored ``base_commit`` instead of later
    HEAD. The existing ad-hoc callers that omit ``conn`` keep their legacy
    behavior (no DB read, no frozen-basis binding). Native dispatch caller
    wiring through this code path is PENDING — Main will sequence it after
    R2 hands off. Do NOT add a default ``conn`` lookup here: silently
    discovering another DB is exactly the bug F02 forbids.
    """
    kind = task.workspace_kind or "scratch"
    if kind == "worktree":
        return _resolve_worktree_workspace(task, board=board, conn=conn)[0]
    if kind == "scratch" and not task.workspace_path:
        p = _kb.workspaces_root(board=board) / task.id
    elif kind == "scratch":
        # Legacy explicit-path scratch tasks get the same absolute-path guard
        # as dir: — same threat model.
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; workspace paths must be absolute"
            )
    elif kind == "dir":
        if not task.workspace_path:
            raise ValueError(f"task {task.id} has workspace_kind=dir but no workspace_path")
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; use an absolute path "
                f"(relative paths are ambiguous against the dispatcher's CWD)"
            )
    else:
        raise ValueError(f"unknown workspace_kind: {kind}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_task_column(conn: sqlite3.Connection, task_id: str, column: str, value: str) -> None:
    with _kb.write_txn(conn):
        conn.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, task_id))


def set_workspace_path(conn: sqlite3.Connection, task_id: str, path: Path | str) -> None:
    """Persist ``tasks.workspace_path`` and stamp ``task_workspace_authority``
    for the same task when ``path`` is inside a git repo. The capture is
    best-effort: a non-git path leaves the authority table alone and the
    run simply gets NULL authority columns.

    The dispatcher claims BEFORE binding its workspace. Only that still-current,
    unexpired, not-yet-spawned claim may receive missing start authority. Historical,
    terminal or partially populated runs are never retroactively certified.
    """
    attempt = conn.execute(
        "SELECT current_run_id,claim_lock FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    _set_task_column(conn, task_id, "workspace_path", str(path))
    try:
        captured = capture_workspace_authority(
            conn,
            task_id=task_id,
            workspace=Path(path),
            source="dispatch.set_workspace_path",
            captured_by="kanban_db_workspace",
        )
    except Exception:
        # Preserve the existing best-effort capture contract, but never stamp a
        # run from a stale task-level authority row after this capture failed.
        return
    if captured is None or attempt is None or attempt["current_run_id"] is None:
        return
    now = int(time.time())
    with _kb.write_txn(conn):
        conn.execute(
            """
            UPDATE task_runs SET
                workspace_start_commit = ?,
                workspace_start_tree = ?,
                workspace_authority_sha256 = ?
            WHERE task_id = ? AND id = ? AND claim_lock = ?
              AND status = 'running' AND outcome IS NULL AND ended_at IS NULL
              AND worker_pid IS NULL AND claim_expires > ?
              AND workspace_start_commit IS NULL
              AND workspace_start_tree IS NULL
              AND workspace_authority_sha256 IS NULL
              AND EXISTS (
                  SELECT 1 FROM tasks t
                  WHERE t.id = task_runs.task_id AND t.current_run_id = task_runs.id
                    AND t.status = 'running' AND t.worker_pid IS NULL
                    AND t.claim_lock = task_runs.claim_lock AND t.claim_expires > ?
              )
            """,
            (captured["base_commit"], captured["base_tree"], captured["authority_sha256"],
             task_id, attempt["current_run_id"], attempt["claim_lock"], now, now),
        )


def set_branch_name(conn: sqlite3.Connection, task_id: str, branch_name: str) -> None:
    """Persist ``tasks.branch_name`` only.

    Branch renames are a display-metadata operation: the Git basis (commit,
    tree) hasn't changed, so the ``task_workspace_authority`` digest must NOT
    drift. The frozen plan row in ``task_workspace_plans`` likewise stays
    unchanged. Historical rows that pre-date this contract keep their original
    digest (``branch_name`` was already part of the capture payload in
    v1_workspace_authority_20260928; we simply stop rewriting it from this
    code path going forward).
    """
    _set_task_column(conn, task_id, "branch_name", branch_name)


def capture_workspace_authority(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    workspace: Path,
    branch_name: Optional[str] = None,
    source: str,
    captured_by: str,
) -> Optional[dict]:
    """Capture ``task_workspace_authority`` for ``task_id`` from the
    current state of ``workspace``.

    ``source`` identifies the capture path that produced the row
    (``dispatch.set_workspace_path``, ``claim_task``, ``resolve_workspace``);
    ``captured_by`` is the actor (profile name / caller id). Both are stored
    so a downstream consumer can tell which code path stamped the row.

    ``workspace_root`` is the selected project from the immutable task
    plan when present, not the child checkout used for fresh observation.
    Unplanned captures use the actual checkout root. The fresh v2 capture
    digest binds that project as well as the observed commit/tree; it is
    distinct from the plan basis digest. Historical run stamps are untouched.
    ``plan_version`` is NULL on fresh captures (it belongs to the immutable
    plan producer, not the observation capture).

    Returns the captured row as a dict, or ``None`` if ``workspace`` is not
    inside a git repo (the dispatcher's normal flow still continues — the
    run simply gets NULL authority columns and downstream readers skip it).
    """
    try:
        workspace_abs = workspace.resolve(strict=False)
    except OSError:
        return None
    repo_root = _git_toplevel(workspace_abs)
    if repo_root is None:
        return None
    plan = resolve_workspace_plan_from_commit(conn, task_id)
    selected_root = repo_root
    if plan is not None:
        task = _kb.get_task(conn, task_id)
        if task is None:
            raise ValueError("capture_workspace_authority: planned task is missing")
        selected_root, _, _ = _validate_plan_for_materialization(
            plan, task, expected_repo_root=Path(plan["workspace_root"]),
        )
        if (str(workspace) != plan["workspace_path"]
                or str(workspace_abs) != plan["workspace_path"]
                or repo_root != workspace_abs
                or not _is_registered_worktree(selected_root, workspace_abs)):
            raise ValueError(
                "capture_workspace_authority: checkout does not match the frozen selected project"
            )
    head_proc = _git(repo_root, "rev-parse", "HEAD", timeout=10)
    if head_proc.returncode != 0:
        return None
    base_commit = head_proc.stdout.strip()
    if not base_commit:
        return None
    tree_proc = _git(repo_root, "rev-parse", f"{base_commit}^{{tree}}", timeout=10)
    if tree_proc.returncode != 0:
        return None
    base_tree = tree_proc.stdout.strip()
    if not base_tree:
        return None
    if plan is not None and (base_commit != plan["base_commit"] or base_tree != plan["base_tree"]):
        raise ValueError("capture_workspace_authority: checkout changed from the frozen basis")
    authority_payload = (
        f"v2_selected_project_capture\n{task_id}\n{selected_root}\n{base_commit}\n{base_tree}\n"
        f"{source}\n{(branch_name or '').strip()}"
    )
    authority_sha256 = hashlib.sha256(authority_payload.encode("utf-8")).hexdigest()
    captured_at = datetime.now(timezone.utc).isoformat()
    # F02: record the SELECTED project anchor alongside the captured fields.
    # ``plan_version`` stays NULL here — fresh observation has no plan row.
    with _kb.write_txn(conn):
        conn.execute(
            """
            INSERT INTO task_workspace_authority(
                task_id, base_commit, base_tree, authority_sha256,
                source, branch_name, captured_at, captured_by,
                workspace_root, plan_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(task_id) DO UPDATE SET
                base_commit      = excluded.base_commit,
                base_tree        = excluded.base_tree,
                authority_sha256 = excluded.authority_sha256,
                source           = excluded.source,
                branch_name      = excluded.branch_name,
                captured_at      = excluded.captured_at,
                captured_by      = excluded.captured_by,
                workspace_root   = excluded.workspace_root
            """,
            (
                task_id, base_commit, base_tree, authority_sha256,
                source, (branch_name or "").strip() or None,
                captured_at, captured_by, str(selected_root),
            ),
        )
    return {
        "task_id": task_id,
        "base_commit": base_commit,
        "base_tree": base_tree,
        "authority_sha256": authority_sha256,
        "source": source,
        "branch_name": (branch_name or "").strip() or None,
        "captured_at": captured_at,
        "captured_by": captured_by,
        "workspace_root": str(selected_root),
        "plan_version": None,
    }


# ---------------------------------------------------------------------------
# F02 frozen-workspace-basis producer (Core-only).
# ---------------------------------------------------------------------------
#
# ``task_workspace_plans`` is the IMMUTABLE per-task plan row that binds the
# selected project anchor, the canonical worktree target, and the frozen
# base commit/tree BEFORE native root activation. The plan lives in its own
# table; fresh observation (``task_workspace_authority``) and the frozen plan
# are distinct producers in distinct domains — their digest inputs differ.
#
# API seam: ``bind_workspace_plan`` and ``resolve_workspace_plan_from_commit``
# take an explicit ``conn`` (not derived from filesystem or Git lookups).
# The legacy ad-hoc flow ``resolve_workspace`` / ``_resolve_worktree_workspace``
# still has no ``conn`` and is NOT modified by this producer — worktree
# materialization is the dispatcher's job, owned by R2 in the original
# candidate. Main will sequence the dispatch caller wiring after R2 hands
# off. Native dispatch integration is PENDING, not proved.

_PLAN_VERSION = 1


def basis_digest(
    *,
    version: int,
    task_id: str,
    workspace_root: str,
    workspace_path: str,
    base_commit: Optional[str],
    base_tree: Optional[str],
) -> str:
    """Deterministic basis digest over
    ``(plan_version, task_id, workspace_root, workspace_path, base_commit, base_tree)``.

    Source / branch / time are NOT digest inputs — they are provenance, not
    basis. The digest changes iff any of the six listed fields changes; the
    same fields always produce the same digest (no random nonce, no
    timestamp). NULL ``base_commit`` / ``base_tree`` (non-git or pre-capture
    state) is rendered as the literal string ``"<NULL>"`` so two unbound
    plans with the same anchor hash identically and a later bind to a real
    commit hashes differently. This is the Core workspace basis digest that
    downstream consumers compare against the captured row.
    """
    commit_rendered = base_commit if base_commit else "<NULL>"
    tree_rendered = base_tree if base_tree else "<NULL>"
    payload = (
        f"{int(version)}\n{task_id}\n{workspace_root}\n"
        f"{workspace_path}\n{commit_rendered}\n{tree_rendered}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bind_workspace_plan(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    workspace_root: Path,
    workspace_path: Path,
    base_commit: Optional[str] = None,
    base_tree: Optional[str] = None,
    plan_version: int = _PLAN_VERSION,
    bound_by: str = "kanban_db_workspace",
) -> dict:
    """Bind the immutable plan row for ``task_id``.

    Behavior:

    * If no plan row exists for ``task_id`` and the task is in a pre-activation
      status (``blocked`` / ``todo``), INSERT the row. This is the activation
      path: ``create_swarm`` runs ``bind_workspace_plan`` BEFORE flipping the
      planning root to ``done``.
    * If a plan row exists and its stored digest equals the new one, this is
      an exact replay — return the existing row unchanged (no rewrite).
    * If a plan row exists with a different digest, raise ``ValueError`` —
      conflicting active plans refuse without reset.

    ``base_commit`` / ``base_tree`` may be ``None`` when the selected project
    is not yet a git repo (no basis exists to capture). The plan row records
    NULL for those fields and the basis digest is computed over the
    ``"<NULL>"`` rendering — so a later bind to a real commit produces a
    different digest and the conflict path engages.

    The caller MUST pass an explicit ``conn``. There is no filesystem-derived
    or Git-derived fallback: the plan lives in the kanban DB the caller is
    writing to. Reading another board's DB silently is exactly the bug
    F02 forbids.

    Returns the bound row as a dict (always non-None; raises on refusal).
    """
    if plan_version != _PLAN_VERSION:
        raise ValueError(
            f"bind_workspace_plan: unknown plan_version={plan_version}; "
            f"this Core producer only writes plan_version={_PLAN_VERSION}"
        )
    try:
        root_str = str(workspace_root.resolve(strict=False))
        path_str = str(workspace_path.resolve(strict=False))
    except OSError as exc:
        raise ValueError(f"bind_workspace_plan: cannot resolve path: {exc}") from exc
    digest = basis_digest(
        version=plan_version,
        task_id=task_id,
        workspace_root=root_str,
        workspace_path=path_str,
        base_commit=base_commit,
        base_tree=base_tree,
    )
    bound_at = datetime.now(timezone.utc).isoformat()
    existing = conn.execute(
        "SELECT workspace_root, workspace_path, base_commit, base_tree, "
        "plan_version, basis_digest FROM task_workspace_plans WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if existing is not None:
        existing_digest = existing["basis_digest"]
        if existing_digest == digest:
            # Exact replay — return the row as it stands.
            return {
                "task_id": task_id,
                "workspace_root": existing["workspace_root"],
                "workspace_path": existing["workspace_path"],
                "base_commit": existing["base_commit"],
                "base_tree": existing["base_tree"],
                "plan_version": existing["plan_version"],
                "basis_digest": existing_digest,
            }
        # Conflicting active plan — refuse without reset.
        raise ValueError(
            f"bind_workspace_plan: conflicting active plan for task {task_id!r}; "
            f"existing digest={existing_digest} new digest={digest}. "
            f"Recreate the task or advance the plan version; do not silently reset."
        )
    # First bind. The migration runs CREATE TABLE IF NOT EXISTS, so this
    # INSERT is well-defined. Use allow_nested=True composition so plan
    # binding participates in any outer write_txn (e.g. create_swarm's
    # activation transaction) atomically — when the outer transaction
    # rolls back, every plan row goes with it; when it commits, every
    # plan row is durable. Direct writes outside an outer transaction
    # still use BEGIN IMMEDIATE / COMMIT, exactly like any other Core
    # write path.
    from hermes_cli.kanban_db_connect import write_txn as _write_txn
    with _write_txn(conn, allow_nested=True):
        conn.execute(
            """
            INSERT INTO task_workspace_plans(
                task_id, workspace_root, workspace_path,
                base_commit, base_tree, plan_version,
                basis_digest, bound_at, bound_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id, root_str, path_str,
                base_commit, base_tree, plan_version,
                digest, bound_at, bound_by,
            ),
        )
    return {
        "task_id": task_id,
        "workspace_root": root_str,
        "workspace_path": path_str,
        "base_commit": base_commit,
        "base_tree": base_tree,
        "plan_version": plan_version,
        "basis_digest": digest,
    }


def resolve_workspace_plan_from_commit(
    conn: sqlite3.Connection,
    task_id: str,
) -> Optional[dict]:
    """Read the plan row for ``task_id`` and return it as a dict.

    Returns ``None`` when no plan row exists. The caller MUST pass an
    explicit ``conn`` — there is no filesystem- or Git-derived fallback,
    and this helper never queries a different DB. Resolution from the
    FROZEN ``base_commit`` (not later HEAD) is the contract: callers that
    need to materialize a worktree must ``git worktree add --detach`` at
    ``base_commit`` and check out the stored branch — never rebind on
    later HEAD.
    """
    row = conn.execute(
        "SELECT task_id, workspace_root, workspace_path, base_commit, "
        "base_tree, plan_version, basis_digest, bound_at, bound_by "
        "FROM task_workspace_plans WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
