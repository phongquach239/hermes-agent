"""L1: frozen native plans through the real dispatcher, with an offline spy.

Real Core schema, swarm/claim/capture and Git. No worker process, model,
Gateway, live DB, synthetic approval or L2/L3 claim. The review lane's status
is fixture input; this file does not certify how a reviewer is selected.
"""
from pathlib import Path
import sqlite3
import subprocess

import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kdc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_db_workspace as ws
from hermes_cli import kanban_swarm as swarm


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def prepare(tmp_path, monkeypatch, anchor, lane):
    primary = tmp_path / 'primary'
    primary.mkdir()
    git(primary, 'init', '-q', '-b', 'main')
    git(primary, 'config', 'user.name', 'offline fixture')
    git(primary, 'config', 'user.email', 'fixture@example.invalid')
    (primary / 'basis.txt').write_text('frozen\n')
    git(primary, 'add', 'basis.txt')
    git(primary, 'commit', '-qm', 'fixture')
    selected = primary
    if anchor == 'linked':
        selected = tmp_path / 'selected'
        git(primary, 'worktree', 'add', '-qb', 'selected-anchor', str(selected), 'HEAD')
    conn = sqlite3.connect(tmp_path / 'board.db')
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    for _ in range(2):
        kdc._migrate_add_optional_columns(conn)
        conn.commit()
    monkeypatch.setattr(kb, '_fire_kanban_lifecycle_hook', lambda *a, **kw: None)
    monkeypatch.setattr(dispatch, '_profile_exists_fn', lambda: lambda _: True)
    created = swarm.create_swarm(
        conn, goal='Native frozen plan dispatch',
        workers=[swarm.SwarmWorkerSpec(profile='offline-fixture', title='Produce', body='offline')],
        verifier_assignee='offline-review', synthesizer_assignee='offline-synth',
        workspace_kind='worktree', workspace_path=str(selected), per_task_worktrees=True)
    task_id = created.worker_ids[0]
    if lane == 'review':
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
        conn.commit()
    plan = ws.resolve_workspace_plan_from_commit(conn, task_id)
    assert plan is not None and plan['workspace_root'] == str(selected)
    assert not Path(plan['workspace_path']).exists()
    return conn, selected, task_id, plan


def drive(conn, task_id, lane):
    calls = []

    def spawn(task, path, *args, **kwargs):
        run = conn.execute('SELECT * FROM task_runs WHERE id=?', (task.current_run_id,)).fetchone()
        capture = conn.execute('SELECT * FROM task_workspace_authority WHERE task_id=?', (task_id,)).fetchone()
        calls.append((path, dict(run), dict(capture)))
        return 0  # explicit offline callback, not a real process/worker

    row = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
    result = dispatch.DispatchResult()
    dispatch._dispatch_lane_task(
        conn, row, row['assignee'], result, lane=lane, dry_run=False,
        ttl_seconds=300, board=None, failure_limit=3, spawn_fn=spawn,
        per_profile_cap=None, per_profile_running={})
    return result, calls


def advance(root):
    (root / 'later.txt').write_text('unrelated later work\n')
    git(root, 'add', 'later.txt')
    git(root, 'commit', '-qm', 'later')


@pytest.mark.parametrize('anchor', ['primary', 'linked'])
@pytest.mark.parametrize('lane', ['ready', 'review'])
@pytest.mark.parametrize('head_state', ['unchanged', 'advanced', 'unborn'])
def test_native_dispatch_uses_selected_frozen_plan(tmp_path, monkeypatch, anchor, lane, head_state):
    conn, root, task_id, plan = prepare(tmp_path, monkeypatch, anchor, lane)
    try:
        if head_state == 'advanced':
            advance(root)
            assert git(root, 'rev-parse', 'HEAD') != plan['base_commit']
        elif head_state == 'unborn':
            ref = git(root, 'symbolic-ref', 'HEAD')
            git(root, 'update-ref', '-d', ref)
            assert subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  capture_output=True).returncode != 0
        plans_before = [tuple(r) for r in conn.execute('SELECT * FROM task_workspace_plans ORDER BY task_id')]
        result, calls = drive(conn, task_id, lane)
        assert len(calls) == 1, dict(conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone())
        assert len(result.spawned) == 1 and result.spawned[0][0] == task_id
        path, run, capture = calls[0]
        assert path == plan['workspace_path']
        assert git(path, 'rev-parse', 'HEAD') == run['workspace_start_commit'] == plan['base_commit']
        assert git(path, 'rev-parse', 'HEAD^{tree}') == run['workspace_start_tree'] == plan['base_tree']
        assert capture['workspace_root'] == str(root)
        assert capture['authority_sha256'] == run['workspace_authority_sha256']
        assert [tuple(r) for r in conn.execute('SELECT * FROM task_workspace_plans ORDER BY task_id')] == plans_before
    finally:
        conn.close()


@pytest.mark.parametrize('lane', ['ready', 'review'])
@pytest.mark.parametrize('defect', ['digest', 'path_alias', 'existing_branch'])
def test_native_dispatch_refuses_before_git_side_effects(tmp_path, monkeypatch, lane, defect):
    conn, root, task_id, plan = prepare(tmp_path, monkeypatch, 'primary', lane)
    try:
        advance(root)
        target = Path(plan['workspace_path'])
        foreign = tmp_path / 'foreign'
        foreign.mkdir()
        if defect == 'digest':
            conn.execute('UPDATE task_workspace_plans SET basis_digest=? WHERE task_id=?', ('0' * 64, task_id))
            conn.commit()
        elif defect == 'path_alias':
            (root / '.worktrees').symlink_to(foreign, target_is_directory=True)
        else:
            git(root, 'branch', 'wt/' + task_id, 'HEAD')
        plans_before = [tuple(r) for r in conn.execute('SELECT * FROM task_workspace_plans ORDER BY task_id')]
        refs_before = git(root, 'show-ref')
        registrations = git(root, 'worktree', 'list', '--porcelain')
        result, calls = drive(conn, task_id, lane)
        assert not calls and not result.spawned
        assert not target.exists(), 'invalid frozen plan must refuse before creating any checkout'
        assert list(foreign.iterdir()) == []
        assert git(root, 'show-ref') == refs_before
        assert git(root, 'worktree', 'list', '--porcelain') == registrations
        assert [tuple(r) for r in conn.execute('SELECT * FROM task_workspace_plans ORDER BY task_id')] == plans_before
        run = conn.execute('SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1', (task_id,)).fetchone()
        assert run['outcome'] == 'spawn_failed'
        assert run['workspace_start_commit'] is None and run['workspace_authority_sha256'] is None
    finally:
        conn.close()
