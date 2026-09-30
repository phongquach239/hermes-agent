"""L1 capture anchor regressions: real Core, disposable Git and SQLite only."""
import sqlite3
import subprocess
from pathlib import Path

import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kdc
from hermes_cli import kanban_db_workspace as ws
from hermes_cli import kanban_swarm as swarm


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def anchor(tmp_path):
    repo = tmp_path / 'selected project'
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'main')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    git(repo, 'config', 'user.name', 'offline-fixture')
    (repo / 'basis.txt').write_text('frozen\n')
    git(repo, 'add', 'basis.txt')
    git(repo, 'commit', '-qm', 'frozen')
    conn = sqlite3.connect(tmp_path / 'fixture.db')
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kdc._migrate_add_optional_columns(conn)
    conn.commit()
    try:
        yield conn, repo, git(repo, 'rev-parse', 'HEAD'), git(repo, 'rev-parse', 'HEAD^{tree}')
    finally:
        conn.close()


def snapshot(conn):
    return {table: [tuple(r) for r in conn.execute(f'SELECT * FROM {table} ORDER BY rowid')]
            for table in ('tasks', 'task_runs', 'task_events', 'task_workspace_authority',
                          'task_workspace_plans')}


def planned(conn, selected, base, tree):
    created = swarm.create_swarm(
        conn, goal='capture anchor regression',
        workers=[swarm.SwarmWorkerSpec(profile='fixture', title='fixture', body='offline')],
        verifier_assignee='fixture-review', synthesizer_assignee='fixture-synth',
        workspace_kind='worktree', workspace_path=str(selected), per_task_worktrees=True,
        git_base_commit=base, git_base_tree=tree, idempotency_key='capture-anchor')
    task = kb.claim_task(conn, created.worker_ids[0], ttl_seconds=300)
    assert task is not None
    checkout = ws.resolve_workspace(task, conn=conn)
    return task, checkout


@pytest.mark.parametrize('foreign', [False, True])
def test_unplanned_capture_digest_binds_selected_project(anchor, tmp_path, foreign):
    conn, repo, base, tree = anchor
    conn.execute("INSERT INTO tasks(id,title,status,workspace_kind,created_at) "
                 "VALUES ('same-task','fixture','blocked','worktree',1)")
    conn.commit()
    first = ws.capture_workspace_authority(
        conn, task_id='same-task', workspace=repo, source='fixture', captured_by='fixture')
    selected = repo
    if foreign:
        selected = tmp_path / 'foreign project'
        git(tmp_path, 'clone', '-q', str(repo), str(selected))
    second = ws.capture_workspace_authority(
        conn, task_id='same-task', workspace=selected, source='fixture', captured_by='fixture')
    assert first['base_commit'] == second['base_commit'] == base
    assert first['base_tree'] == second['base_tree'] == tree
    assert first['workspace_root'] == str(repo)
    assert second['workspace_root'] == str(selected)
    assert (first['authority_sha256'] != second['authority_sha256']) is foreign


@pytest.mark.parametrize('linked', [False, True])
def test_fresh_capture_keeps_selected_plan_anchor(anchor, tmp_path, linked):
    conn, selected, base, tree = anchor
    if linked:
        primary = selected
        selected = tmp_path / 'linked selected project'
        git(primary, 'worktree', 'add', '-qb', 'selected', str(selected), base)
    task, checkout = planned(conn, selected, base, tree)
    plan_before = dict(conn.execute('SELECT * FROM task_workspace_plans WHERE task_id=?',
                                   (task.id,)).fetchone())
    ws.set_workspace_path(conn, task.id, checkout)
    captured = dict(conn.execute('SELECT * FROM task_workspace_authority WHERE task_id=?',
                                (task.id,)).fetchone())
    started = dict(conn.execute('SELECT * FROM task_runs WHERE id=?',
                               (task.current_run_id,)).fetchone())
    assert Path(captured['workspace_root']) == selected
    assert captured['base_commit'] == started['workspace_start_commit'] == base
    assert captured['base_tree'] == started['workspace_start_tree'] == tree
    assert captured['authority_sha256'] == started['workspace_authority_sha256']
    assert dict(conn.execute('SELECT * FROM task_workspace_plans WHERE task_id=?',
                            (task.id,)).fetchone()) == plan_before


@pytest.mark.parametrize('invalid', ['foreign_checkout', 'changed_head'])
def test_planned_capture_refuses_wrong_checkout_without_writes(anchor, tmp_path, invalid):
    conn, selected, base, tree = anchor
    task, checkout = planned(conn, selected, base, tree)
    ws.set_workspace_path(conn, task.id, checkout)
    if invalid == 'foreign_checkout':
        wrong = tmp_path / 'foreign checkout'
        git(tmp_path, 'clone', '-q', str(checkout), str(wrong))
        assert git(wrong, 'rev-parse', 'HEAD') == base
    else:
        wrong = checkout
        (wrong / 'basis.txt').write_text('drift after planning\n')
        git(wrong, 'add', 'basis.txt')
        git(wrong, 'commit', '-qm', 'drift')
        assert git(wrong, 'rev-parse', 'HEAD') != base
    before = snapshot(conn)
    with pytest.raises(ValueError):
        ws.capture_workspace_authority(
            conn, task_id=task.id, workspace=wrong,
            source='fixture', captured_by='fixture')
    assert snapshot(conn) == before
