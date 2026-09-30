"""L1 real Core/Git: activation validates frozen plans; replay cannot heal history."""
from pathlib import Path
import sqlite3
import subprocess

import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kdc
from hermes_cli import kanban_db_workspace as ws
from hermes_cli import kanban_swarm as swarm


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def commit(repo, text):
    (repo / 'basis.txt').write_text(text)
    git(repo, 'add', 'basis.txt')
    git(repo, 'commit', '-qm', 'offline fixture')
    return git(repo, 'rev-parse', 'HEAD'), git(repo, 'rev-parse', 'HEAD^{tree}')


def snapshot(conn):
    return list(conn.iterdump())


@pytest.fixture
def context(tmp_path, monkeypatch):
    repo = tmp_path / 'project'
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'main')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    git(repo, 'config', 'user.name', 'offline-fixture')
    base, tree = commit(repo, 'frozen\n')
    conn = sqlite3.connect(tmp_path / 'board.db')
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kdc._migrate_add_optional_columns(conn)
    conn.commit()
    hooks = []
    monkeypatch.setattr(kb, '_fire_kanban_lifecycle_hook', lambda *a, **kw: hooks.append((a, kw)))
    for module in (kb, kdc, ws, swarm):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd().resolve())
    yield conn, repo, base, tree, hooks
    conn.close()


def create(conn, repo, base=None, tree=None):
    return swarm.create_swarm(
        conn, goal='Validate activation basis',
        workers=[swarm.SwarmWorkerSpec(profile='fixture', title='Produce', body='offline')],
        verifier_assignee='fixture-review', synthesizer_assignee='fixture-synth',
        workspace_kind='worktree', workspace_path=str(repo), per_task_worktrees=True,
        git_base_commit=base, git_base_tree=tree, idempotency_key='activation-probe')


@pytest.mark.parametrize('role', ['root', 'worker', 'verifier', 'synthesizer'])
def test_active_missing_plan_refuses_without_recertifying(context, role):
    conn, repo, base, tree, hooks = context
    created = create(conn, repo, base, tree)
    ids = dict(root=created.root_id, worker=created.worker_ids[0],
               verifier=created.verifier_id, synthesizer=created.synthesizer_id)
    conn.execute('DELETE FROM task_workspace_plans WHERE task_id=?', (ids[role],))
    conn.commit()
    before, prior_hooks = snapshot(conn), list(hooks)
    with pytest.raises(ValueError, match='plan'):
        create(conn, repo, base, tree)
    assert snapshot(conn) == before
    assert hooks == prior_hooks


@pytest.mark.parametrize('defect', ['wrong_tree', 'symbolic', 'short', 'padded', 'missing_commit'])
def test_invalid_frozen_basis_refuses_before_activation(context, monkeypatch, defect):
    conn, repo, base, tree, hooks = context
    _, other_tree = commit(repo, 'later\n')
    bad_base, bad_tree = base, tree
    if defect == 'wrong_tree':
        bad_tree = other_tree
    elif defect == 'symbolic':
        bad_base = 'HEAD'
        bad_tree = other_tree
    elif defect == 'short':
        bad_base = base[:12]
    elif defect == 'padded':
        bad_base = base + ' '
    else:
        bad_base = '0' * len(base)
    activations = []
    original = swarm._activate_root_inline

    def observe(*args, **kwargs):
        activations.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(swarm, '_activate_root_inline', observe)
    before = snapshot(conn)
    with pytest.raises(ValueError):
        create(conn, repo, bad_base, bad_tree)
    assert activations == []
    assert hooks == []
    assert snapshot(conn) == before
    assert not (repo / '.worktrees').exists()


@pytest.mark.parametrize('explicit', [False, True])
def test_exact_replay_after_head_moves_preserves_frozen_basis(context, explicit):
    conn, repo, base, tree, hooks = context
    args = (base, tree) if explicit else (None, None)
    created = create(conn, repo, *args)
    commit(repo, 'later unrelated work\n')
    before, prior_hooks = snapshot(conn), list(hooks)
    assert create(conn, repo, *args) == created
    assert snapshot(conn) == before
    assert hooks == prior_hooks
    assert {tuple(r) for r in conn.execute('SELECT base_commit,base_tree FROM task_workspace_plans')} == {(base, tree)}


def test_valid_frozen_basis_exists_before_native_activation(context, monkeypatch):
    conn, repo, base, tree, hooks = context
    activations = []
    original = swarm._activate_root_inline

    def observe(db, root_id, **kwargs):
        tasks = db.execute('SELECT id FROM tasks').fetchall()
        for row in tasks:
            plan = ws.resolve_workspace_plan_from_commit(db, row['id'])
            assert plan['base_commit'] == base and plan['base_tree'] == tree
            assert plan['workspace_root'] == str(repo)
            assert plan['workspace_path'] == str(repo / '.worktrees' / row['id'])
        activations.append(root_id)
        return original(db, root_id, **kwargs)

    monkeypatch.setattr(swarm, '_activate_root_inline', observe)
    created = create(conn, repo, base, tree)
    assert activations == [created.root_id]
    assert len(hooks) == 1
    assert kb.get_task(conn, created.root_id).status == 'done'
    assert not conn.in_transaction


@pytest.mark.parametrize('kind', ['non_git', 'unborn_head'])
def test_missing_implicit_basis_refuses_before_activation(context, tmp_path, monkeypatch, kind):
    conn, _, _, _, hooks = context
    root = tmp_path / kind
    root.mkdir()
    if kind == 'unborn_head':
        git(root, 'init', '-q', '-b', 'main')
    probe = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True)
    assert probe.returncode != 0
    kb.create_task(conn, title='Unrelated history stays', initial_status='blocked')
    activations = []
    original = swarm._activate_root_inline

    def observe(*args, **kwargs):
        activations.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(swarm, '_activate_root_inline', observe)
    before = snapshot(conn)
    with pytest.raises(ValueError, match='frozen.*basis'):
        create(conn, root)
    assert activations == []
    assert hooks == []
    assert snapshot(conn) == before
    assert not (root / '.worktrees').exists()


@pytest.mark.parametrize('kind', ['dir', 'scratch'])
def test_unplanned_non_git_swarm_preserves_legacy_flow(context, tmp_path, kind):
    conn, _, _, _, hooks = context
    root = tmp_path / 'plain-directory'
    root.mkdir()
    created = swarm.create_swarm(
        conn, goal='Legacy non-Git planning',
        workers=[swarm.SwarmWorkerSpec(profile='fixture', title='Produce', body='offline')],
        verifier_assignee='fixture-review', synthesizer_assignee='fixture-synth',
        workspace_kind=kind, workspace_path=str(root), per_task_worktrees=False)
    assert kb.get_task(conn, created.root_id).status == 'done'
    assert conn.execute('SELECT COUNT(*) FROM task_workspace_plans').fetchone()[0] == 0
    assert len(hooks) == 1


def test_active_implicit_replay_does_not_require_current_head(context):
    conn, repo, base, tree, hooks = context
    created = create(conn, repo)
    git(repo, 'update-ref', '-d', 'refs/heads/main')
    assert subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True).returncode != 0
    before, prior_hooks = snapshot(conn), list(hooks)
    assert create(conn, repo) == created
    assert snapshot(conn) == before
    assert hooks == prior_hooks
    assert {tuple(r) for r in conn.execute('SELECT base_commit,base_tree FROM task_workspace_plans')} == {(base, tree)}
