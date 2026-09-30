"""Parent L1 adjudication: real Git/DB; native host only, NOT macOS proof."""
from pathlib import Path
import sqlite3
import subprocess
import unicodedata

import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kdc
from hermes_cli import kanban_db_workspace as ws


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / 'board.db')
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kdc._migrate_add_optional_columns(conn)
    conn.commit()
    yield conn
    conn.close()


def prepare(db, root):
    root.mkdir()
    git(root, 'init', '-q', '-b', 'main')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'offline fixture')
    (root / 'basis').write_text('original\n')
    git(root, 'add', 'basis')
    git(root, 'commit', '-qm', 'fixture')
    base, tree = git(root, 'rev-parse', 'HEAD'), git(root, 'rev-parse', 'HEAD^{tree}')
    tid = kb.create_task(db, title='Real capture', initial_status='blocked',
                         workspace_kind='worktree', workspace_path=str(root))
    target = root / '.worktrees' / tid
    with kb.write_txn(db):
        db.execute('UPDATE tasks SET workspace_path=? WHERE id=?', (str(target), tid))
    plan = ws.bind_workspace_plan(db, task_id=tid, workspace_root=root,
                                 workspace_path=target, base_commit=base, base_tree=tree)
    task = kb.get_task(db, tid)
    return task, target, plan


@pytest.mark.parametrize('form', ['NFC', 'NFD'])
def test_native_unicode_resolve_capture_and_git_toplevel(db, tmp_path, form):
    root = tmp_path / unicodedata.normalize(form, 'project Persönlich')
    task, target, plan = prepare(db, root)
    assert ws.resolve_workspace(task, conn=db) == target
    assert ws.resolve_workspace(task, conn=db) == target
    # Path parsing normalizes a trailing slash before the helper compares Paths.
    assert ws._git_toplevel(Path(str(target) + '/')) == target
    result = ws.capture_workspace_authority(db, task_id=task.id, workspace=target,
                                            source='fixture', captured_by='fixture')
    assert result is not None
    row = db.execute('SELECT * FROM task_workspace_authority WHERE task_id=?', (task.id,)).fetchone()
    assert row['workspace_root'] == plan['workspace_root']
    assert row['base_commit'] == plan['base_commit']
    assert row['base_tree'] == plan['base_tree']


@pytest.mark.parametrize('defect', ['tilde', 'leading_space', 'trailing_slash', 'dot_segment'])
def test_noncanonical_root_refuses_before_git_even_with_matching_digest(db, tmp_path, monkeypatch, defect):
    root = tmp_path / 'project with spaces'
    task, target, plan = prepare(db, root)
    assert ws._validate_plan_for_materialization(plan, task, expected_repo_root=root)[0] == root
    altered = dict(plan)
    altered['workspace_root'] = {'tilde': '~/project', 'leading_space': ' ' + str(root),
                                'trailing_slash': str(root) + '/', 'dot_segment': str(root) + '/.'}[defect]
    altered['basis_digest'] = ws.basis_digest(
        version=altered['plan_version'], task_id=task.id,
        workspace_root=altered['workspace_root'], workspace_path=altered['workspace_path'],
        base_commit=altered['base_commit'], base_tree=altered['base_tree'])
    calls = []
    original = ws._git_common_dir

    def observe(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(ws, '_git_common_dir', observe)
    before = list(db.iterdump())
    with pytest.raises(ValueError, match='literal canonical absolute spelling'):
        ws._validate_plan_for_materialization(altered, task, expected_repo_root=root)
    assert calls == []
    assert list(db.iterdump()) == before
    assert not target.exists()


@pytest.mark.parametrize('kind', ['own', 'own_backlink_alias', 'copied_pointer', 'symlink_pointer', 'forged_backlink_alias', 'forged_direct_backlink'])
def test_real_backlink_identity_and_unique_registration(db, tmp_path, kind):
    root = tmp_path / 'project'
    task, target, plan = prepare(db, root)
    branch = 'wt/' + task.id
    foreign = kind in {'copied_pointer', 'symlink_pointer', 'forged_backlink_alias', 'forged_direct_backlink'}
    git(root, 'worktree', 'add', '-qb', 'other-target-branch' if foreign else branch,
        str(target), plan['base_commit'])
    own_gitdir = ws._git_dir(target)
    assert ws._is_registered_worktree(root, target) is True
    if foreign:
        sibling = root / '.worktrees' / 'sibling'
        git(root, 'worktree', 'add', '-qb', branch, str(sibling), plan['base_commit'])
        sibling_gitdir = ws._git_dir(sibling)
        if kind == 'symlink_pointer':
            (target / '.git').unlink()
            (target / '.git').symlink_to(sibling / '.git')
        else:
            (target / '.git').write_bytes((sibling / '.git').read_bytes())
    if kind in {'own_backlink_alias', 'forged_backlink_alias'}:
        alias1, alias2 = root / 'backlink-one', root / 'backlink-two'
        alias1.symlink_to(alias2)
        alias2.symlink_to(target / '.git')
        metadata = sibling_gitdir if foreign else own_gitdir
        (metadata / 'gitdir').write_text(str(alias1) + '\n')
        assert alias1.resolve() == target / '.git'
    if kind == 'forged_direct_backlink':
        (sibling_gitdir / 'gitdir').write_text(str(target / '.git') + '\n')
    listing = git(root, 'worktree', 'list', '--porcelain')
    occurrences = sum(line == 'worktree ' + str(target) for line in listing.splitlines())
    assert occurrences == {'own_backlink_alias': 0, 'forged_direct_backlink': 2}.get(kind, 1)
    before, refs = list(db.iterdump()), git(root, 'show-ref')
    if foreign or kind == 'own_backlink_alias':
        with pytest.raises(ValueError, match='not a registered worktree'):
            ws.resolve_workspace(task, conn=db)
        assert ws._is_registered_worktree(root, target) is False
    else:
        assert ws.resolve_workspace(task, conn=db) == target
        assert ws._is_registered_worktree(root, target) is True
    assert list(db.iterdump()) == before
    assert git(root, 'show-ref') == refs
    assert git(root, 'worktree', 'list', '--porcelain') == listing
