"""GitSpawn / GHSA-7x36-8jrh-v4pw: repository-named filter cases for the hm-loop carry.

The carry cherry-picks upstream's ``noninteractive_repo_git_env`` series (c77a2dda9f ..
b7fd5527dc) but leaves ``test_gitspawn_config_injection.py`` at its pre-series content, so a
``hermes update`` merge of upstream into the carry branch stays conflict-free. These are that
series' kanban ``_git``, recovery-hint, completion-probe, session-snapshot and subagent
worktree-add cases, copied from upstream b7fd5527dc without the assertions on sites the carry
does not harden: worktree_ops (1ff01f4cf7, 13ea517514, f3767103e8 not carried) and worktree_gc
(kept at live behaviour: its blanked system config drops Git for Windows' core.autocrlf).

Delete this file once the carry is rebased onto an upstream that contains b7fd5527dc.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli._subprocess_compat import FILTER_DISCOVERY_FAILED, noninteractive_repo_git_env
from tests.security.test_gitspawn_config_injection import (  # noqa: F401  (fixture used by name)
    _fired,
    malicious_repo,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


_CLEAN_GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                  "GIT_CONFIG_NOSYSTEM": "1"}


def test_index_reading_probes_and_kanban_gc_git_are_safe(malicious_repo, tmp_path):
    """``status`` / ``ls-files`` / ``worktree add`` read the index, which runs ``core.fsmonitor``;
    ``worktree add`` also runs the repository's hooks, and ``branch -D`` its reference-transaction
    hook. Recovery hint, completion probe, kanban worktree and worktree-gc ``status``.

    hm-loop carry: upstream also covers the worktree_ops reclaim/branch-deletion/ssh sites here; those
    fixes (1ff01f4cf7, 13ea517514, f3767103e8) are not carried, so their assertions are omitted."""
    from hermes_cli import kanban_db_workspace as kw
    from tools.async_delegation_recovery_hints import git_state_hint
    from tui_gateway import server
    repo, marker = malicious_repo
    assert git_state_hint(str(repo)) is not None
    assert "README" in list(server._git_repo_files(str(repo)))
    kw._ensure_git_worktree(repo, tmp_path / "wt2", "safe2")
    assert (tmp_path / "wt2" / "README").exists()
    assert _fired(marker) == []


def _make_filter_repo(tmp: Path, attrs: str, config, marker: str) -> Path:
    """Committed repo whose ``.gitattributes`` is *attrs*; ``config(repo, marker)`` is appended to ``.git/config``."""
    repo = tmp / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=_CLEAN_GIT_ENV)
    (repo / "README").write_text("hi\n")
    (repo / ".gitattributes").write_text(attrs)
    ident = ["-c", "user.email=a@b", "-c", "user.name=a"]
    subprocess.run(["git", "-C", str(repo), *ident, "add", "."], check=True, env=_CLEAN_GIT_ENV)
    subprocess.run(["git", "-C", str(repo), *ident, "commit", "-qm", "init"], check=True, env=_CLEAN_GIT_ENV)
    with open(repo / ".git" / "config", "a") as fh:
        fh.write(config(repo, marker))
    return repo


def _evil_filter(marker: str, name: str = "evil") -> str:
    return (f'[filter "{name}"]\n\tsmudge = touch \'{marker}.smudge\'; cat\n'
            f'\tclean = touch \'{marker}.clean\'; cat\n\trequired = true\n')


def _evil_include(condition, nested: bool = False):
    def config(repo: Path, marker: str) -> str:
        (repo / ".git" / "evil.inc").write_text(_evil_filter(marker))
        target = "evil.inc"
        if nested:
            (repo / ".git" / "outer.inc").write_text("[include]\n\tpath = evil.inc\n")
            target = "outer.inc"
        return f'[includeIf "{condition(repo)}"]\n\tpath = {target}\n'
    return config


def _credential_include(repo: Path, marker: str) -> str:
    (repo / ".git" / "creds.inc").write_text("[http]\n\textraheader = x\n")
    return f'[includeIf "gitdir:{(repo / ".git").as_posix()}"]\n\tpath = creds.inc\n'


def _include_flood(repo: Path, marker: str) -> str:
    """More include targets than discovery reads on every hardened call."""
    sections = []
    for i in range(17):
        (repo / ".git" / f"inc{i}").write_text("")
        sections.append(f'[includeIf "onbranch:b{i}"]\n\tpath = inc{i}\n')
    return "".join(sections)


@pytest.mark.parametrize("attrs, config, refused", [
    pytest.param("README filter=evil\n", lambda r, m: _evil_filter(m), False, id="plain"),
    pytest.param("README filter=Evil\n",
                 lambda r, m: '[filter "evil"]\n\tsmudge = cat\n\tclean = cat\n' + _evil_filter(m, "Evil"),
                 False, id="case_collision"),
    pytest.param("README filter=evil\n", _evil_include(lambda r: "onbranch:safe"), False, id="onbranch_include"),
    pytest.param("README filter=evil\n",
                 _evil_include(lambda r: f"gitdir:{(r / '.git' / 'worktrees').as_posix()}/"), False,
                 id="gitdir_include"),
    # An include inside an include target is not walked again: refuse.
    pytest.param("README filter=evil\n", _evil_include(lambda r: "onbranch:safe", nested=True), True,
                 id="nested_include"),
    # The actions/checkout credential include: a target with no filters must not block the repo.
    pytest.param("README filter=evil\n", _credential_include, False, id="credential_include"),
    pytest.param("README filter=evil\n", _include_flood, True, id="include_flood"),
    pytest.param("README filter=evil\n",
                 lambda r, m: "".join(f'[filter "f{i}"]\n\tclean = cat\n' for i in range(300)), True,
                 id="filter_flood"),
    # Malformed config: `git config` dies (rc 128), which is neither "found" (0) nor "none" (1).
    pytest.param("README filter=evil\n", lambda r, m: _evil_filter(m) + '[filter "evil"\n', True,
                 id="broken_config"),
])
def test_repo_named_filters_never_run_from_kanban_gc_or_hints(tmp_path, attrs, config, refused):
    """A filter driver is named by ``.gitattributes``, so the fixed env pins cannot reach it:
    ``worktree add`` runs its smudge command and ``status`` its clean command. Subsection names are
    case-sensitive, so ``[filter "Evil"]`` must be neutralized next to a benign ``[filter "evil"]``.
    An ``includeIf`` target is read whatever its condition (``onbranch:`` matches the new branch,
    ``gitdir:`` the ``.git/worktrees/<name>`` dir of ``worktree add``), so its filters are neutralized
    too. Discovery that cannot be trusted refuses the git call: an include nested in an include
    target, a huge filter inventory (argv/env E2BIG), or a config git cannot parse."""
    from hermes_cli import kanban_db_workspace as kw
    from tools.async_delegation_recovery_hints import git_state_hint
    marker = (tmp_path / "FILTER").as_posix()
    repo = _make_filter_repo(tmp_path, attrs, config, marker)

    if refused:
        res = kw._git(repo, "worktree", "add", "-b", "safe", str(tmp_path / "wt"), "HEAD", timeout=30)
        assert (res.returncode, res.stderr) == (1, FILTER_DISCOVERY_FAILED)
        assert sorted(p.name for p in tmp_path.glob("FILTER.*")) == []
        return

    # git prints repo-local origins relative to the top level, so discovery from a subdirectory must agree.
    (repo / "sub").mkdir()
    assert noninteractive_repo_git_env(repo / "sub") == noninteractive_repo_git_env(repo)
    kw._ensure_git_worktree(repo, tmp_path / "wt", "safe")
    assert (tmp_path / "wt" / "README").read_text() == "hi\n"
    (repo / "README").write_text("hi\n")  # same size, new mtime: status must re-hash it
    os.utime(repo / "README", (time.time() + 60, time.time() + 60))
    assert git_state_hint(str(repo)) is not None
    # The automatic session-start snapshot (status) and a delegated subagent's worktree (checkout).
    from agent.coding_context import build_coding_workspace_block
    from tools.subagent_worktree import create_subagent_worktree
    assert "- Status:" in build_coding_workspace_block(repo)
    sub = create_subagent_worktree(str(repo), "filters")
    assert sub is not None and (Path(sub["path"]) / "README").read_text() == "hi\n"
    # hm-loop carry: upstream also checks subagent finalization, the kanban-teardown dirty probe and
    # hermes -w here; those sites (1ff01f4cf7) are not carried.
    assert sorted(p.name for p in tmp_path.glob("FILTER.*")) == []
