from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb


@contextlib.contextmanager
def _lock(acquired: bool = True):
    yield acquired


def _request(**override: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "hm-loop-managed-frontier-dispatch/v1",
        "run_id": "run-1",
        "board": "phase6-test",
        "graph_sha256": "a" * 64,
        "graph_generation": 1,
        "frontier_epoch": 7,
        "ready_nodes": ["A"],
    }
    value.update(override)
    return value


def _install_ready_graph(monkeypatch: pytest.MonkeyPatch, claim):
    graph = {"frontier_epoch": 7, "create_key": "create-1"}
    producer_a = {"task_id": "t_a", "task_kind": "producer", "status": "ready"}
    producer_b = {"task_id": "t_b", "task_kind": "producer", "status": "ready"}
    monkeypatch.setattr(kb, "_assert_managed_board_identity", lambda *_args: None)
    monkeypatch.setattr(kb, "_managed_graph_row_for_identity", lambda *_args, **_kwargs: graph)
    monkeypatch.setattr(kb, "_managed_frontier_result", lambda *_args: {"ready_nodes": ["A"]})
    monkeypatch.setattr(kb, "_managed_logical_groups", lambda *_args: ({"A": [producer_a, producer_b]}, {}))
    monkeypatch.setattr(kb, "_configured_max_in_progress_per_profile", lambda: (True, 2))
    monkeypatch.setattr(kb, "_managed_host_capacity_lock", lambda: _lock())
    monkeypatch.setattr(kb, "write_txn", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(kb, "_claim_managed_task", claim)


def test_dispatch_managed_frontier_claims_complete_ready_group_under_private_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def claim(_conn, task_id: str, **kwargs: object) -> object:
        calls.append((task_id, kwargs))
        return SimpleNamespace(current_run_id=1, assignee=f"worker-{task_id}")

    _install_ready_graph(monkeypatch, claim)
    result = kb.dispatch_managed_frontier(sqlite3.connect(":memory:"), _request())

    assert result == {"accepted_nodes": ["A"], "deferred_nodes": []}
    assert [task_id for task_id, _ in calls] == ["t_a", "t_b"]
    assert all(kwargs["_managed_capacity_lock_held"] is True for _, kwargs in calls)
    assert all(kwargs["_managed_batch"] is True for _, kwargs in calls)
    assert all(kwargs["max_in_progress_per_profile"] == 2 for _, kwargs in calls)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"frontier_epoch": 8}, kb.ManagedTaskAuthorityError),
        ({"ready_nodes": ["B"]}, kb.ManagedTaskAuthorityError),
        ({"ready_nodes": ["A", "A"]}, kb.ManagedGraphValidationError),
    ],
)
def test_dispatch_managed_frontier_rejects_stale_substituted_or_noncanonical_authority(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    error: type[Exception],
) -> None:
    _install_ready_graph(
        monkeypatch,
        lambda *_args, **_kwargs: SimpleNamespace(
            current_run_id=1, assignee="stub"
        ),
    )
    with pytest.raises(error):
        kb.dispatch_managed_frontier(sqlite3.connect(":memory:"), _request(**override))


def test_dispatch_managed_frontier_defers_a_group_if_any_private_claim_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claimed: list[str] = []

    def claim(_conn, task_id: str, **_kwargs: object) -> object | None:
        claimed.append(task_id)
        if task_id == "t_b":
            return None
        return SimpleNamespace(current_run_id=1, assignee=f"worker-{task_id}")

    _install_ready_graph(monkeypatch, claim)
    result = kb.dispatch_managed_frontier(sqlite3.connect(":memory:"), _request())

    assert result == {"accepted_nodes": [], "deferred_nodes": ["A"]}
    assert claimed == ["t_a", "t_b"]


# ---------------------------------------------------------------------------
# Real-DB tests (Phase 6 causal matrix, H-01 / H-02 / H-07)
#
# These tests do NOT replace the 5/5 mock unit tests above. They build a real
# SQLite board + real managed graph via the production code paths, and only
# stub the *boundary* hooks (lifecycle dispatcher, capacity lock) that have
# filesystem / inter-process side effects. No write_txn,
# _claim_managed_task, or _managed_profile_capacity_available is replaced.
# ---------------------------------------------------------------------------


@pytest.fixture
def managed_board(tmp_path, monkeypatch):
    """Real SQLite board with an empty kanban DB and HERMES_HOME isolated.

    Patches ``_assert_managed_board_identity`` only because it cross-checks
    ``kanban_db_path`` against the connected file. Inside a tmp_path tmp
    DB the resolver returns a different absolute file than ``sqlite3.connect``
    created — but the underlying identity check is exercised in the wider
    suite via the cross-stack tests, so stubbing the resolver to return
    ``conn`` is safe for these causal-only tests.
    """

    home = tmp_path / ".hermes"
    home.mkdir()
    db_path = home / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kanban_db_path = Path(str(db_path))
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: kanban_db_path)
    monkeypatch.setattr(
        kb, "_assert_managed_board_identity", lambda *_a, **_k: None
    )

    @contextlib.contextmanager
    def _acquired_lock():
        yield True

    monkeypatch.setattr(kb, "_managed_host_capacity_lock", _acquired_lock)
    monkeypatch.setattr(
        kb, "_configured_max_in_progress_per_profile", lambda: (True, 10_000)
    )
    kb.init_db()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _hash_request(request):
    body = {k: v for k, v in request.items() if k != "request_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _make_two_producer_logical_node(*, run_id: str, create_key: str, board: str):
    """Build a minimal valid managed graph request with logical node A and two producers."""
    nodes = [
        {
            "node_id": "n_prod_a1",
            "logical_node_id": "A",
            "task_id": "t_prod_a1",
            "title": "producer A1",
            "body": "first producer",
            "assignee": "alice",
            "task_kind": "producer",
            "task_generation": 1,
            "task_run_generation": 1,
            "mutating": False,
            "resources": [],
            "expected_artifacts": ["artifact-a1"],
            "notify_targets": [],
            "workspace_kind": "scratch",
            "workspace_path": None,
            "base_commit": None,
            "owned_paths": [],
            "owning_root_id": None,
            "owning_root_sha256": None,
        },
        {
            "node_id": "n_prod_a2",
            "logical_node_id": "A",
            "task_id": "t_prod_a2",
            "title": "producer A2",
            "body": "second producer",
            "assignee": "bob",
            "task_kind": "producer",
            "task_generation": 1,
            "task_run_generation": 1,
            "mutating": False,
            "resources": [],
            "expected_artifacts": ["artifact-a1"],
            "notify_targets": [],
            "workspace_kind": "scratch",
            "workspace_path": None,
            "base_commit": None,
            "owned_paths": [],
            "owning_root_id": None,
            "owning_root_sha256": None,
        },
        {
            "node_id": "n_ver_a",
            "logical_node_id": "A",
            "task_id": "t_ver_a",
            "title": "verifier A",
            "body": "verifier for A",
            "assignee": "carol",
            "task_kind": "verifier",
            "task_generation": 1,
            "task_run_generation": 1,
            "mutating": False,
            "resources": [],
            "expected_artifacts": [],
            "notify_targets": [],
            "workspace_kind": "scratch",
            "workspace_path": None,
            "base_commit": None,
            "owned_paths": [],
            "owning_root_id": None,
            "owning_root_sha256": None,
        },
        {
            "node_id": "n_bar_a",
            "logical_node_id": "A",
            "task_id": "t_bar_a",
            "title": "barrier A",
            "body": "barrier for A",
            "assignee": None,
            "task_kind": "barrier",
            "task_generation": 1,
            "task_run_generation": 1,
            "mutating": False,
            "resources": [],
            "expected_artifacts": [],
            "notify_targets": [],
            "workspace_kind": "scratch",
            "workspace_path": None,
            "base_commit": None,
            "owned_paths": [],
            "owning_root_id": None,
            "owning_root_sha256": None,
        },
    ]
    edges = [
        {"parent_node_id": "n_prod_a1", "child_node_id": "n_ver_a"},
        {"parent_node_id": "n_prod_a2", "child_node_id": "n_ver_a"},
        {"parent_node_id": "n_ver_a", "child_node_id": "n_bar_a"},
    ]
    request = {
        "schema": "managed_task_graph_request_v1",
        "board": board,
        "run_id": run_id,
        "create_key": create_key,
        "graph_sha256": hashlib.sha256(
            f"graph-{run_id}-{create_key}".encode()
        ).hexdigest(),
        "graph_generation": 1,
        "final_join_node_id": "n_bar_a",
        "nodes": nodes,
        "edges": edges,
    }
    request["request_sha256"] = _hash_request(request)
    return request


def _stage_and_activate(conn, request):
    kb.create_managed_task_graph(conn, request)
    return kb.activate_managed_task_graph(
        conn,
        create_key=request["create_key"],
        request_sha256=request["request_sha256"],
        graph_sha256=request["graph_sha256"],
        graph_generation=request["graph_generation"],
    )


def _frontier_request(request):
    return {
        "schema": "hm-loop-managed-frontier-dispatch/v1",
        "run_id": request["run_id"],
        "board": request["board"],
        "graph_sha256": request["graph_sha256"],
        "graph_generation": request["graph_generation"],
        "frontier_epoch": 0,
        "ready_nodes": ["A"],
    }


@pytest.fixture
def managed_home(tmp_path, monkeypatch):
    """Real multi-board Hermes home without identity or transaction stubs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _rehashed(request):
    value = copy.deepcopy(request)
    value["graph_sha256"] = hashlib.sha256(
        f"{value['run_id']}:{value['create_key']}:{len(value['nodes'])}".encode()
    ).hexdigest()
    value["request_sha256"] = _hash_request(value)
    return value


def _single_producer_request(*, run_id: str, create_key: str, board: str, suffix: str):
    value = _make_two_producer_logical_node(
        run_id=run_id,
        create_key=create_key,
        board=board,
    )
    removed_node = "n_prod_a2"
    value["nodes"] = [
        node for node in value["nodes"] if node["node_id"] != removed_node
    ]
    value["edges"] = [
        edge
        for edge in value["edges"]
        if removed_node not in (edge["parent_node_id"], edge["child_node_id"])
    ]
    node_map = {
        node["node_id"]: f"{node['node_id']}_{suffix}" for node in value["nodes"]
    }
    for node in value["nodes"]:
        node["node_id"] = node_map[node["node_id"]]
        node["task_id"] = f"{node['task_id']}_{suffix}"
        node["logical_node_id"] = f"A_{suffix}"
    for edge in value["edges"]:
        edge["parent_node_id"] = node_map[edge["parent_node_id"]]
        edge["child_node_id"] = node_map[edge["child_node_id"]]
    value["final_join_node_id"] = node_map[value["final_join_node_id"]]
    return _rehashed(value)


def _mutating_request(
    *,
    run_id: str,
    create_key: str,
    board: str,
    suffix: str,
    root: Path,
    base_commit: str,
    resources: list[str],
):
    value = _single_producer_request(
        run_id=run_id,
        create_key=create_key,
        board=board,
        suffix=suffix,
    )
    producer = next(
        node for node in value["nodes"] if node["task_kind"] == "producer"
    )
    producer.update(
        {
            "mutating": True,
            "resources": sorted(resources),
            "workspace_kind": "worktree",
            "workspace_path": str(root / ".worktrees" / producer["task_id"]),
            "base_commit": base_commit,
            "owned_paths": [f"owned/{producer['task_id']}.txt"],
            "owning_root_id": str(root),
            "owning_root_sha256": hashlib.sha256(str(root).encode()).hexdigest(),
        }
    )
    return _rehashed(value)


def _database_path(conn: sqlite3.Connection) -> Path:
    return Path(conn.execute("PRAGMA database_list").fetchone()[2])


def _open_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _claim_from_path(path: Path, task_id: str, board: str, claimer: str) -> bool:
    conn = _open_connection(path)
    try:
        return (
            kb._claim_managed_task(
                conn,
                task_id,
                board=board,
                claimer=claimer,
                max_in_progress_per_profile=10_000,
                _managed_capacity_lock_held=True,
            )
            is not None
        )
    finally:
        conn.close()


def _init_git_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "phase6-test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "phase6@example.invalid"],
        check=True,
    )
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "seed.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "seed"], check=True
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _producer_id(request) -> str:
    return next(
        node["task_id"]
        for node in request["nodes"]
        if node["task_kind"] == "producer"
    )


def test_managed_frontier_dispatch_real_db_claims_full_group_under_real_authority(
    managed_board,
) -> None:
    """Real SQLite round-trip: dispatch the group, verify durable state.

    Replaces the mock-only assertion in the 5/5 suite with a real graph
    materialize + activate + dispatch sequence that proves the CAS, the
    task_runs row, and the claimed event all exist after the outer commit.
    """
    request = _make_two_producer_logical_node(
        run_id="run-real-dispatch",
        create_key="hm-loop.graph.real.g1",
        board="default",
    )
    _stage_and_activate(managed_board, request)
    result = kb.dispatch_managed_frontier(managed_board, _frontier_request(request))
    assert result == {"accepted_nodes": ["A"], "deferred_nodes": []}
    rows = managed_board.execute(
        "SELECT id, status, claim_lock, current_run_id FROM tasks "
        "WHERE id IN (?, ?) ORDER BY id",
        ("t_prod_a1", "t_prod_a2"),
    ).fetchall()
    assert [row[1] for row in rows] == ["running", "running"]
    assert all(row[2] and row[3] is not None for row in rows)
    assert managed_board.execute(
        "SELECT COUNT(*) FROM task_runs WHERE status='running'"
    ).fetchone()[0] == 2
    assert managed_board.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind='claimed' AND task_id IN (?, ?)",
        ("t_prod_a1", "t_prod_a2"),
    ).fetchone()[0] == 2


def test_managed_frontier_partial_group_failure_leaves_no_runnable_rows(
    managed_board, monkeypatch
) -> None:
    """H-01 / H-02 causal test: second producer fails, outer rolls back.

    After the injected failure: both producers must remain on ``ready``,
    no task_runs row, no managed_resource_leases, no claimed event, and
    no lifecycle hook fired for either (because the outer transaction
    rolled back). Retry without the injected failure must claim the full
    group successfully.
    """
    request = _make_two_producer_logical_node(
        run_id="run-partial-rollback",
        create_key="hm-loop.graph.partial.g1",
        board="default",
    )
    _stage_and_activate(managed_board, request)
    fires: list[tuple[str, str, dict]] = []

    def _capture(event, task_id, **fields):
        fires.append((event, task_id, fields))

    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", _capture)

    _real_claim = kb._claim_managed_task

    def _claim_with_b_fail(conn, task_id, **kwargs):
        if task_id == "t_prod_a2":
            return None  # CAS / capacity miss equivalent to deferred
        return _real_claim(conn, task_id, **kwargs)

    monkeypatch.setattr(kb, "_claim_managed_task", _claim_with_b_fail)

    result = kb.dispatch_managed_frontier(
        managed_board, _frontier_request(request)
    )
    assert result == {"accepted_nodes": [], "deferred_nodes": ["A"]}

    states = managed_board.execute(
        "SELECT id, status, claim_lock, current_run_id FROM tasks "
        "WHERE id IN (?, ?) ORDER BY id",
        ("t_prod_a1", "t_prod_a2"),
    ).fetchall()
    assert [(row[0], row[1]) for row in states] == [
        ("t_prod_a1", "ready"),
        ("t_prod_a2", "ready"),
    ]
    assert managed_board.execute(
        "SELECT COUNT(*) FROM task_runs"
    ).fetchone()[0] == 0
    assert managed_board.execute(
        "SELECT COUNT(*) FROM managed_resource_leases WHERE released_at IS NULL"
    ).fetchone()[0] == 0
    assert managed_board.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind='claimed'"
    ).fetchone()[0] == 0
    assert fires == [], (
        f"H-01 violation: hook fired before outer commit rolled back: {fires}"
    )

    monkeypatch.setattr(kb, "_claim_managed_task", _real_claim)
    fires.clear()
    retry = kb.dispatch_managed_frontier(
        managed_board, _frontier_request(request)
    )
    assert retry == {"accepted_nodes": ["A"], "deferred_nodes": []}
    assert managed_board.execute(
        "SELECT COUNT(*) FROM tasks WHERE status='running'"
    ).fetchone()[0] == 2
    assert len(fires) == 2
    assert {fire[0] for fire in fires} == {"kanban_task_claimed"}
    assert sorted(fire[1] for fire in fires) == ["t_prod_a1", "t_prod_a2"]


def test_managed_frontier_lifecycle_hook_observes_post_commit_state(
    managed_board, monkeypatch
) -> None:
    """Success path: hooks must fire with conn.in_transaction == False and the
    durable row visible. Locks down the kanban_db.py:189-196 contract.
    """
    request = _make_two_producer_logical_node(
        run_id="run-post-commit",
        create_key="hm-loop.graph.post.g1",
        board="default",
    )
    _stage_and_activate(managed_board, request)

    observed_txn: list[bool] = []
    observed_status: list[str] = []

    def _capture(event, task_id, **fields):
        observed_txn.append(managed_board.in_transaction)
        row = managed_board.execute(
            "SELECT status FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        observed_status.append(row[0] if row else "<missing>")

    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", _capture)

    result = kb.dispatch_managed_frontier(
        managed_board, _frontier_request(request)
    )
    assert result == {"accepted_nodes": ["A"], "deferred_nodes": []}
    assert len(observed_txn) == 2
    assert all(state is False for state in observed_txn), (
        f"H-01 violation: hook ran with conn.in_transaction={observed_txn}"
    )
    assert all(status == "running" for status in observed_status)


def test_complete_managed_task_rejects_stale_authority_before_evidence(
    managed_board,
) -> None:
    """H-07: stale completion must be rejected at the authority check,
    BEFORE the evidence validation runs. The lock/generation fence is
    the primary gate; this test locks down that forged lock or stale
    generation rejects before any evidence payload is parsed.
    """
    request = _make_two_producer_logical_node(
        run_id="run-stale-fresh",
        create_key="hm-loop.graph.stale-fresh.g1",
        board="default",
    )
    _stage_and_activate(managed_board, request)
    dispatch_result = kb.dispatch_managed_frontier(
        managed_board, _frontier_request(request)
    )
    assert dispatch_result == {"accepted_nodes": ["A"], "deferred_nodes": []}

    writer_id = "t_prod_a1"
    live_authority = kb.managed_task_claim_authority(managed_board, writer_id)
    assert live_authority is not None

    # Forged lock token must be rejected at the authority fence, before
    # any evidence schema validation runs.
    with pytest.raises(kb.ManagedTaskAuthorityError):
        kb.complete_managed_task(
            managed_board,
            writer_id,
            claim_lock="forged-lock-not-equal-to-real",
            claim_generation=int(live_authority["claim_generation"]),
            result="should be rejected",
            evidence=None,
        )
    assert managed_board.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind='done' AND task_id=?",
        (writer_id,),
    ).fetchone()[0] == 0
    assert managed_board.execute(
        "SELECT status FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (writer_id,),
    ).fetchone()[0] == "running"

    # Stale generation with the correct lock must also be rejected.
    with pytest.raises(kb.ManagedTaskAuthorityError):
        kb.complete_managed_task(
            managed_board,
            writer_id,
            claim_lock=live_authority["claim_lock"],
            claim_generation=int(live_authority["claim_generation"]) - 1,
            result="stale generation",
            evidence=None,
        )

    # And live authority must still be the one and only way through the
    # fence — confirming the live authority tuple is unchanged after both
    # stale rejections above.
    re_live = kb.managed_task_claim_authority(managed_board, writer_id)
    assert re_live is not None
    assert re_live["claim_lock"] == live_authority["claim_lock"]
    assert int(re_live["claim_generation"]) == int(
        live_authority["claim_generation"]
    )


def test_managed_graph_creation_injected_failure_leaves_no_runnable_rows(
    managed_board,
) -> None:
    request = _make_two_producer_logical_node(
        run_id="run-atomic-failure",
        create_key="hm-loop.graph.atomic-failure.g1",
        board="default",
    )
    managed_board.execute(
        "CREATE TRIGGER fail_second_authority BEFORE INSERT ON managed_task_authority "
        "WHEN NEW.task_id='t_prod_a2' BEGIN SELECT RAISE(ABORT, 'injected'); END"
    )

    with pytest.raises(sqlite3.DatabaseError, match="injected"):
        kb.create_managed_task_graph(managed_board, request)

    for table in (
        "managed_task_graphs",
        "managed_graph_frontiers",
        "managed_task_authority",
        "managed_task_resources",
        "tasks",
        "task_links",
    ):
        assert managed_board.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_concurrent_equal_graph_creation_returns_one_identity(managed_home) -> None:
    request = _make_two_producer_logical_node(
        run_id="run-concurrent-create",
        create_key="hm-loop.graph.concurrent-create.g1",
        board="default",
    )
    path = kb.kanban_db_path(board="default")

    def create_once(_index: int):
        conn = _open_connection(path)
        try:
            return kb.create_managed_task_graph(conn, request)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create_once, range(2)))

    assert results[0] == results[1]
    with kb.connect(board="default") as conn:
        assert conn.execute("SELECT COUNT(*) FROM managed_task_graphs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == len(
            request["nodes"]
        )


def test_managed_graph_equal_replay_exact_and_unequal_hash_fails_closed(
    managed_home,
) -> None:
    request = _make_two_producer_logical_node(
        run_id="run-replay",
        create_key="hm-loop.graph.replay.g1",
        board="default",
    )
    with kb.connect(board="default") as conn:
        first = kb.create_managed_task_graph(conn, request)
        assert kb.create_managed_task_graph(conn, request) == first

        unequal = copy.deepcopy(request)
        unequal["nodes"][0]["title"] = "changed bytes under same create key"
        unequal["request_sha256"] = _hash_request(unequal)
        with pytest.raises(kb.ManagedGraphConflictError):
            kb.create_managed_task_graph(conn, unequal)

        assert conn.execute("SELECT COUNT(*) FROM managed_task_graphs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == len(
            request["nodes"]
        )


def test_two_connections_claim_same_managed_node_once(managed_home) -> None:
    request = _single_producer_request(
        run_id="run-node-race",
        create_key="hm-loop.graph.node-race.g1",
        board="default",
        suffix="node_race",
    )
    with kb.connect(board="default") as conn:
        _stage_and_activate(conn, request)
        path = _database_path(conn)
    task_id = _producer_id(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda claimer: _claim_from_path(path, task_id, "default", claimer),
                ("race-a", "race-b"),
            )
        )

    assert sum(results) == 1
    with kb.connect(board="default") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND status='running'",
            (task_id,),
        ).fetchone()[0] == 1


def test_two_runs_in_shared_board_contend_same_global_resource(
    managed_home, tmp_path
) -> None:
    root = tmp_path / "resource-root"
    root.mkdir()
    requests = [
        _mutating_request(
            run_id=f"run-shared-resource-{index}",
            create_key=f"hm-loop.graph.shared-resource.{index}",
            board="default",
            suffix=f"shared_{index}",
            root=root,
            base_commit="1" * 40,
            resources=["repo:phase6/shared"],
        )
        for index in (1, 2)
    ]
    with kb.connect(board="default") as conn:
        for request in requests:
            _stage_and_activate(conn, request)
        path = _database_path(conn)
    task_ids = [_producer_id(request) for request in requests]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: _claim_from_path(path, item[1], "default", item[0]),
                zip(("shared-a", "shared-b"), task_ids),
            )
        )

    assert sum(results) == 1
    with kb.connect(board="default") as conn:
        lease = conn.execute(
            "SELECT task_id FROM managed_resource_leases "
            "WHERE resource_key='repo:phase6/shared' AND released_at IS NULL"
        ).fetchone()
        assert lease is not None
        assert lease["task_id"] == task_ids[results.index(True)]


def test_disjoint_resources_claim_concurrently(managed_home, tmp_path) -> None:
    root = tmp_path / "disjoint-root"
    root.mkdir()
    requests = [
        _mutating_request(
            run_id=f"run-disjoint-{index}",
            create_key=f"hm-loop.graph.disjoint.{index}",
            board="default",
            suffix=f"disjoint_{index}",
            root=root,
            base_commit="2" * 40,
            resources=[f"repo:phase6/disjoint-{index}"],
        )
        for index in (1, 2)
    ]
    with kb.connect(board="default") as conn:
        for request in requests:
            _stage_and_activate(conn, request)
        path = _database_path(conn)
    task_ids = [_producer_id(request) for request in requests]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: _claim_from_path(path, item[1], "default", item[0]),
                zip(("disjoint-a", "disjoint-b"), task_ids),
            )
        )

    assert results == [True, True]
    with kb.connect(board="default") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM managed_resource_leases WHERE released_at IS NULL"
        ).fetchone()[0] == 2


def test_managed_multi_resource_claim_is_lexical_all_or_none(
    managed_home, tmp_path
) -> None:
    root = tmp_path / "multi-resource-root"
    root.mkdir()
    resources = ["repo:phase6/a", "repo:phase6/b"]
    requests = [
        _mutating_request(
            run_id=f"run-multi-resource-{index}",
            create_key=f"hm-loop.graph.multi-resource.{index}",
            board="default",
            suffix=f"multi_{index}",
            root=root,
            base_commit="3" * 40,
            resources=resources,
        )
        for index in (1, 2)
    ]
    with kb.connect(board="default") as conn:
        for request in requests:
            _stage_and_activate(conn, request)
        path = _database_path(conn)
    task_ids = [_producer_id(request) for request in requests]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: _claim_from_path(path, item[1], "default", item[0]),
                zip(("multi-a", "multi-b"), task_ids),
            )
        )

    assert sum(results) == 1
    winner = task_ids[results.index(True)]
    with kb.connect(board="default") as conn:
        leases = conn.execute(
            "SELECT resource_key,task_id FROM managed_resource_leases "
            "WHERE released_at IS NULL ORDER BY resource_key"
        ).fetchall()
        assert [(row["resource_key"], row["task_id"]) for row in leases] == [
            ("repo:phase6/a", winner),
            ("repo:phase6/b", winner),
        ]


def test_expired_managed_lease_reclaim_requires_dead_exact_generation(
    managed_home, tmp_path
) -> None:
    root = tmp_path / "reclaim-root"
    root.mkdir()
    requests = [
        _mutating_request(
            run_id=f"run-reclaim-{index}",
            create_key=f"hm-loop.graph.reclaim.{index}",
            board="default",
            suffix=f"reclaim_{index}",
            root=root,
            base_commit="4" * 40,
            resources=["repo:phase6/reclaim"],
        )
        for index in (1, 2)
    ]
    first_id, second_id = [_producer_id(request) for request in requests]
    with kb.connect(board="default") as conn:
        for request in requests:
            _stage_and_activate(conn, request)
        assert kb._claim_managed_task(
            conn,
            first_id,
            board="default",
            claimer="reclaim-owner",
            max_in_progress_per_profile=10_000,
            _managed_capacity_lock_held=True,
        )
        conn.execute(
            "UPDATE managed_resource_leases SET expires_at=0 "
            "WHERE resource_key='repo:phase6/reclaim'"
        )
        conn.commit()

        assert kb._claim_managed_task(
            conn,
            second_id,
            board="default",
            claimer="reclaim-contender",
            max_in_progress_per_profile=10_000,
            _managed_capacity_lock_held=True,
        ) is None

        current_run_id = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (first_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE tasks SET status='blocked',current_run_id=NULL,claim_lock=NULL "
            "WHERE id=?",
            (first_id,),
        )
        conn.execute(
            "UPDATE task_runs SET status='blocked',ended_at=1 WHERE id=?",
            (current_run_id,),
        )
        conn.commit()

        assert kb._claim_managed_task(
            conn,
            second_id,
            board="default",
            claimer="reclaim-contender",
            max_in_progress_per_profile=10_000,
            _managed_capacity_lock_held=True,
        )
        lease = conn.execute(
            "SELECT task_id FROM managed_resource_leases "
            "WHERE resource_key='repo:phase6/reclaim' AND released_at IS NULL"
        ).fetchone()
        assert lease["task_id"] == second_id
        assert conn.execute(
            "SELECT COUNT(*) FROM managed_resource_lease_events "
            "WHERE resource_key='repo:phase6/reclaim' AND kind='released' "
            "AND reason='expired_reclaimed'"
        ).fetchone()[0] == 1


def test_bind_managed_workspace_rejects_project_root_duplicate_foreign_and_allows_disjoint(
    managed_home, tmp_path
) -> None:
    root = tmp_path / "workspace-root"
    base_commit = _init_git_repo(root)

    project_root = _mutating_request(
        run_id="run-workspace-root",
        create_key="hm-loop.graph.workspace-root.g1",
        board="default",
        suffix="workspace_root",
        root=root,
        base_commit=base_commit,
        resources=["repo:phase6/workspace-root"],
    )
    project_root_producer = next(
        node for node in project_root["nodes"] if node["task_kind"] == "producer"
    )
    project_root_producer["workspace_path"] = str(root)
    project_root = _rehashed(project_root)

    duplicate = _make_two_producer_logical_node(
        run_id="run-workspace-duplicate",
        create_key="hm-loop.graph.workspace-duplicate.g1",
        board="default",
    )
    duplicate_path = root / ".worktrees" / "duplicate"
    for index, producer in enumerate(
        node for node in duplicate["nodes"] if node["task_kind"] == "producer"
    ):
        producer.update(
            {
                "mutating": True,
                "resources": [f"repo:phase6/duplicate-{index}"],
                "workspace_kind": "worktree",
                "workspace_path": str(duplicate_path),
                "base_commit": base_commit,
                "owned_paths": [f"owned/duplicate-{index}.txt"],
                "owning_root_id": str(root),
                "owning_root_sha256": hashlib.sha256(str(root).encode()).hexdigest(),
            }
        )
    duplicate = _rehashed(duplicate)

    with kb.connect(board="default") as conn:
        with pytest.raises(kb.ManagedGraphValidationError):
            kb.create_managed_task_graph(conn, project_root)
        with pytest.raises(kb.ManagedGraphValidationError):
            kb.create_managed_task_graph(conn, duplicate)

        foreign = _mutating_request(
            run_id="run-workspace-foreign",
            create_key="hm-loop.graph.workspace-foreign.g1",
            board="default",
            suffix="workspace_foreign",
            root=root,
            base_commit=base_commit,
            resources=["repo:phase6/workspace-foreign"],
        )
        _stage_and_activate(conn, foreign)
        foreign_id = _producer_id(foreign)
        assert kb._claim_managed_task(
            conn,
            foreign_id,
            board="default",
            claimer="foreign-owner",
            max_in_progress_per_profile=10_000,
            _managed_capacity_lock_held=True,
        )
        foreign_target = Path(
            next(
                node["workspace_path"]
                for node in foreign["nodes"]
                if node["task_kind"] == "producer"
            )
        )
        foreign_target.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(foreign_target)], check=True)
        foreign_task = kb.get_task(conn, foreign_id)
        assert foreign_task is not None
        with pytest.raises(kb.ManagedTaskAuthorityError, match="foreign workspace"):
            kb.resolve_workspace(foreign_task, board="default", conn=conn)

        disjoint_requests = [
            _mutating_request(
                run_id=f"run-workspace-disjoint-{index}",
                create_key=f"hm-loop.graph.workspace-disjoint.{index}",
                board="default",
                suffix=f"workspace_disjoint_{index}",
                root=root,
                base_commit=base_commit,
                resources=[f"repo:phase6/workspace-disjoint-{index}"],
            )
            for index in (1, 2)
        ]
        resolved = []
        for index, request in enumerate(disjoint_requests):
            _stage_and_activate(conn, request)
            task_id = _producer_id(request)
            assert kb._claim_managed_task(
                conn,
                task_id,
                board="default",
                claimer=f"disjoint-workspace-{index}",
                max_in_progress_per_profile=10_000,
                _managed_capacity_lock_held=True,
            )
            task = kb.get_task(conn, task_id)
            assert task is not None
            resolved.append(
                kb.resolve_workspace(task, board="default", conn=conn)
            )

        assert resolved[0] != resolved[1]
        assert all(path.is_dir() for path in resolved)
        assert all(kb._git_common_dir(path) == kb._git_common_dir(root) for path in resolved)


def test_generic_mutations_reject_managed_producer_barrier_and_descendant(
    managed_home,
) -> None:
    request = _single_producer_request(
        run_id="run-generic-fence",
        create_key="hm-loop.graph.generic-fence.g1",
        board="default",
        suffix="generic_fence",
    )
    producer_id = _producer_id(request)
    verifier_id = next(
        node["task_id"]
        for node in request["nodes"]
        if node["task_kind"] == "verifier"
    )
    barrier_id = next(
        node["task_id"]
        for node in request["nodes"]
        if node["task_kind"] == "barrier"
    )
    with kb.connect(board="default") as conn:
        _stage_and_activate(conn, request)
        before = {
            row["id"]: row["status"]
            for row in conn.execute(
                "SELECT id,status FROM tasks WHERE id IN (?,?,?)",
                (producer_id, verifier_id, barrier_id),
            )
        }
        edge_count = conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0]

        calls = (
            lambda: kb.complete_task(conn, producer_id, summary="forbidden"),
            lambda: kb.archive_task(conn, barrier_id),
            lambda: kb.request_changes(conn, producer_id, reason="forbidden"),
            lambda: kb.link_tasks(conn, producer_id, barrier_id),
            lambda: kb.unlink_tasks(conn, producer_id, verifier_id),
        )
        for call in calls:
            with pytest.raises(kb.ManagedTaskAuthorityError):
                call()

        after = {
            row["id"]: row["status"]
            for row in conn.execute(
                "SELECT id,status FROM tasks WHERE id IN (?,?,?)",
                (producer_id, verifier_id, barrier_id),
            )
        }
        assert after == before
        assert conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0] == edge_count


def test_managed_capacity_counts_other_board_and_unreadable_board_fails_closed(
    managed_home,
) -> None:
    kb.create_board("second")
    with kb.connect(board="second") as other:
        busy_id = kb.create_task(other, title="busy", assignee="alice")
        assert kb.claim_task(other, busy_id, board="second")

    board_slugs = {row["slug"] for row in kb.list_boards(include_archived=False)}
    assert {"default", "second"}.issubset(board_slugs)
    assert kb._managed_readonly_profile_count(
        kb.kanban_db_path(board="second"), "alice"
    ) == 1

    request = _single_producer_request(
        run_id="run-capacity",
        create_key="hm-loop.graph.capacity.g1",
        board="default",
        suffix="capacity",
    )
    producer_id = _producer_id(request)
    with kb.connect(board="default") as conn:
        _stage_and_activate(conn, request)
        assert kb._managed_profile_capacity_available(
            conn, producer_id, board="default", limit=1
        ) is False
        assert kb._claim_managed_task(
            conn,
            producer_id,
            board="default",
            claimer="capacity-contender",
            max_in_progress_per_profile=1,
            _managed_capacity_lock_held=True,
        ) is None

    with kb.connect(board="second") as other:
        other.execute("UPDATE tasks SET status='done' WHERE id=?", (busy_id,))
        other.commit()
    kb.create_board("broken")
    broken_path = kb.kanban_db_path(board="broken")
    broken_path.unlink()
    sqlite3.connect(str(broken_path)).close()
    assert kb._managed_readonly_profile_count(broken_path, "alice") is None

    with kb.connect(board="default") as conn:
        assert kb._claim_managed_task(
            conn,
            producer_id,
            board="default",
            claimer="capacity-contender",
            max_in_progress_per_profile=1,
            _managed_capacity_lock_held=True,
        ) is None
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (producer_id,)
        ).fetchone()[0] == "ready"
