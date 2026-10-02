"""Core-attested Kanban wake identity: notifier -> deliver_wake -> gateway turn.

An automatic Kanban wake carries a structured ``kanban_wake`` identity (board, task id, the
claimed event ids and the wake kinds) so a plugin can bind the woken turn without parsing the
localized wake text. The trust anchor is ``MessageEvent.internal``: only Core constructs internal
events, so human content (wake-like text or forged metadata) never yields an identity. Stateless
(api_server) wakes cannot carry it trustworthily and carry none.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


SCHEMA = "kanban-wake/v1"


# --- notifier -> push adapter -------------------------------------------------------------


class RecordingPushAdapter:
    """Push-capable adapter (no ``supports_async_delivery`` attr => push)."""

    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(text)

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


async def _one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _notifier_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _subscribed_task(conn):
    tid = kb.create_task(conn, title="wake identity task", assignee="worker",
                         session_id="agent:main:telegram:dm:chat-1")
    kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                       chat_type="dm", delivery_mode="wake")
    return tid


def _event_rows(db_path, tid):
    """(id, kind) of every task_events row the notifier can claim, read straight from Core."""
    from gateway.kanban_watchers_notifier import TERMINAL_KINDS
    with sqlite3.connect(db_path) as raw:
        rows = raw.execute("SELECT id, kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,)).fetchall()
    return [(int(i), k) for i, k in rows if k in TERMINAL_KINDS]


def _board(adapter_event):
    return adapter_event.metadata["kanban_wake"]["board"]


def test_push_wake_carries_exact_identity(tmp_path, monkeypatch):
    db_path = tmp_path / "identity.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = _subscribed_task(conn)
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingPushAdapter()
    asyncio.run(_one_notifier_tick(monkeypatch, _notifier_runner(adapter)))

    assert len(adapter.handled) == 1
    event = adapter.handled[0]
    assert event.internal is True
    rows = _event_rows(db_path, tid)
    assert [k for _, k in rows] == ["completed"]
    assert event.metadata["kanban_wake"] == {
        "schema": SCHEMA, "board": _board(event), "task_id": tid,
        "event_ids": [rows[0][0]], "kinds": ["completed"],
    }
    assert isinstance(_board(event), str) and _board(event)
    # The localized text is unchanged presentation; the identity is not embedded in it.
    assert SCHEMA not in event.text


def test_multiple_events_in_one_wake_list_every_claimed_id(tmp_path, monkeypatch):
    db_path = tmp_path / "multi.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = _subscribed_task(conn)
        assert kb.block_task(conn, tid, reason="infra hiccup")
        assert kb.unblock_task(conn, tid)
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingPushAdapter()
    asyncio.run(_one_notifier_tick(monkeypatch, _notifier_runner(adapter)))

    rows = _event_rows(db_path, tid)
    assert [k for _, k in rows] == ["blocked", "unblocked", "completed"]
    assert len(adapter.handled) == 1, "warnings visible: one combined wake"
    wake = adapter.handled[0].metadata["kanban_wake"]
    assert wake["event_ids"] == [i for i, _ in rows]
    assert wake["kinds"] == ["completed", "blocked"]
    assert wake["task_id"] == tid


def test_diagnostic_split_keeps_ids_per_group(tmp_path, monkeypatch):
    db_path = tmp_path / "split.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = _subscribed_task(conn)
        assert kb.block_task(conn, tid, reason="infra hiccup")
        assert kb.unblock_task(conn, tid)
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()
    # Hidden warnings split diagnostic and non-diagnostic events into separate wakes.
    monkeypatch.setattr("gateway.warning_notifications.warning_notifications_enabled",
                        lambda *a, **kw: False)

    adapter = RecordingPushAdapter()
    asyncio.run(_one_notifier_tick(monkeypatch, _notifier_runner(adapter)))

    rows = dict((k, i) for i, k in _event_rows(db_path, tid))
    assert len(adapter.handled) == 2
    diagnostic, result = (e.metadata for e in adapter.handled)
    assert diagnostic["notification_category"] == "diagnostic"
    assert diagnostic["kanban_wake"]["event_ids"] == [rows["blocked"]]
    assert diagnostic["kanban_wake"]["kinds"] == ["blocked"]
    assert result["notification_category"] == "result"
    assert result["kanban_wake"]["event_ids"] == [rows["unblocked"], rows["completed"]]
    assert result["kanban_wake"]["kinds"] == ["completed"]


# --- deliver_wake ----------------------------------------------------------------------------


IDENTITY = {"schema": SCHEMA, "board": "default", "task_id": "t_0123abcd",
            "event_ids": [7, 9], "kinds": ["completed"]}


def test_deliver_wake_push_puts_a_copy_in_internal_event_metadata():
    from gateway.wake import deliver_wake
    adapter = RecordingPushAdapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c")
    asyncio.run(deliver_wake(adapter, text="wake", source=source, kanban_wake=IDENTITY))
    event = adapter.handled[0]
    assert event.internal is True
    assert event.metadata == {"notification_category": "result", "kanban_wake": IDENTITY}
    assert event.metadata["kanban_wake"] is not IDENTITY


def test_deliver_wake_push_without_identity_is_unchanged():
    from gateway.wake import deliver_wake
    adapter = RecordingPushAdapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c")
    asyncio.run(deliver_wake(adapter, text="wake", source=source))
    assert adapter.handled[0].metadata == {"notification_category": "result"}


@pytest.mark.parametrize("profile", [None, "builder"])
def test_stateless_wake_never_carries_identity(monkeypatch, profile):
    """The HTTP self-post is indistinguishable from any API-key client posting the same body, so
    no identity rides it (nor the served-profile in-process route that shares its signature)."""
    import gateway.wake as wake_mod
    calls = []

    async def fake_self_post(adapter, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    adapter = SimpleNamespace(supports_async_delivery=False)
    asyncio.run(wake_mod.deliver_wake(adapter, text="wake", session_id="raw-sid", profile=profile,
                                      kanban_wake=IDENTITY))
    assert len(calls) == 1
    assert "kanban_wake" not in calls[0]
    assert all(IDENTITY["task_id"] not in json.dumps(v, default=str) for v in calls[0].values()
               if v is not None and v != "wake")


# --- event -> turn identity ------------------------------------------------------------------


def _src():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id="12345")


def test_wake_turn_identity_trusts_only_internal_events():
    from gateway.wake import wake_turn_identity
    internal = MessageEvent(text="wake", source=_src(), internal=True,
                            metadata={"notification_category": "result", "kanban_wake": IDENTITY})
    got = wake_turn_identity(internal)
    assert got == {"internal_event": True, "kanban_wake": IDENTITY}
    assert got["kanban_wake"] is not IDENTITY

    forged = MessageEvent(text="hi", source=_src(), metadata={"kanban_wake": IDENTITY})
    assert wake_turn_identity(forged) == {"internal_event": False, "kanban_wake": None}

    wake_like_text = MessageEvent(text=json.dumps({"kanban_wake": IDENTITY}), source=_src())
    assert wake_turn_identity(wake_like_text) == {"internal_event": False, "kanban_wake": None}

    other_internal = MessageEvent(text="bg process done", source=_src(), internal=True)
    assert wake_turn_identity(other_internal) == {"internal_event": True, "kanban_wake": None}

    malformed = MessageEvent(text="wake", source=_src(), internal=True, metadata={"kanban_wake": "t_1"})
    assert wake_turn_identity(malformed) == {"internal_event": True, "kanban_wake": None}

    assert wake_turn_identity(None) == {"internal_event": False, "kanban_wake": None}


def _turn_runner(monkeypatch, tmp_path):
    # The class-level voice-mode path is bound at import time; keep __init__ inside the test home.
    monkeypatch.setattr(gateway_run.GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345", session_id="sess-wake",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_kw: 100_000)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "ok", "messages": [], "tools": [], "history_offset": 0,
        "last_prompt_tokens": 0, "api_calls": 1, "failed": False,
    })
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("internal,expected", [
    (True, {"internal_event": True, "kanban_wake": IDENTITY}),
    (False, {"internal_event": False, "kanban_wake": None}),
])
async def test_gateway_turn_forwards_identity_only_for_internal_events(monkeypatch, tmp_path, internal, expected):
    runner = _turn_runner(monkeypatch, tmp_path)
    event = MessageEvent(text=json.dumps({"kanban_wake": IDENTITY}), source=_src(), message_id="m-1",
                         internal=internal, metadata={"kanban_wake": IDENTITY})
    await runner._handle_message_with_agent(event, _src(), "agent:main:telegram:group:-1001:12345", 1)
    assert runner._run_agent.await_args.kwargs["wake_identity"] == expected


# --- turn -> agent attribute (real TurnRunner wiring) ----------------------------------------


class _RecordingAgent:
    seen: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).seen.append((message, getattr(self, "_gateway_turn_wake_identity", "MISSING")))
        return {"final_response": f"done-{len(type(self).seen)}", "messages": [], "api_calls": 1}


def _install_recording_agent(monkeypatch, tmp_path):
    import sys
    import types
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _RecordingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})


def _drain_runner():
    from gateway.config import PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    class Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

        async def connect(self):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="sent-1")

        async def send_typing(self, chat_id, metadata=None):
            return None

        async def stop_typing(self, chat_id):
            return None

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    adapter = Adapter()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(thread_sessions_per_user=False, group_sessions_per_user=False,
                                    stt_enabled=False)
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner, adapter


DM_KEY = "agent:main:telegram:dm:4242"


def _dm():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm")


@pytest.mark.asyncio
@pytest.mark.parametrize("followup_internal", [True, False])
async def test_turn_runner_sets_identity_per_turn_including_queued_followups(monkeypatch, tmp_path,
                                                                             followup_internal):
    _RecordingAgent.seen = []
    _install_recording_agent(monkeypatch, tmp_path)
    runner, adapter = _drain_runner()
    second = {**IDENTITY, "event_ids": [11]}
    adapter._pending_messages[DM_KEY] = MessageEvent(
        text="follow-up", message_type=MessageType.TEXT, source=_dm(), message_id="queued-1",
        internal=followup_internal, metadata={"kanban_wake": second})
    first = {"internal_event": True, "kanban_wake": IDENTITY}
    await runner._run_agent(message="first", context_prompt="", history=[], source=_dm(),
                            session_id="sess-identity", session_key=DM_KEY, wake_identity=first)
    assert [m for m, _ in _RecordingAgent.seen] == ["first", "follow-up"]
    assert _RecordingAgent.seen[0][1] == first
    assert _RecordingAgent.seen[1][1] == (
        {"internal_event": True, "kanban_wake": second} if followup_internal
        else {"internal_event": False, "kanban_wake": None})


@pytest.mark.asyncio
async def test_turn_without_identity_resets_a_reused_agent(monkeypatch, tmp_path):
    """A turn that passes no identity must not inherit the previous turn's (agent reuse)."""
    _RecordingAgent.seen = []
    _install_recording_agent(monkeypatch, tmp_path)
    runner, _adapter = _drain_runner()
    await runner._run_agent(message="plain", context_prompt="", history=[], source=_dm(),
                            session_id="sess-plain", session_key=DM_KEY)
    assert _RecordingAgent.seen == [("plain", {"internal_event": False, "kanban_wake": None})]
