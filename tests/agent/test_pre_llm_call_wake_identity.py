"""``pre_llm_call`` receives the gateway's per-turn wake identity (``internal_event`` /
``kanban_wake``), consumed once so a reused agent never replays a stale identity."""

from types import SimpleNamespace

import pytest

from agent.turn_context import _collect_pre_llm_call_context

IDENTITY = {"schema": "kanban-wake/v1", "board": "default", "task_id": "t_0123abcd",
            "event_ids": [7, 9], "kinds": ["completed"]}


@pytest.fixture
def hook_calls(monkeypatch):
    calls = []

    def fake_invoke_hook(name, **kwargs):
        if name == "pre_llm_call":
            calls.append(kwargs)
            kwargs["kanban_wake"] and kwargs["kanban_wake"]["event_ids"].append(999)  # hostile plugin
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", fake_invoke_hook)
    return calls


def _agent(**attrs):
    return SimpleNamespace(session_id="sess", model="m", platform="telegram", **attrs)


def _collect(agent):
    return _collect_pre_llm_call_context(agent, effective_task_id="sess", turn_id="turn-1",
                                         original_user_message="wake", messages=[],
                                         conversation_history=[])


def test_internal_wake_identity_reaches_hook_once(hook_calls):
    staged = {"internal_event": True, "kanban_wake": {**IDENTITY, "event_ids": [7, 9]}}
    agent = _agent(_gateway_turn_wake_identity=staged)
    _collect(agent)
    assert hook_calls[0]["internal_event"] is True
    assert hook_calls[0]["kanban_wake"]["task_id"] == IDENTITY["task_id"]
    # The hook got a copy: its mutation never reaches the staged identity.
    assert staged["kanban_wake"]["event_ids"] == [7, 9]
    # One-shot: a second turn on the same (cached) agent without fresh wiring sees nothing.
    _collect(agent)
    assert hook_calls[1]["internal_event"] is False
    assert hook_calls[1]["kanban_wake"] is None


@pytest.mark.parametrize("attrs", [
    {},  # CLI / TUI / api_server agents: the gateway never stages anything
    {"_gateway_turn_wake_identity": None},
    {"_gateway_turn_wake_identity": {"internal_event": False, "kanban_wake": IDENTITY}},
    {"_gateway_turn_wake_identity": {"internal_event": "yes", "kanban_wake": IDENTITY}},
    {"_gateway_turn_wake_identity": {"internal_event": True, "kanban_wake": "t_0123abcd"}},
    {"_gateway_turn_wake_identity": "garbage"},
])
def test_untrusted_or_absent_identity_is_dropped(hook_calls, attrs):
    _collect(_agent(**attrs))
    kwargs = hook_calls[0]
    expected_internal = attrs.get("_gateway_turn_wake_identity") == {
        "internal_event": True, "kanban_wake": "t_0123abcd"}
    assert kwargs["internal_event"] is expected_internal
    assert kwargs["kanban_wake"] is None
    # Existing fields are unchanged.
    assert kwargs["session_id"] == "sess" and kwargs["user_message"] == "wake"
