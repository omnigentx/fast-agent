"""Inbox delivery must survive a failed or cancelled model call."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fast_agent.spawn.inbox_watcher_hook import InboxWatcherHook
from fast_agent.spawn.message_bus import MessageBus
from fast_agent.types.llm_stop_reason import LlmStopReason


@pytest.mark.asyncio
async def test_inbox_stays_unread_until_model_response(tmp_path):
    bus = MessageBus(tmp_path)
    message = bus.send("PM", "Dev", "change the acceptance criteria")
    hook = InboxWatcherHook("Dev", tmp_path, check_interval=60)
    first_runner = MagicMock()

    await hook.before_llm_call(first_runner, [])
    assert [item.message_id for item in bus.read_unread("Dev")] == [message.message_id]
    assert first_runner.append_messages.call_count == 1

    # The provider fails, so after_llm_call never runs. A new runner must
    # receive the same instruction even inside the throttle interval.
    retry_runner = MagicMock()
    await hook.before_llm_call(retry_runner, [])
    assert retry_runner.append_messages.call_count == 1
    await hook.after_llm_call(retry_runner, SimpleNamespace(stop_reason=LlmStopReason.END_TURN))
    assert bus.read_unread("Dev") == []


@pytest.mark.asyncio
async def test_cancelled_model_call_does_not_acknowledge(tmp_path):
    bus = MessageBus(tmp_path)
    message = bus.send("PM", "Dev", "new requirement")
    hook = InboxWatcherHook("Dev", tmp_path, check_interval=0)
    runner = MagicMock()

    await hook.before_llm_call(runner, [])
    await hook.after_llm_call(runner, SimpleNamespace(stop_reason=LlmStopReason.CANCELLED))
    assert [item.message_id for item in bus.read_unread("Dev")] == [message.message_id]

    # Retrying on the same runner reuses the appended delta; no duplicate.
    await hook.before_llm_call(runner, [])
    assert runner.append_messages.call_count == 1
    await hook.after_llm_call(runner, SimpleNamespace(stop_reason=LlmStopReason.TOOL_USE))
    assert bus.read_unread("Dev") == []


@pytest.mark.asyncio
async def test_only_staged_batch_is_acknowledged(tmp_path):
    bus = MessageBus(tmp_path)
    for number in range(3):
        bus.send("PM", "Dev", f"revision {number}")
    hook = InboxWatcherHook("Dev", tmp_path, check_interval=0, max_per_inject=2)
    runner = MagicMock()

    await hook.before_llm_call(runner, [])
    await hook.after_llm_call(runner, SimpleNamespace(stop_reason=LlmStopReason.END_TURN))
    assert [item.content for item in bus.read_unread("Dev")] == ["revision 2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_hooks", [False, True])
async def test_isolated_runner_wires_ack_for_both_hook_paths(tmp_path, monkeypatch, existing_hooks):
    from fast_agent.agents.tool_runner import ToolRunnerHooks
    from fast_agent.spawn import inbox_watcher_hook, isolated_runner

    bus = MessageBus(tmp_path)
    bus.send("PM", "Dev", "revision")
    watcher = InboxWatcherHook("Dev", tmp_path, check_interval=0)
    monkeypatch.setattr(inbox_watcher_hook, "create_inbox_watcher", lambda: watcher)
    monkeypatch.setattr(isolated_runner, "emit_event", lambda *_args, **_kw: None)
    child = MagicMock()
    child.tool_runner_hooks = ToolRunnerHooks() if existing_hooks else None
    child.message_history = []

    isolated_runner._install_tool_hooks({"Dev": child}, "run-1", "Dev")
    runner = MagicMock()
    runner.request_params = SimpleNamespace(model="test")
    await child.tool_runner_hooks.before_llm_call(runner, [])
    assert len(bus.read_unread("Dev")) == 1
    await child.tool_runner_hooks.after_llm_call(
        runner, SimpleNamespace(stop_reason=LlmStopReason.END_TURN, content=[])
    )
    assert bus.read_unread("Dev") == []
