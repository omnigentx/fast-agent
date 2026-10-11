"""Keep-alive handoffs and live inbox hooks must deliver one envelope."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fast_agent.spawn.inbox_delivery import build_inbox_prompt
from fast_agent.spawn.inbox_watcher_hook import InboxWatcherHook
from fast_agent.spawn.message_bus import MessageBus
from fast_agent.types import LlmStopReason


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_first', [False, True])
async def test_handoff_is_not_injected_twice_and_ack_waits_for_success(tmp_path, cancel_first):
    bus = MessageBus(tmp_path)
    envelope = bus.send('PM', 'Dev', 'keep the changed requirement')
    prompt = build_inbox_prompt([envelope])
    hook = InboxWatcherHook('Dev', tmp_path, check_interval=0)
    runner = MagicMock()
    await hook.before_llm_call(runner, [prompt])
    runner.append_messages.assert_not_called()
    assert len(bus.read_unread('Dev')) == 1
    if cancel_first:
        await hook.after_llm_call(runner, SimpleNamespace(stop_reason=LlmStopReason.CANCELLED))
        assert len(bus.read_unread('Dev')) == 1
        runner = MagicMock()
        await hook.before_llm_call(runner, [prompt])
        runner.append_messages.assert_not_called()
    await hook.after_llm_call(runner, SimpleNamespace(stop_reason=LlmStopReason.END_TURN))
    assert bus.read_unread('Dev') == []


@pytest.mark.asyncio
async def test_new_message_alongside_existing_handoff_is_delivered(tmp_path):
    bus = MessageBus(tmp_path)
    first = bus.send('PM', 'Dev', 'first revision')
    bus.send('PM', 'Dev', 'second revision')
    hook = InboxWatcherHook('Dev', tmp_path, check_interval=0)
    runner = MagicMock()
    await hook.before_llm_call(runner, [build_inbox_prompt([first])])
    assert runner.append_messages.call_count == 1
    delta = runner.append_messages.call_args.args[0]
    assert 'second revision' in delta.content[0].text
    assert 'first revision' not in delta.content[0].text
    await hook.after_llm_call(runner, SimpleNamespace(stop_reason=LlmStopReason.TOOL_USE))
    assert bus.read_unread('Dev') == []
