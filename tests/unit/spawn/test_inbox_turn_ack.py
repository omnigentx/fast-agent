"""Inbox delivery survives a failed or interrupted agent turn."""

from unittest.mock import AsyncMock, Mock

import pytest

from fast_agent.spawn import isolated_runner


@pytest.mark.asyncio
async def test_failed_send_does_not_ack_inbox(monkeypatch):
    agent = Mock()
    agent.send = AsyncMock(side_effect=RuntimeError("interrupted"))
    bus = Mock()
    save = AsyncMock()
    monkeypatch.setattr(isolated_runner, "_save_agent_context_snapshot", save)

    with pytest.raises(RuntimeError, match="interrupted"):
        await isolated_runner._send_pending_and_ack(
            agent, "request", "run-a", "Alex", "idle", bus, ["message-a"],
        )

    save.assert_not_awaited()
    bus.mark_done.assert_not_called()


@pytest.mark.asyncio
async def test_successful_send_persists_before_ack(monkeypatch):
    order = []
    agent = Mock()
    agent.send = AsyncMock(side_effect=lambda _: order.append("send"))
    bus = Mock()
    bus.mark_done.side_effect = lambda *_: order.append("ack")

    async def save(*_):
        order.append("snapshot")

    monkeypatch.setattr(isolated_runner, "_save_agent_context_snapshot", save)
    await isolated_runner._send_pending_and_ack(
        agent, "request", "run-a", "Alex", "idle", bus, ["message-a"],
    )

    assert order == ["send", "snapshot", "ack"]
