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

    async def save(*_, **kwargs):
        order.append("snapshot")

    monkeypatch.setattr(isolated_runner, "_save_agent_context_snapshot", save)
    await isolated_runner._send_pending_and_ack(
        agent, "request", "run-a", "Alex", "idle", bus, ["message-a"],
    )

    assert order == ["send", "snapshot", "ack"]


@pytest.mark.asyncio
async def test_snapshot_preserves_team_name_not_role(monkeypatch):
    import sys
    from types import ModuleType

    save = AsyncMock()
    service = ModuleType("services.context_persistence")
    service.save_agent_context = save
    monkeypatch.setitem(sys.modules, "services.context_persistence", service)
    monkeypatch.setenv("TEAM_MY_ROLE", "dev")
    monkeypatch.setenv("TEAM_SESSION_ID", "team-session")
    agent = Mock()
    agent.message_history = []
    agent.send = AsyncMock(return_value="completed")
    await isolated_runner._send_pending_and_ack(
        agent, "request", "run-a", "Alex", "idle", None, [],
        team_name="game-team",
    )
    assert save.await_args.kwargs["team_name"] == "game-team"
    assert save.await_args.kwargs["session_id"] == "team-session"
