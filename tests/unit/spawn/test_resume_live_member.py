import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fast_agent.spawn.agent_channel import AgentChannel
from fast_agent.spawn.message_bus import MessageBus
from fast_agent.spawn.servers import agent_spawner_server as spawner


@pytest.mark.asyncio
async def test_explicit_resume_queues_existing_live_team_process(monkeypatch, tmp_path):
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(tmp_path / "registry.db"))
    record = SimpleNamespace(
        is_terminal=False,
        status="idle",
        lifecycle="resumable",
        agent_name="QE",
        session_id="team-a",
        run_id="live-run",
        original_config={
            "agent_name": "QE",
            "env_vars": {"TEAM_SESSION_ID": "team-a", "TEAM_MESSAGES_DIR": str(tmp_path)},
        },
    )
    monkeypatch.setattr(
        spawner, "_registry", SimpleNamespace(get=lambda _: record, list_all=lambda: [record])
    )
    monkeypatch.setattr(AgentChannel, "is_alive", lambda *a, **kw: True)
    signals = []
    monkeypatch.setattr(
        AgentChannel, "send_signal", lambda *a, **kw: signals.append((a, kw)) or True
    )
    start = AsyncMock()
    monkeypatch.setattr(spawner, "run_isolated_agent_background", start)
    out = json.loads(await spawner.resume_spawn("live-run", "Run focused retro"))
    assert out["status"] == "resumed" and out["new_run_id"] == "live-run"
    assert out["delivery"] == "queued_existing_process"
    start.assert_not_called()
    assert signals[0][1] == {"session_id": "team-a", "run_id": "live-run"}
    assert MessageBus(str(tmp_path)).read_unread("QE")[0].content == "Run focused retro"


@pytest.mark.asyncio
async def test_explicit_resume_refuses_multiple_live_team_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(tmp_path / "registry.db"))
    records = [
        SimpleNamespace(
            is_terminal=False,
            status="idle",
            lifecycle="resumable",
            agent_name="QE",
            session_id="team-a",
            run_id=rid,
            original_config={
                "env_vars": {"TEAM_SESSION_ID": "team-a", "TEAM_MESSAGES_DIR": str(tmp_path)}
            },
        )
        for rid in ["a", "b"]
    ]
    monkeypatch.setattr(
        spawner, "_registry", SimpleNamespace(get=lambda _: records[0], list_all=lambda: records)
    )
    monkeypatch.setattr(AgentChannel, "is_alive", lambda *a, **kw: True)
    start = AsyncMock()
    monkeypatch.setattr(spawner, "run_isolated_agent_background", start)
    out = json.loads(await spawner.resume_spawn("a", "Continue"))
    assert "Multiple live runs" in out["error"]
    start.assert_not_called()


@pytest.mark.asyncio
async def test_simultaneous_explicit_resume_claims_only_one_launch(monkeypatch, tmp_path):
    import asyncio

    from fast_agent.spawn.live_resume import guard_member_resume

    record = SimpleNamespace(agent_name="QE", session_id="team-a", original_config={})
    monkeypatch.setattr(spawner, "_registry", SimpleNamespace(get=lambda _: record))
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(tmp_path / "registry.db"))
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    @guard_member_resume
    async def launch(run_id, follow_up_task):
        calls.append(run_id)
        entered.set()
        await release.wait()
        return '{"status":"resumed"}'

    first = asyncio.create_task(launch("old", "first"))
    await entered.wait()
    second = json.loads(await launch("old", "second"))
    assert second["status"] == "busy" and calls == ["old"]
    release.set()
    assert json.loads(await first)["status"] == "resumed"
    assert json.loads(await launch("old", "later"))["status"] == "resumed"


@pytest.mark.asyncio
async def test_failed_wake_keeps_directive_without_respawn(monkeypatch, tmp_path):
    from fast_agent.spawn.live_resume import queue_live_follow_up

    record = SimpleNamespace(
        agent_name="QE",
        session_id="team-a",
        run_id="live",
        original_config={"env_vars": {"TEAM_MESSAGES_DIR": str(tmp_path)}},
    )
    monkeypatch.setattr(AgentChannel, "is_alive", lambda *a, **kw: True)
    monkeypatch.setattr(AgentChannel, "send_signal", lambda *a, **kw: False)
    result = json.loads(
        queue_live_follow_up(
            SimpleNamespace(list_all=lambda: [record]), record, "live", "Retry later"
        )
    )
    assert result["message_queued"] and result["status"] == "not_resumed"
    assert MessageBus(str(tmp_path)).read_unread("QE")[0].content == "Retry later"
