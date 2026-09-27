"""A Jarvis directive must land in the PM's session-scoped inbox."""

import json
from types import SimpleNamespace

import pytest

from fast_agent.spawn.message_bus import MessageBus
from fast_agent.spawn.servers import agent_spawner_server as spawner


@pytest.mark.asyncio
async def test_send_team_message_uses_pm_inbox_from_spawn_record(
    monkeypatch, tmp_path,
):
    session_id = "team-123"
    inbox_dir = tmp_path / "messages" / session_id
    session = SimpleNamespace(
        template={"orchestrator": "pm"},
        agents={"Bennett [PM]": {"role": "pm", "run_id": "pm-run"}},
    )
    record = SimpleNamespace(
        status="running",
        original_config={"env_vars": {"TEAM_MESSAGES_DIR": str(inbox_dir)}},
    )
    monkeypatch.setattr(spawner, "get_team_session", lambda _: session)
    monkeypatch.setattr(
        spawner, "_registry", SimpleNamespace(get_latest=lambda _: record),
    )
    monkeypatch.setattr(spawner, "_PROJECT_DIR", tmp_path)

    result = json.loads(await spawner.send_team_message(
        session_id, "Use revision checks", priority="high",
    ))

    assert result["status"] == "sent"
    assert result["to"] == "Bennett [PM]"
    unread = MessageBus(inbox_dir).read_unread("Bennett [PM]")
    assert len(unread) == 1
    assert unread[0].message_id == result["message_id"]
    assert unread[0].content == "Use revision checks"
    assert not (tmp_path / ".runtime" / "state" / "messages" /
                "bennett__pm_inbox.jsonl").exists()
