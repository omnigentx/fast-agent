"""Team directives must enter the inbox watched by the selected session."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_legacy_blocking_mode_returns_push_session_without_polling(monkeypatch):
    from fast_agent.spawn.servers import agent_spawner_server as server

    called = {}

    async def fake_spawn(**kwargs):
        called.update(kwargs)
        return SimpleNamespace(
            session_id="team-1", team_name="sample", workspace="/tmp/team-1",
            template={"name": "sample"},
            agents={"PM": {"run_id": "run-1", "role": "pm", "status": "running"}},
        )

    monkeypatch.setattr(server, "_spawn_team", fake_spawn)
    monkeypatch.setattr(server, "_emit_team_event", lambda *args, **kwargs: None)
    result = json.loads(await server.spawn_team_tool(
        template="sample", project_brief="work", team_name="sample",
        mode="blocking", timeout_seconds=300,
    ))
    assert called["mode"] == "background"
    assert result["status"] == "orchestrator_spawned"
    assert "push events" in result["mode_notice"]

@pytest.mark.asyncio
async def test_directives_are_isolated_by_team_session(tmp_path: Path, monkeypatch) -> None:
    from fast_agent.spawn.message_bus import MessageBus
    from fast_agent.spawn.servers import agent_spawner_server as server

    sessions = {
        "team-a": SimpleNamespace(
            template={"orchestrator": "pm"},
            agents={"Alex": {"role": "pm", "run_id": "run-a"}},
        ),
        "team-b": SimpleNamespace(
            template={"orchestrator": "pm"},
            agents={"Bailey": {"role": "pm", "run_id": "run-b"}},
        ),
    }
    directories = {
        "run-a": tmp_path / "messages" / "team-a",
        "run-b": tmp_path / "messages" / "team-b",
    }

    class Registry:
        def get_latest(self, run_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                status="running",
                original_config={
                    "env_vars": {"TEAM_MESSAGES_DIR": str(directories[run_id])}
                },
            )

    monkeypatch.setattr(server, "get_team_session", sessions.get)
    monkeypatch.setattr(server, "_registry", Registry())

    first = json.loads(await server.send_team_message("team-a", "change A"))
    second = json.loads(await server.send_team_message("team-b", "change B"))

    assert first["status"] == second["status"] == "sent"
    assert [m.content for m in MessageBus(directories["run-a"]).read_unread("Alex")] == ["change A"]
    assert [m.content for m in MessageBus(directories["run-b"]).read_unread("Bailey")] == ["change B"]
    assert not (tmp_path / "messages" / "alex_inbox.jsonl").exists()


@pytest.mark.asyncio
async def test_missing_session_inbox_fails_before_queueing(tmp_path: Path, monkeypatch) -> None:
    from fast_agent.spawn.servers import agent_spawner_server as server

    session = SimpleNamespace(
        template={"orchestrator": "pm"},
        agents={"Alex": {"role": "pm", "run_id": "run-a"}},
    )
    record = SimpleNamespace(status="idle", original_config={"env_vars": {}})
    registry = SimpleNamespace(get_latest=lambda _: record)
    wake = AsyncMock()

    monkeypatch.setattr(server, "get_team_session", lambda _: session)
    monkeypatch.setattr(server, "_registry", registry)
    monkeypatch.setattr(server, "_check_and_resume_on_inbox", wake)

    result = json.loads(await server.send_team_message("team-a", "change A"))

    assert "error" in result
    wake.assert_not_awaited()
    assert not list(tmp_path.rglob("*_inbox.jsonl"))
