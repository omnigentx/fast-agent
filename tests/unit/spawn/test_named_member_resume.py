"""Resolve stable team member identity without copying ephemeral run IDs."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fast_agent.spawn.servers import agent_spawner_server as s


def record(name="QE", session="team-a", run="current-qe"):
    return SimpleNamespace(
        agent_name=name,
        session_id=session,
        run_id=run,
        original_config={"env_vars": {"TEAM_SESSION_ID": session}},
    )


@pytest.mark.asyncio
async def test_named_member_resolves_current_run_and_uses_existing_resume(monkeypatch):
    session = SimpleNamespace(agents={"qe": {"agent_name": "QE", "run_id": "old-qe"}})
    monkeypatch.setattr(s, "get_team_session", lambda sid: session if sid == "team-a" else None)
    monkeypatch.setattr(s, "_registry", SimpleNamespace(get_latest=lambda rid: record()))
    resume = AsyncMock(return_value='{"status":"resumed"}')
    monkeypatch.setattr(s, "resume_spawn", resume)
    result = json.loads(await s.resume_team_member("team-a", "QE", "Read checkpoint"))
    assert result["status"] == "resumed"
    resume.assert_awaited_once_with("current-qe", "Read checkpoint")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["missing_team", "missing_name", "foreign", "wrong_name", "ambiguous"]
)
async def test_named_resolution_fails_closed_without_waking_anyone(monkeypatch, case):
    agents = {"qe": {"agent_name": "QE", "run_id": "old"}}
    if case == "ambiguous":
        agents["qe2"] = {"agent_name": "QE", "run_id": "other"}
    monkeypatch.setattr(
        s,
        "get_team_session",
        lambda sid: None if case == "missing_team" else SimpleNamespace(agents=agents),
    )
    resolved = record(
        session="team-b" if case == "foreign" else "team-a",
        name="Dev" if case == "wrong_name" else "QE",
    )
    monkeypatch.setattr(s, "_registry", SimpleNamespace(get_latest=lambda _: resolved))
    resume = AsyncMock()
    monkeypatch.setattr(s, "resume_spawn", resume)
    result = json.loads(
        await s.resume_team_member(
            "team-a", "Absent" if case == "missing_name" else "QE", "Read checkpoint"
        )
    )
    assert "error" in result
    resume.assert_not_called()
