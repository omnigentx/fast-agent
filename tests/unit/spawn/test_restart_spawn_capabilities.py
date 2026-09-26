"""Restarted agents retain the same team identity and tool capabilities."""

import json
from types import SimpleNamespace

import pytest

from fast_agent.spawn.servers import agent_spawner_server as spawner


@pytest.mark.asyncio
async def test_restart_spawn_preserves_env_and_skills(monkeypatch):
    env_vars = {"TEAM_SESSION_ID": "team-a", "TEAM_MY_NAME": "Taylor [Dev]"}
    skills = ["dev-workflow", "engineering-principles"]
    record = SimpleNamespace(
        is_terminal=True, lifecycle="resumable", session_id="team-a",
        task="original task", role="dev", agent_name="Taylor [Dev]",
        team_name="team-a", restart_count=0,
        original_config={
            "task": "original task", "role": "dev", "agent_name": "Taylor [Dev]",
            "team_name": "team-a", "env_vars": env_vars, "skills": skills,
        },
    )
    registry = SimpleNamespace(
        get=lambda _: record,
        _data={"old-run": {"restart_count": 0}},
        _load=lambda: None,
        _save=lambda: None,
    )
    monkeypatch.setattr(spawner, "_registry", registry)
    captured = {}

    async def fake_start(**kwargs):
        captured.update(kwargs)
        return "new-run"

    monkeypatch.setattr(spawner, "run_isolated_agent_background", fake_start)

    result = json.loads(await spawner.restart_spawn("old-run"))

    assert result["new_run_id"] == "new-run"
    assert captured["env_vars"] == env_vars
    assert captured["skills"] == skills
    assert captured["session_id"] == "team-a"
