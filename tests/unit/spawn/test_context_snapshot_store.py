"""Snapshot loading must work in the standalone MCP subprocess."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from fast_agent.spawn.context_snapshot_store import load_latest_context_json
from fast_agent.spawn.servers import agent_spawner_server as spawner


def test_load_latest_context_json_scopes_team_session(monkeypatch, tmp_path):
    db_path = tmp_path / "registry.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE agent_context_snapshots ("
            "id INTEGER PRIMARY KEY, agent_name TEXT, session_id TEXT, "
            "context_json TEXT, created_at REAL)"
        )
        connection.executemany(
            "INSERT INTO agent_context_snapshots "
            "(agent_name, session_id, context_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            [
                ("Taylor [Dev]", "team-a", '["old"]', 1),
                ("Taylor [Dev]", "team-b", '["other team"]', 3),
                ("Taylor [Dev]", "team-a", '["latest"]', 2),
            ],
        )
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(db_path))

    assert load_latest_context_json("Taylor [Dev]", session_id="team-a") == '["latest"]'
    assert load_latest_context_json("Taylor [Dev]", session_id="unknown") is None
    assert load_latest_context_json("Taylor [Dev]") == '["other team"]'


def test_missing_db_is_not_created(monkeypatch, tmp_path):
    db_path = tmp_path / "absent.db"
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(db_path))

    assert load_latest_context_json("Taylor [Dev]") is None
    assert not db_path.exists()


@pytest.mark.asyncio
async def test_resume_spawn_reads_history_without_jarvis_import(monkeypatch, tmp_path):
    db_path = tmp_path / "registry.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE agent_context_snapshots ("
            "id INTEGER PRIMARY KEY, agent_name TEXT, session_id TEXT, "
            "context_json TEXT, created_at REAL)"
        )
        connection.execute(
            "INSERT INTO agent_context_snapshots "
            "(agent_name, session_id, context_json, created_at) "
            "VALUES ('Taylor [Dev]', 'team-a', '[{\"role\":\"user\"}]', 1)"
        )
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(db_path))
    record = SimpleNamespace(
        is_terminal=False, status="idle", lifecycle="resumable",
        original_config={"agent_name": "Taylor [Dev]", "env_vars": {"TEAM_SESSION_ID": "team-a"}},
        agent_name="Taylor [Dev]", role="dev", team_name="team-a", restart_count=0,
    )
    registry = SimpleNamespace(
        get=lambda _: record, _data={}, _load=lambda: None, _save=lambda: None,
    )
    monkeypatch.setattr(spawner, "_registry", registry)
    monkeypatch.setattr(spawner, "get_team_session", lambda _: None)
    observed = {}

    async def fake_start(**kwargs):
        observed.update(kwargs)
        return "next-run"

    monkeypatch.setattr(spawner, "run_isolated_agent_background", fake_start)
    result = json.loads(await spawner.resume_spawn("original-run", "Continue work"))

    assert result["new_run_id"] == "next-run"
    history_file = Path(observed["history_file"])
    assert json.loads(history_file.read_text(encoding="utf-8")) == [
        {"role": "user"}
    ]
    history_file.unlink()
