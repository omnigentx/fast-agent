"""Team MCP subprocesses must receive a compact, usable addressing roster."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import yaml

from fast_agent.spawn.config_reader import get_server_env
from fast_agent.spawn.isolated_runner import create_child_config
from fast_agent.spawn.servers import email_server


def _team_roles() -> dict[str, dict[str, str]]:
    return {
        "pm": {"agent_name": "Blake [PM]", "instruction": "long instructions" * 100},
        "dev": {"agent_name": "Dakota [Dev]", "instruction": "more instructions" * 100},
        "qe": {"agent_name": "Bailey [QE]", "instruction": "test instructions" * 100},
    }


def test_team_server_env_contains_only_addressing_roster(monkeypatch):
    monkeypatch.setenv("TEAM_ROLES_CONFIG", json.dumps(_team_roles()))
    monkeypatch.setenv("TEAM_SESSION_ID", "guard-e2e")

    for server in ("email", "meeting_room", "agent_spawner"):
        env = get_server_env(server, "/tmp/team", "Bailey [QE]")
        assert env is not None
        assert env["TEAM_SESSION_ID"] == "guard-e2e"
        assert env["TEAM_MY_NAME"] == "Bailey [QE]"
        assert json.loads(env["TEAM_ROLES_CONFIG"]) == {
            "pm": {"agent_name": "Blake [PM]"},
            "dev": {"agent_name": "Dakota [Dev]"},
            "qe": {"agent_name": "Bailey [QE]"},
        }
        assert "instructions" not in env["TEAM_ROLES_CONFIG"]

    assert get_server_env("filesystem") is None


def test_child_config_propagates_roster_and_email_accepts_teammates(monkeypatch, tmp_path: Path):
    (tmp_path / "fastagent.config.yaml").write_text(
        yaml.safe_dump(
            {
                "mcp": {
                    "servers": {
                        "email": {
                            "command": "python",
                            "args": ["-m", "fast_agent.spawn.servers.email_server"],
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setenv("TEAM_ROLES_CONFIG", json.dumps(_team_roles()))
    monkeypatch.setenv("TEAM_MY_NAME", "Bailey [QE]")
    monkeypatch.setenv("TEAM_SESSION_ID", "guard-e2e")

    config_dir = create_child_config(
        project_dir=tmp_path,
        workspace_dir=str(tmp_path / "workspace"),
        servers=["email"],
        agent_name="Bailey [QE]",
    )
    config = yaml.safe_load((Path(config_dir) / "fastagent.config.yaml").read_text())
    child_env = config["mcp"]["servers"]["email"]["env"]
    assert len(child_env["TEAM_ROLES_CONFIG"]) < 150
    assert child_env["TEAM_SESSION_ID"] == "guard-e2e"

    # Model the explicit env passed to the MCP subprocess. The real E2E
    # failure was that this mapping had no TEAM_ROLES_CONFIG at all.
    monkeypatch.setenv("TEAM_ROLES_CONFIG", child_env["TEAM_ROLES_CONFIG"])
    bus = Mock()
    bus.send.return_value = SimpleNamespace(message_id="msg-1")
    monkeypatch.setattr(email_server, "get_bus", lambda: bus)
    monkeypatch.setattr(email_server, "auto_wake_if_idle", lambda _name: None)
    result = json.loads(
        email_server.send_email(
            to="Blake [PM]", cc="Dakota [Dev]", body="FYI", my_name="Bailey [QE]"
        )
    )
    assert result["status"] == "sent"
    assert result["from"] == "Bailey [QE]"
    assert result["sent"][0]["to"] == "Blake [PM]"
    assert result["cc_sent"][0]["to"] == "Dakota [Dev]"
    assert bus.send.call_count == 2


def test_invalid_team_roster_is_not_forwarded(monkeypatch):
    monkeypatch.setenv("TEAM_ROLES_CONFIG", "not-json")
    env = get_server_env("email", "/tmp/team", "Bailey [QE]")
    assert env is not None
    assert "TEAM_ROLES_CONFIG" not in env
