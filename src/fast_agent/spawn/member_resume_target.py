"""Resolve one member of one team without accepting an inferred run identifier."""

from __future__ import annotations

from typing import Any


def resolve_member_run(registry: Any, session: Any, session_id: str, agent_name: str) -> str:
    if not session_id or not agent_name or session is None:
        raise ValueError("Provide an existing session_id and exact agent_name")
    matches = [info for info in session.agents.values() if info.get("agent_name") == agent_name]
    if len(matches) != 1:
        raise ValueError("Team member was not found uniquely; inspect get_team_status")
    original = matches[0].get("run_id")
    record = registry.get_latest(original) if original else None
    if record is None or record.agent_name != agent_name:
        raise ValueError("Current member run is unavailable; no member was resumed")
    env = (record.original_config or {}).get("env_vars") or {}
    recorded_team = getattr(record, "session_id", "") or env.get("TEAM_SESSION_ID")
    if recorded_team != session_id or env.get("TEAM_SESSION_ID", session_id) != session_id:
        raise ValueError("Current run does not belong to the requested team")
    return record.run_id
