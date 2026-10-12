"""Deliver explicit follow-ups to an existing team process without respawning it."""

from __future__ import annotations

import json
from typing import Any

from fast_agent.spawn.agent_channel import AgentChannel
from fast_agent.spawn.message_bus import MessageBus


def queue_live_follow_up(registry: Any, record: Any, run_id: str, task: str) -> str | None:
    cfg = record.original_config or {}
    env = cfg.get("env_vars") or {}
    session_id = env.get("TEAM_SESSION_ID") or getattr(record, "session_id", "")
    if not session_id:
        return None
    name = cfg.get("agent_name") or record.agent_name
    records = registry.list_all() if hasattr(registry, "list_all") else [record]
    live = [
        r
        for r in records
        if r.agent_name == name
        and (
            getattr(r, "session_id", "")
            or (r.original_config or {}).get("env_vars", {}).get("TEAM_SESSION_ID")
        )
        == session_id
        and AgentChannel.is_alive(name, session_id=session_id, run_id=getattr(r, "run_id", run_id))
    ]
    if len(live) > 1:
        return json.dumps(
            {
                "error": "Multiple live runs exist for the same team member; reconcile before resuming.",
                "status": "not_resumed",
            }
        )
    if not live:
        # The launch lock can be released before the new child binds its
        # socket. Confirm process identity, not merely the registry status.
        if not registry.has_running_resume(name, session_id, verify_process=True):
            return None
        messages_dir = env.get("TEAM_MESSAGES_DIR")
        if not messages_dir:
            return json.dumps(
                {
                    "status": "not_resumed",
                    "error": "Launching team member has no session-scoped message directory.",
                }
            )
        message = MessageBus(messages_dir).send(
            from_name="Jarvis",
            to_name=name,
            content=task,
            message_type="directive",
            context={"session_id": session_id},
        )
        return json.dumps(
            {
                "status": "queued",
                "agent_name": name,
                "delivery": "queued_launching_process",
                "message_id": message.message_id,
            }
        )
    target = live[0]
    messages_dir = ((target.original_config or {}).get("env_vars") or {}).get("TEAM_MESSAGES_DIR")
    if not messages_dir:
        return json.dumps(
            {
                "error": "Live team member has no session-scoped message directory.",
                "status": "not_resumed",
            }
        )
    message = MessageBus(messages_dir).send(
        from_name="Jarvis",
        to_name=name,
        content=task,
        message_type="directive",
        context={"session_id": session_id},
    )
    live_id = getattr(target, "run_id", run_id)
    try:
        woke = AgentChannel.send_signal(name, "wake", session_id=session_id, run_id=live_id)
    except OSError:
        woke = False
    if not woke:
        return json.dumps(
            {
                "status": "not_resumed",
                "error": "Directive saved but wake delivery failed.",
                "message_queued": True,
                "message_id": message.message_id,
            }
        )
    return json.dumps(
        {
            "status": "resumed",
            "original_run_id": run_id,
            "new_run_id": live_id,
            "agent_name": name,
            "delivery": "queued_existing_process",
            "message_id": message.message_id,
        }
    )


def guard_member_resume(function):
    """Share the automatic-resume launch lock across independent MCP processes."""
    import fcntl
    import functools
    import hashlib
    import os
    from pathlib import Path

    @functools.wraps(function)
    async def guarded(run_id: str, follow_up_task: str) -> str:
        from fast_agent.spawn.servers import agent_spawner_server as server

        record = server._registry.get(run_id)
        cfg = record.original_config if record else {}
        env = (cfg or {}).get("env_vars") or {}
        session = env.get("TEAM_SESSION_ID") or getattr(record, "session_id", "")
        if not session:
            return await function(run_id, follow_up_task)
        backend = getattr(server._registry, "_backend", None)
        path = (
            getattr(backend, "_db_path", "")
            or getattr(backend, "_file", "")
            or os.environ.get("SPAWN_REGISTRY_DB")
        )
        if not path:
            return json.dumps(
                {"status": "not_resumed", "error": "Team launch registry path is unavailable."}
            )
        directory = Path(path).parent / "team_launch_locks"
        directory.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(f"{session}\0{record.agent_name}".encode()).hexdigest()[:24]
        with (directory / f"{name}.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return json.dumps(
                    {
                        "status": "busy",
                        "error": "A resume for this team member is already in progress.",
                    }
                )
            try:
                return await function(run_id, follow_up_task)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    return guarded
