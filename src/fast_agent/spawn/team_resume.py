"""Resume a team without confusing idle processes with terminal runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

from fast_agent.spawn.agent_channel import AgentChannel
from fast_agent.spawn.message_bus import MessageBus
from fast_agent.spawn.spawn_registry import SpawnRegistry
from fast_agent.spawn.team_spawner import TeamSession

logger = logging.getLogger(__name__)


async def resume_team_members(
    session: TeamSession,
    follow_up_task: str,
    registry: SpawnRegistry,
    resume_agent: Callable[[str, str], Awaitable[str]],
    project_dir: Path,
) -> dict[str, Any]:
    """Restart dead idle/terminal members; deliver to live idle members in place.

    A wake signal is acceptance, not proof of completed work: report ``queued``
    separately from process restarts. Reserved, never-started roles stay reserved.
    """
    results = {}
    resumed = queued = failed = skipped = 0
    for name, info in session.agents.items():
        if info.get("status") == "available":
            results[name] = {"status": "skipped", "reason": "role not started"}
            skipped += 1
            continue
        try:
            record = registry.get_latest(info.get("run_id", ""))
            if not record:
                raise ValueError("Spawn record missing; cannot restore this member")
            if record.agent_name != name:
                raise ValueError("Resume chain points to a different agent")
            if record.session_id and record.session_id != session.session_id:
                raise ValueError("Resume chain points to a different team session")
            if record.status != "idle" and not record.is_terminal:
                results[name] = {"status": "skipped", "reason": f"agent is {record.status}"}
                skipped += 1
                continue
            if record.lifecycle != "resumable":
                raise ValueError(f"Lifecycle {record.lifecycle!r} is not resumable")

            if AgentChannel.is_alive(name):
                cfg = record.original_config or {}
                env = cfg.get("env_vars") or {}
                messages_dir = env.get("TEAM_MESSAGES_DIR") or str(
                    project_dir / ".runtime" / "state" / "messages" / session.session_id
                )
                msg = MessageBus(messages_dir).send(
                    from_name="Jarvis",
                    to_name=name,
                    content=follow_up_task,
                    message_type="directive",
                    context={"session_id": session.session_id},
                )
                try:
                    delivered = AgentChannel.send_signal(name, "wake")
                except Exception:
                    logger.exception("Wake delivery failed for %s", name)
                    delivered = False
                if not delivered:
                    results[name] = {
                        "status": "error",
                        "reason": "Directive saved but wake delivery failed",
                        "message_id": msg.message_id,
                        "message_queued": True,
                    }
                    failed += 1
                    continue
                results[name] = {
                    "status": "queued",
                    "run_id": record.run_id,
                    "message_id": msg.message_id,
                }
                queued += 1
                continue

            result = json.loads(await resume_agent(record.run_id, follow_up_task))
            if result.get("status") != "resumed" or not result.get("new_run_id"):
                results[name] = {"status": "error", "reason": result.get("error", str(result))}
                failed += 1
                continue
            session.update_agent_run_id(name, result["new_run_id"])
            results[name] = {"status": "resumed", "new_run_id": result["new_run_id"]}
            resumed += 1
        except Exception as exc:
            logger.exception("Failed to resume team member %s", name)
            results[name] = {"status": "error", "reason": str(exc)}
            failed += 1

    accepted = resumed + queued
    blocked = any(
        r["status"] == "skipped" and r["reason"] != "role not started" for r in results.values()
    )
    status = (
        ("partial" if failed or blocked else ("resumed" if resumed else "queued"))
        if accepted
        else "not_resumed"
    )
    if accepted:
        session.sprint_status = "running"
    return {
        "status": status,
        "session_id": session.session_id,
        "team_name": session.team_name,
        "resumed_agents": resumed,
        "queued_agents": queued,
        "failed_agents": failed,
        "skipped_agents": skipped,
        "agents": results,
        "message": f"{resumed} restarted, {queued} queued, {failed} failed, {skipped} skipped. "
        "Execution progress is delivered through agent activity events.",
    }


# flock covers independent MCP server processes, not just tasks on one loop.
# Keep lock files after release: unlinking a locked inode lets a concurrent
# caller lock a different inode for the same session.
@contextmanager
def team_resume_lock(project_dir: Path, session_id: str) -> Iterator[None]:
    directory = project_dir / ".runtime" / "state" / "team-resume-locks"
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(session_id.encode()).hexdigest()
    with (directory / name).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
