"""Strict native history restoration before an agent consumes any inbox message."""

from pathlib import Path
from typing import Any

from fast_agent.mcp.prompts.prompt_load import load_history_into_agent


def restore_history(agent: Any, path: Path) -> None:
    """A missing or invalid resume history aborts startup, never starts fresh."""
    try:
        load_history_into_agent(agent, path)
        if not agent.message_history:
            raise ValueError("Resume history contains no messages")
    finally:
        # Only these private, auto-resume materializations are disposable.
        # User supplied histories and generic fast-agent session files remain.
        if path.name.startswith("db_resume_"):
            path.unlink(missing_ok=True)
