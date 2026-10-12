"""Read Jarvis conversation snapshots without importing its application package."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def load_latest_context_json(agent_name: str, *, session_id: str | None = None) -> str | None:
    """Return the newest persisted history for one agent in one team session.

    The registry database path is shared with the child runner. A read-only
    connection prevents an absent or misconfigured database from being created.
    """
    db_path = os.environ.get("SPAWN_REGISTRY_DB")
    if not db_path:
        logger.error("Cannot resume %s: SPAWN_REGISTRY_DB is not set", agent_name)
        return None

    try:
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=10) as connection:
            row = _read_latest_snapshot(connection, agent_name, session_id)
        return row[0] if row else None
    except sqlite3.Error:
        logger.exception("Cannot load conversation snapshot for %s", agent_name)
        return None


def materialize_resume_history(agent_name: str, *, session_id: str, db_path: str) -> str:
    """Create a private native history file from a strictly team-owned snapshot."""
    import json
    import tempfile

    if not session_id:
        raise ValueError("Team session identity is required for automatic resume")
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=10) as connection:
            row = _read_latest_snapshot(connection, agent_name, session_id)
    except sqlite3.Error as exc:
        raise ValueError("Cannot read authoritative resume snapshot") from exc
    if not row:
        raise ValueError("No owned conversation snapshot")
    try:
        history = json.loads(row[0])
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid conversation snapshot JSON") from exc
    if not isinstance(history, dict) or not history.get("messages"):
        raise ValueError("Conversation snapshot must contain native messages")
    from fast_agent.mcp.prompt_serialization import from_json

    try:
        from_json(row[0])
    except Exception as exc:
        raise ValueError("Invalid native conversation snapshot") from exc
    fd, path = tempfile.mkstemp(prefix="db_resume_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(row[0])
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    return path


def _read_latest_snapshot(connection: sqlite3.Connection, agent_name: str, session_id: str | None):
    """Newest owned raw or completed working snapshot; never cross team scope."""
    scope = " AND session_id = ?" if session_id else ""
    parameters = [agent_name, session_id] if session_id else [agent_name]
    raw = connection.execute(
        "SELECT context_json, created_at FROM agent_context_snapshots WHERE agent_name = ?"
        + scope
        + " ORDER BY created_at DESC, id DESC LIMIT 1",
        parameters,
    ).fetchone()
    has_working = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='context_compaction_events'"
    ).fetchone()
    working = None
    if has_working:
        working = connection.execute(
            "SELECT working_context_json, created_at FROM context_compaction_events "
            "WHERE agent_name = ? AND status = 'completed' AND working_context_json IS NOT NULL"
            + scope
            + " ORDER BY created_at DESC, id DESC LIMIT 1",
            parameters,
        ).fetchone()
    return max((row for row in (raw, working) if row), key=lambda row: row[1], default=None)
