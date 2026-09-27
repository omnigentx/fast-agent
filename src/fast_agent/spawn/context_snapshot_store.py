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
            query = (
                "SELECT context_json FROM agent_context_snapshots "
                "WHERE agent_name = ?"
            )
            params = [agent_name]
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            query += " ORDER BY created_at DESC, id DESC LIMIT 1"
            row = connection.execute(query, params).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        logger.exception("Cannot load conversation snapshot for %s", agent_name)
        return None
