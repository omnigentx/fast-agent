from __future__ import annotations

from types import SimpleNamespace

import pytest
from anyio import Lock

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry


def _build_context(configs: dict[str, MCPServerSettings]) -> Context:
    registry = ServerRegistry()
    registry.registry = configs
    return Context(server_registry=registry)


class _SessionStub:
    def __init__(self) -> None:
        self.client_info = SimpleNamespace(name="fast-agent-mcp", version="1.0.0")
        self.effective_elicitation_mode = "none"
        self.experimental_session_supported = True
        self.experimental_session_features = ("create", "delete")
        self.experimental_session_cookie = {
            "sessionId": "sess-cookie-1",
            "state": "state-1",
        }
        self.experimental_session_title = None


class _ServerConnStub:
    def __init__(self, config: MCPServerSettings) -> None:
        self.server_implementation = SimpleNamespace(name="demo-server", version="0.1.0")
        self.server_capabilities = None
        self.client_capabilities = {"experimental": {"experimental/sessions": {}}}
        self.session = _SessionStub()
        self._initialized_event = SimpleNamespace(is_set=lambda: True)
        self._error_message = None
        self.server_instructions_available = False
        self.server_instructions_enabled = True
        self.server_instructions = None
        self.server_config = config
        self.session_id = "local"
        self._get_session_id_cb = None
        self.transport_metrics = None
        self._ping_ok_count = 0
        self._ping_fail_count = 0
        self._ping_consecutive_failures = 0
        self._ping_last_ok_at = None
        self._ping_last_fail_at = None
        self._ping_last_error = None

    def is_healthy(self) -> bool:
        return True

    def build_ping_activity_buckets(self, _bucket_seconds: int, bucket_count: int) -> list[str]:
        return ["none"] * bucket_count


class _ManagerStub:
    def __init__(self, server_conn: _ServerConnStub) -> None:
        self._lock = Lock()
        self.running_servers = {"demo": server_conn}


@pytest.mark.asyncio
async def test_collect_server_status_includes_experimental_session_cookie() -> None:
    config = MCPServerSettings(name="demo", transport="stdio", command="echo")
    context = _build_context({"demo": config})

    aggregator = MCPAggregator(
        server_names=["demo"],
        connection_persistence=True,
        context=context,
    )
    aggregator.initialized = True

    server_conn = _ServerConnStub(config)
    manager = _ManagerStub(server_conn)
    setattr(aggregator, "_persistent_connection_manager", manager)

    status_map = await aggregator.collect_server_status()
    status = status_map["demo"]

    assert status.experimental_session_supported is True
    assert status.experimental_session_features == ["create", "delete"]
    assert status.session_cookie == {
        "sessionId": "sess-cookie-1",
        "state": "state-1",
    }
    assert status.session_title is None
    assert status.session_id == "local"


class _FailedServerConnStub(_ServerConnStub):
    """A stdio server that crashed at startup: parent side only saw
    'Connection closed', but the subprocess wrote a real traceback to stderr."""

    def __init__(self, config: MCPServerSettings) -> None:
        super().__init__(config)
        # Parent-side error is generic and unhelpful on its own.
        self._error_message = "McpError: Connection closed"
        self._stderr = (
            "Traceback (most recent call last):",
            '  File "tools/story_server.py", line 10, in <module>',
            "    from helpers.http_safety import get_capped_text",
            "ModuleNotFoundError: No module named 'helpers'",
        )

    def is_healthy(self) -> bool:
        return False

    def recent_stdio_stderr_lines(self) -> tuple[str, ...]:
        return self._stderr


@pytest.mark.asyncio
async def test_collect_server_status_surfaces_stdio_stderr_for_failed_server() -> None:
    config = MCPServerSettings(name="demo", transport="stdio", command="python")
    context = _build_context({"demo": config})

    aggregator = MCPAggregator(
        server_names=["demo"],
        connection_persistence=True,
        context=context,
    )
    aggregator.initialized = True

    server_conn = _FailedServerConnStub(config)
    manager = _ManagerStub(server_conn)
    setattr(aggregator, "_persistent_connection_manager", manager)

    status_map = await aggregator.collect_server_status()
    status = status_map["demo"]

    assert status.is_connected is False
    assert status.error_message is not None
    # The generic parent-side error is preserved...
    assert "Connection closed" in status.error_message
    # ...and the real subprocess cause is now surfaced for debugging.
    assert "Recent stderr from stdio server:" in status.error_message
    assert "ModuleNotFoundError: No module named 'helpers'" in status.error_message


@pytest.mark.asyncio
async def test_collect_server_status_falls_back_to_captured_attach_error() -> None:
    """The real failure mode: a stdio server crashes at startup, its connection is
    dropped from running_servers, but load_servers() captured the cause. The UI must
    still see it instead of "No error message reported"."""
    config = MCPServerSettings(name="demo", transport="stdio", command="python")
    context = _build_context({"demo": config})

    aggregator = MCPAggregator(
        server_names=["demo"],
        connection_persistence=True,
        context=context,
    )
    aggregator.initialized = True
    aggregator._server_attach_errors = {
        "demo": (
            "MCP Server: 'demo': Failed to initialize - see details.\n\n"
            "Recent stderr from stdio server:\n"
            "  ModuleNotFoundError: No module named 'helpers'"
        )
    }

    # running_servers is empty — the conn was dropped on failure.
    manager = _ManagerStub.__new__(_ManagerStub)
    manager._lock = Lock()
    manager.running_servers = {}
    setattr(aggregator, "_persistent_connection_manager", manager)

    status_map = await aggregator.collect_server_status()
    status = status_map["demo"]

    assert status.is_connected is False
    assert status.error_message is not None
    assert "ModuleNotFoundError: No module named 'helpers'" in status.error_message
