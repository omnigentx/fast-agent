"""Integration test: a stdio MCP server that crashes at startup must surface its
subprocess stderr (the real cause) — not just the parent-side "Connection closed".

Regression for the LoggerTextIO.fileno()->/dev/null bug, where the child's stderr
was silently discarded and never reached record_stdio_stderr()/the UI.
"""

from __future__ import annotations

import sys

import pytest

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry


@pytest.mark.asyncio
async def test_failed_stdio_server_surfaces_subprocess_stderr(tmp_path) -> None:
    # A server script that crashes immediately with a ModuleNotFoundError.
    script = tmp_path / "broken_server.py"
    script.write_text("import a_module_that_does_not_exist_xyz\n")

    cfg = MCPServerSettings(
        name="demo",
        transport="stdio",
        command=sys.executable,
        args=[str(script)],
    )
    registry = ServerRegistry()
    registry.registry = {"demo": cfg}
    context = Context(server_registry=registry)

    aggregator = MCPAggregator(
        server_names=["demo"], connection_persistence=True, context=context
    )
    async with aggregator:
        await aggregator.load_servers()
        status_map = await aggregator.collect_server_status()

    status = status_map["demo"]
    assert status.is_connected is not True
    assert status.error_message is not None
    # The captured subprocess stderr must carry the real cause.
    assert "Recent stderr from stdio server:" in status.error_message
    assert "ModuleNotFoundError" in status.error_message
    assert "a_module_that_does_not_exist_xyz" in status.error_message
