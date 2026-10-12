"""Noninteractive tool subprocesses must not consume the agent control channel."""
import asyncio
import json
import logging
import subprocess
import sys
import textwrap

import pytest
from mcp.types import TextContent

from fast_agent.tools.shell_runtime import ShellRuntime


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX command quoting")
def test_shell_cannot_read_or_consume_parent_control_frames() -> None:
    script = textwrap.dedent('''
        import asyncio, json, logging, shlex, sys
        from mcp.types import TextContent
        from fast_agent.tools.shell_runtime import ShellRuntime
        from fast_agent.ui.display_suppression import suppress_interactive_display

        async def run():
            runtime = ShellRuntime("test", logging.getLogger("stdin-test"))
            runtime._show_bash_output = False
            command = shlex.quote(sys.executable) + " -c " + shlex.quote(
                "import sys; sys.stdout.write(sys.stdin.read())"
            )
            with suppress_interactive_display():
                result = await runtime.execute({"command": command})
            output = "".join(block.text for block in result.content if isinstance(block, TextContent))
            print(json.dumps({"output": output, "control": sys.stdin.readline()}))

        asyncio.run(run())
    ''')
    control_frame = '{"jsonrpc":"2.0","method":"ping","id":24}\n'
    child = subprocess.run(
        [sys.executable, "-c", script],
        input=control_frame, text=True, capture_output=True, timeout=15, check=True,
    )
    result = json.loads(child.stdout.strip().splitlines()[-1])
    assert control_frame not in result["output"]
    assert result["control"] == control_frame


@pytest.mark.asyncio
async def test_explicit_shell_pipeline_still_receives_its_own_input() -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX pipeline syntax")
    runtime = ShellRuntime("test", logging.getLogger(__name__))
    runtime._show_bash_output = False
    result = await asyncio.wait_for(
        runtime.execute({"command": "printf 'pipeline-payload' | cat"}), timeout=10,
    )
    assert not result.isError
    assert "pipeline-payload" in "".join(
        block.text for block in result.content if isinstance(block, TextContent)
    )
