import asyncio
import sys
import time
from unittest.mock import Mock

import pytest

from fast_agent.spawn import isolated_spawner as spawner
from fast_agent.spawn.spawn_events import SpawnEvent


@pytest.mark.asyncio
async def test_display_failure_keeps_draining_child_events(monkeypatch, tmp_path, caplog):
    create = asyncio.create_subprocess_exec
    events = [
        SpawnEvent("thinking", "diag", "QE").to_json_line(),
        SpawnEvent("result", "diag", "QE").to_json_line(),
    ]
    code = "import sys,time;time.sleep(.05);" + ";".join(
        f"print({line!r},file=sys.stderr,flush=True)" for line in events
    )

    async def launch(*args, **kwargs):
        return await create(sys.executable, "-c", code, **kwargs)

    monkeypatch.setattr(spawner.asyncio, "create_subprocess_exec", launch)
    display = Mock()
    display.handle_event.side_effect = [RuntimeError("display failed"), None]
    result = await spawner._run_subprocess(
        "diag",
        {"role": "QE"},
        "unused",
        str(tmp_path / "result.json"),
        5,
        time.time(),
        tmp_path,
        display_manager=display,
    )
    assert result["status"] == "completed"
    assert display.handle_event.call_count == 2
    assert "display failed" in caplog.text
    assert "NameError" not in caplog.text


@pytest.mark.asyncio
async def test_child_failure_returns_bounded_recent_stderr(monkeypatch, tmp_path):
    create = asyncio.create_subprocess_exec
    code = (
        "import sys,time;time.sleep(.05);"
        '[print("diagnostic-%d"%i,file=sys.stderr) for i in range(3000)];'
        "sys.exit(3)"
    )

    async def launch(*args, **kwargs):
        return await create(sys.executable, "-c", code, **kwargs)

    monkeypatch.setattr(spawner.asyncio, "create_subprocess_exec", launch)
    result = await spawner._run_subprocess(
        "diag", {"role": "QE"}, "unused", str(tmp_path / "result.json"), 5, time.time(), tmp_path
    )
    assert result["status"] == "error"
    assert "diagnostic-2999" in result["error"]
    assert "diagnostic-0\n" not in result["error"]
    assert len(result["error"].splitlines()) <= 512
