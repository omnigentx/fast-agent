"""The mid-turn hook must read the same session inbox as team writers."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mcp.types import TextContent

from fast_agent.spawn.inbox_watcher_hook import create_inbox_watcher
from fast_agent.spawn.message_bus import MessageBus


@pytest.mark.asyncio
async def test_session_message_reaches_next_model_step_without_legacy_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = tmp_path / "state" / "messages" / "session-a"
    bus = MessageBus(messages)
    bus.send("Dashboard", "Dev", "NEW_REQUIREMENT_RUNTIME")
    monkeypatch.setenv("TEAM_MY_NAME", "Dev")
    monkeypatch.setenv("TEAM_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("TEAM_MESSAGES_DIR", str(messages))
    watcher = create_inbox_watcher()
    assert watcher is not None
    runner = MagicMock()
    await watcher.before_llm_call(runner, [])
    prompt = runner.append_messages.call_args.args[0]
    assert "NEW_REQUIREMENT_RUNTIME" in "".join(
        block.text for block in prompt.content if isinstance(block, TextContent)
    )


def test_explicit_missing_session_does_not_fall_back_to_other_inbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".messages").mkdir()
    monkeypatch.setenv("TEAM_MY_NAME", "Dev")
    monkeypatch.setenv("TEAM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TEAM_MESSAGES_DIR", str(tmp_path / "missing-session"))
    assert create_inbox_watcher() is None


def test_legacy_workspace_inbox_remains_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".messages").mkdir()
    monkeypatch.setenv("TEAM_MY_NAME", "Dev")
    monkeypatch.setenv("TEAM_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("TEAM_MESSAGES_DIR", raising=False)
    assert create_inbox_watcher() is not None
