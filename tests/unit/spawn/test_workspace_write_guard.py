from types import SimpleNamespace

import pytest

from fast_agent.spawn.workspace_write_guard import WorkspaceWriteGuard


def request(name, path):
    return SimpleNamespace(
        tool_calls={
            "call": SimpleNamespace(
                params=SimpleNamespace(name="filesystem-" + name, arguments={"path": str(path)})
            )
        }
    )


def test_concurrent_editor_rejected_until_owner_finishes(tmp_path):
    path = tmp_path / "game.js"
    dev = WorkspaceWriteGuard(tmp_path, "Dev")
    qe = WorkspaceWriteGuard(tmp_path, "QE")
    dev.before(request("write_file", path))
    with pytest.raises(RuntimeError, match="Dev"):
        qe.before(request("edit_file", path))
    dev.release()
    qe.before(request("edit_file", path))
    qe.release()


def test_read_then_external_change_requires_reread(tmp_path):
    path = tmp_path / "game.js"
    path.write_text("old")
    guard = WorkspaceWriteGuard(tmp_path, "Dev")
    guard.before(request("read_file", path))
    guard.after(SimpleNamespace(tool_results={"call": SimpleNamespace(isError=False)}))
    path.write_text("newer")
    with pytest.raises(RuntimeError, match="changed since"):
        guard.before(request("edit_file", path))


def test_failed_tool_read_does_not_establish_snapshot(tmp_path):
    guard = WorkspaceWriteGuard(tmp_path, "Dev")
    path = tmp_path / "game.js"
    guard.before(request("read_file", path))
    guard.after(SimpleNamespace(tool_results={"call": SimpleNamespace(isError=True)}))
    path.write_text("created later")
    guard.before(request("write_file", path))
    guard.release()


def test_batch_claims_are_atomic(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    dev, qe = WorkspaceWriteGuard(tmp_path, "Dev"), WorkspaceWriteGuard(tmp_path, "QE")
    dev.before(request("write_file", second))
    batch = request("write_file", first)
    batch.tool_calls["other"] = request("write_file", second).tool_calls["call"]
    with pytest.raises(RuntimeError):
        qe.before(batch)
    assert "a" not in qe.claims()
    dev.release()


def test_dead_writer_reclaimed_without_polling(tmp_path, monkeypatch):
    import json

    from fast_agent.spawn import workspace_write_guard as module

    (tmp_path / ".team-write-claims.json").write_text(
        json.dumps({"game.js": {"owner": "dead", "pid": 999999, "expires": 99999999999}})
    )
    original = module._alive
    monkeypatch.setattr(module, "_alive", lambda pid: False if pid == 999999 else original(pid))
    guard = WorkspaceWriteGuard(tmp_path, "Dev")
    guard.before(request("write_file", tmp_path / "game.js"))
    assert guard.claims()["game.js"]["owner"] == "Dev"
    guard.release()


@pytest.mark.asyncio
async def test_hooks_release_claim_after_failed_tool_without_end_of_turn(tmp_path):
    from fast_agent.spawn.workspace_write_guard import create_workspace_write_hooks

    hooks = create_workspace_write_hooks(tmp_path, "Dev")
    path = tmp_path / "game.js"
    await hooks.before_tool_call(None, request("write_file", path))
    await hooks.after_tool_call(
        None, SimpleNamespace(tool_results={"call": SimpleNamespace(isError=True)})
    )
    qe = WorkspaceWriteGuard(tmp_path, "QE")
    qe.before(request("write_file", path))
    qe.release()


@pytest.mark.asyncio
async def test_real_tool_runner_rejects_collision_before_execution_and_releases(tmp_path):
    from unittest.mock import AsyncMock

    from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

    from fast_agent.agents.tool_runner import ToolRunner
    from fast_agent.spawn.workspace_write_guard import create_workspace_write_hooks
    from fast_agent.types import PromptMessageExtended

    path = tmp_path / "game.js"
    calls = {
        "call": CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name="filesystem-write_file", arguments={"path": str(path), "content": "new"}
            ),
        )
    }
    pending = PromptMessageExtended(role="assistant", content=[], tool_calls=calls)
    other = WorkspaceWriteGuard(tmp_path, "Dev")
    other.before(pending)
    agent = SimpleNamespace(
        name="QE",
        run_tools=AsyncMock(
            return_value=PromptMessageExtended(
                role="user",
                content=[],
                tool_results={
                    "call": CallToolResult(content=[TextContent(type="text", text="written")])
                },
            )
        ),
    )
    runner = ToolRunner(
        agent=agent, messages=[], hooks=create_workspace_write_hooks(tmp_path, "QE")
    )
    runner._pending_tool_request = pending
    result = await runner.generate_tool_call_response()
    assert result.tool_results["call"].isError
    agent.run_tools.assert_not_called()
    other.release()
    runner._pending_tool_request = pending
    runner._pending_tool_response = None
    result = await runner.generate_tool_call_response()
    assert not result.tool_results["call"].isError
    agent.run_tools.assert_awaited_once()
    assert not other.claims()


def _claim_in_process(workspace, connection):
    from pathlib import Path

    guard = WorkspaceWriteGuard(Path(workspace), "Worker")
    guard.before(request("write_file", Path(workspace) / "game.js"))
    connection.send("claimed")
    connection.recv()
    guard.release()
    connection.send("released")


def test_real_process_claim_and_crash_recovery(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_claim_in_process, args=(str(tmp_path), child))
    process.start()
    child.close()
    try:
        assert parent.poll(15) and parent.recv() == "claimed"
        guard = WorkspaceWriteGuard(tmp_path, "QE")
        with pytest.raises(RuntimeError, match="Worker"):
            guard.before(request("write_file", tmp_path / "game.js"))
        process.kill()
        process.join(5)
        guard.before(request("write_file", tmp_path / "game.js"))
        guard.release()
        assert not guard.claims()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        process.close()
        parent.close()


@pytest.mark.parametrize(
    "payload", [b"\x89PNG\r\n\x1a\n", b"%PDF-1.7\n", b"ID3\x04", b"PK\x03\x04", b"bad\x00text"]
)
def test_binary_text_read_is_rejected_before_mcp(tmp_path, payload):
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    guard = WorkspaceWriteGuard(tmp_path, "QE")
    with pytest.raises(RuntimeError, match="read_media_file"):
        guard.before(request("read_text_file", path))
    guard.before(request("read_media_file", path))


def test_utf8_multibyte_at_sample_boundary_is_not_binary(tmp_path):
    path = tmp_path / "chapter.txt"
    path.write_text("a" * 4095 + "ế tiếng Việt")
    WorkspaceWriteGuard(tmp_path, "QE").before(request("read_text_file", path))


def test_mixed_multiple_read_rejects_binary(tmp_path):
    text, image = tmp_path / "code.js", tmp_path / "screen.png"
    text.write_text("const result = true;")
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    batch = request("read_multiple_files", text)
    batch.tool_calls["call"].params.arguments = {"paths": [str(text), str(image)]}
    with pytest.raises(RuntimeError, match="read_media_file"):
        WorkspaceWriteGuard(tmp_path, "QE").before(batch)
