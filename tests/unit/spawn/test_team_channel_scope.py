"""Real-process checks for team/run-scoped wake delivery."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fast_agent.spawn.agent_channel import AgentChannel
from fast_agent.spawn.spawn_registry import SpawnRecord, SpawnRegistry


def _listen_in_child(project_dir: str, session_id: str, run_id: str, pipe) -> None:
    os.environ["SPAWN_PROJECT_DIR"] = project_dir

    async def listen() -> None:
        channel = AgentChannel("Alex [PM]", session_id=session_id, run_id=run_id)
        await channel.start_server()
        pipe.send("ready")
        try:
            pipe.send(await channel.listen(timeout=3))
        finally:
            await channel.stop()
            pipe.close()

    asyncio.run(listen())


def test_same_named_team_members_receive_only_their_own_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two subprocess listeners must not collide or cross-deliver."""
    from fast_agent.spawn.servers import _team_helpers

    monkeypatch.setenv("SPAWN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(tmp_path / "registry.db"))
    registry = SpawnRegistry(tmp_path / "unused.json")
    for session_id in ("team-a", "team-b"):
        registry.register(SpawnRecord(
            run_id=f"run-{session_id}", agent_name="Alex [PM]",
            role="pm", team_name=session_id, session_id=session_id,
            status="idle", original_config={"env_vars": {"TEAM_SESSION_ID": session_id}},
        ))

    context = multiprocessing.get_context("spawn")
    children = []
    for session_id in ("team-a", "team-b"):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_listen_in_child,
            args=(str(tmp_path), session_id, f"run-{session_id}", child),
        )
        process.start()
        child.close()
        children.append((process, parent))

    try:
        assert all(pipe.poll(10) and pipe.recv() == "ready" for _, pipe in children)
        assert _team_helpers.wake_team_agent("team-a", "Alex [PM]", "run-team-a") == "signaled"
        assert children[0][1].poll(5) and children[0][1].recv() == "wake"
        assert not children[1][1].poll(0.2)
        assert _team_helpers.wake_team_agent("team-b", "Alex [PM]", "run-team-b") == "signaled"
        assert children[1][1].poll(5) and children[1][1].recv() == "wake"
        with pytest.raises(LookupError):
            _team_helpers.wake_team_agent("team-b", "Alex [PM]", "run-team-a")
    finally:
        for process, pipe in children:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            pipe.close()
        assert all(process.exitcode == 0 for process, _ in children)


@pytest.mark.asyncio
async def test_old_run_cleanup_cannot_unlink_new_run_socket(tmp_path: Path) -> None:
    old = AgentChannel("Alex [PM]", tmp_path, session_id="team-a", run_id="old")
    new = AgentChannel("Alex [PM]", tmp_path, session_id="team-a", run_id="new")
    await old.start_server()
    await new.start_server()
    try:
        assert old.socket_path != new.socket_path
        await old.stop()
        assert new.socket_path.exists()
        assert AgentChannel.is_alive(
            "Alex [PM]", tmp_path, session_id="team-a", run_id="new",
        )
    finally:
        await old.stop()
        await new.stop()


@pytest.mark.asyncio
async def test_duplicate_run_cannot_displace_live_socket(tmp_path: Path) -> None:
    first = AgentChannel("Alex [PM]", tmp_path, session_id="team-a", run_id="one")
    duplicate = AgentChannel("Alex [PM]", tmp_path, session_id="team-a", run_id="one")
    await first.start_server()
    try:
        with pytest.raises(RuntimeError, match="already active"):
            await duplicate.start_server()
        assert first.socket_path.exists()
    finally:
        await first.stop()


@pytest.mark.asyncio
async def test_launch_claim_survives_more_than_30_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.spawn import isolated_spawner

    record = SpawnRecord(
        run_id="original", agent_name="Alex [PM]", session_id="team-a",
    )
    registry = SimpleNamespace(
        get=lambda _: record,
        _backend=SimpleNamespace(_db_path=str(tmp_path / "registry.db")),
    )
    started = asyncio.Event()
    launches = []

    async def slow_launch(*args, **kwargs):
        launches.append(args[0])
        started.set()
        await asyncio.sleep(31)

    monkeypatch.setattr(isolated_spawner, "_resume_on_inbox_claimed", slow_launch)
    first = asyncio.create_task(isolated_spawner._check_and_resume_on_inbox(
        "original", "Alex [PM]", registry,
    ))
    await asyncio.wait_for(started.wait(), timeout=2)
    await asyncio.sleep(30.1)
    await isolated_spawner._check_and_resume_on_inbox(
        "original", "Alex [PM]", registry,
    )
    await first
    assert launches == ["original"]


@pytest.mark.asyncio
async def test_failed_launch_releases_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.spawn import isolated_spawner

    record = SpawnRecord(
        run_id="original", agent_name="Alex [PM]", session_id="team-a",
    )
    registry = SimpleNamespace(
        get=lambda _: record,
        _backend=SimpleNamespace(_db_path=str(tmp_path / "registry.db")),
    )
    attempts = 0

    async def launch(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("launch failed")

    monkeypatch.setattr(isolated_spawner, "_resume_on_inbox_claimed", launch)
    with pytest.raises(RuntimeError, match="launch failed"):
        await isolated_spawner._check_and_resume_on_inbox(
            "original", "Alex [PM]", registry,
        )
    await isolated_spawner._check_and_resume_on_inbox(
        "original", "Alex [PM]", registry,
    )
    assert attempts == 2
