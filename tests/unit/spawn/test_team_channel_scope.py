"""Real-process checks for team/run-scoped wake delivery."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import time
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


@pytest.mark.asyncio
async def test_sigkilled_running_record_is_rescheduled_after_registry_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.spawn import isolated_spawner
    from fast_agent.spawn.servers import _team_helpers

    monkeypatch.setenv("SPAWN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(tmp_path / "registry.db"))
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        registry = SpawnRegistry(tmp_path / "unused.json")
        registry.register(SpawnRecord(
            run_id="dead-run", agent_name="Alex [PM]", role="pm",
            team_name="team-a", session_id="team-a", status="running",
            pid=sleeper.pid, started_at=time.time() - 180,
            original_config={"env_vars": {"TEAM_SESSION_ID": "team-a"}},
        ))
        sleeper.kill()
        sleeper.wait(timeout=5)
        # A new registry object represents a restarted backend/MCP process.
        restarted = SpawnRegistry(tmp_path / "unused.json")
        assert not restarted.has_running_resume(
            "Alex [PM]", "team-a", verify_process=True,
        )
        scheduled = asyncio.Event()

        async def resume(**kwargs):
            assert kwargs["run_id"] == "dead-run"
            scheduled.set()

        monkeypatch.setattr(isolated_spawner, "_check_and_resume_on_inbox", resume)
        assert _team_helpers.wake_team_agent(
            "team-a", "Alex [PM]", "dead-run",
        ) == "scheduled"
        await asyncio.wait_for(scheduled.wait(), timeout=2)
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait(timeout=5)


@pytest.mark.asyncio
async def test_scheduled_resume_failure_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog,
) -> None:
    from fast_agent.spawn import isolated_spawner
    from fast_agent.spawn.servers import _team_helpers

    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(tmp_path / "registry.db"))
    registry = SpawnRegistry(tmp_path / "unused.json")
    registry.register(SpawnRecord(
        run_id="idle-run", agent_name="Alex [PM]", role="pm",
        team_name="team-a", session_id="team-a", status="idle",
    ))

    async def failed_resume(**kwargs):
        raise RuntimeError("inbox unavailable")

    monkeypatch.setattr(isolated_spawner, "_check_and_resume_on_inbox", failed_resume)
    assert _team_helpers.wake_team_agent(
        "team-a", "Alex [PM]", "idle-run",
    ) == "scheduled"
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert "Scoped resume failed" in caplog.text


def test_running_pid_must_belong_to_the_recorded_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(tmp_path / "registry.db"))
    process = subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(60)", "run_live.json",
    ])
    try:
        registry = SpawnRegistry(tmp_path / "unused.json")
        registry.register(SpawnRecord(
            run_id="live", agent_name="Alex [PM]", role="pm",
            team_name="team-a", session_id="team-a", status="running",
            pid=process.pid, started_at=time.time() - 180,
        ))
        assert registry.has_running_resume(
            "Alex [PM]", "team-a", verify_process=True,
        )
        registry.register(SpawnRecord(
            run_id="wrong", agent_name="Blair [PM]", role="pm",
            team_name="team-b", session_id="team-b", status="running",
            pid=process.pid, started_at=time.time() - 180,
        ))
        assert not registry.has_running_resume(
            "Blair [PM]", "team-b", verify_process=True,
        )
    finally:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.asyncio
async def test_restart_recovers_fresh_running_record_without_child_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One startup wake must recover a launch abandoned before PID storage."""
    from fast_agent.spawn import isolated_spawner
    from fast_agent.spawn.message_bus import MessageBus
    from fast_agent.spawn.servers import _team_helpers
    from fast_agent.spawn.spawn_registry import _process_birth

    monkeypatch.setenv("SPAWN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(tmp_path / "registry.db"))
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        registry = SpawnRegistry(tmp_path / "unused.json")
        messages_dir = tmp_path / ".runtime" / "state" / "messages" / "team-a"
        bus = MessageBus(messages_dir)
        bus.send("Jarvis", "Alex [PM]", "Retry this revision")
        registry.register(SpawnRecord(
            run_id="orphan", agent_name="Alex [PM]", role="pm",
            team_name="team-a", session_id="team-a", status="running",
            pid=None, started_at=time.time(),
            metadata={"launch_owner_pid": owner.pid,
                      "launch_owner_birth": _process_birth(owner.pid)},
            original_config={
                "role": "pm", "team_name": "team-a",
                "project_dir": str(tmp_path), "task": "initial task",
                "env_vars": {"TEAM_SESSION_ID": "team-a",
                             "TEAM_MESSAGES_DIR": str(messages_dir)},
            },
        ))
        assert registry.has_running_resume(
            "Alex [PM]", "team-a", verify_process=True,
        )
        registry._load()
        correct_birth = registry._data["orphan"]["metadata"]["launch_owner_birth"]
        registry._data["orphan"]["metadata"]["launch_owner_birth"] = "different process"
        registry._save()
        assert not registry.has_running_resume(
            "Alex [PM]", "team-a", verify_process=True,
        )
        registry._load()
        registry._data["orphan"]["metadata"]["launch_owner_birth"] = correct_birth
        registry._save()
        registry._data["orphan"]["started_at"] = time.time() - 121
        registry._save()
        assert not registry.has_running_resume(
            "Alex [PM]", "team-a", verify_process=True,
        )
        registry._load()
        registry._data["orphan"]["started_at"] = time.time()
        registry._save()
        owner.kill()
        owner.wait(timeout=5)
        restarted = SpawnRegistry(tmp_path / "unused.json")
        assert not restarted.has_running_resume(
            "Alex [PM]", "team-a", verify_process=True,
        )
        scheduled = asyncio.Event()

        async def background_spawn(**kwargs):
            assert kwargs["session_id"] == "team-a"
            assert kwargs["agent_name"] == "Alex [PM]"
            scheduled.set()
            return "new-run"

        monkeypatch.setattr(isolated_spawner, "run_isolated_agent_background", background_spawn)
        assert _team_helpers.wake_team_agent(
            "team-a", "Alex [PM]", "orphan",
        ) == "scheduled"
        await asyncio.wait_for(scheduled.wait(), timeout=2)
        assert len(bus.read_unread("Alex [PM]")) == 1
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def test_invalid_registry_path_is_not_created_or_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.spawn.servers._team_helpers import get_project_registry

    missing = tmp_path / "missing.db"
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(missing))
    assert get_project_registry() is None
    assert not missing.exists()

    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(empty))
    assert get_project_registry() is None

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a sqlite database")
    monkeypatch.setenv("SPAWN_REGISTRY_DB", str(corrupt))
    assert get_project_registry() is None
