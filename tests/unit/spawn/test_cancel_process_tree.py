"""Exercise cancellation with a real launcher and an independent child session."""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import psutil
import pytest

from fast_agent.spawn import isolated_spawner as spawner
from fast_agent.spawn.spawn_registry import SpawnRegistry


@pytest.mark.asyncio
async def test_cancel_stops_child_in_separate_session(tmp_path):
    child_file = tmp_path / 'child.pid'
    code = (
        'import subprocess,sys,time; '
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'], "
        'start_new_session=True); '
        f"open({str(child_file)!r},'w').write(str(p.pid)); "
        'time.sleep(120)'
    )
    launcher = await asyncio.create_subprocess_exec(sys.executable, '-c', code)
    child = None
    try:
        async def wait_for_child():
            while not child_file.exists():
                await asyncio.sleep(.01)
        await asyncio.wait_for(wait_for_child(), 5)
        child = psutil.Process(int(child_file.read_text()))
        spawner._background_processes['tree-test'] = launcher
        assert await spawner.cancel_spawn('tree-test')
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert launcher.returncode is not None
    finally:
        spawner._background_processes.pop('tree-test', None)
        if child is not None and child.is_running():
            child.kill()
        if launcher.returncode is None:
            launcher.kill()
        await launcher.wait()


@pytest.mark.asyncio
async def test_explicit_cancel_does_not_become_idle(monkeypatch, tmp_path):
    started = asyncio.Event()

    async def wait_forever(**kwargs):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(spawner, 'run_isolated_agent', wait_forever)
    registry = SpawnRegistry(tmp_path / 'registry.json')
    hooks = SimpleNamespace(on_registered=AsyncMock(), on_cancelled=AsyncMock())
    run_id = await spawner.run_isolated_agent_background(
        task='work', project_dir=tmp_path, lifecycle='resumable',
        agent_name='Dev', registry=registry, spawn_lifecycle_hooks=hooks,
    )
    await started.wait()
    task = spawner._background_tasks[run_id]
    assert await spawner.cancel_spawn(run_id, registry, hooks)
    await task
    assert registry.get(run_id).status == 'cancelled'
    hooks.on_cancelled.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_detach_remains_idle(monkeypatch, tmp_path):
    started = asyncio.Event()

    async def wait_forever(**kwargs):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(spawner, 'run_isolated_agent', wait_forever)
    registry = SpawnRegistry(tmp_path / 'registry.json')
    run_id = await spawner.run_isolated_agent_background(
        task='work', project_dir=tmp_path, lifecycle='resumable',
        agent_name='Dev', registry=registry,
    )
    await started.wait()
    task = spawner._background_tasks[run_id]
    task.cancel()
    await task
    assert registry.get(run_id).status == 'idle'


@pytest.mark.asyncio
async def test_escalation_does_not_kill_unrelated_process(tmp_path):
    from fast_agent.spawn.process_tree import SpawnProcessTree

    unrelated = await asyncio.create_subprocess_exec(
        sys.executable, '-c', 'import time;time.sleep(120)'
    )
    stubborn = await asyncio.create_subprocess_exec(
        sys.executable, '-c',
        'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(120)',
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert await stubborn.stdout.readline() == b'ready\n'
        tree = SpawnProcessTree(stubborn.pid)
        await asyncio.to_thread(tree.terminate, .05)
        await asyncio.wait_for(stubborn.wait(), 2)
        assert stubborn.returncode != 0
        assert unrelated.returncode is None
    finally:
        for process in [stubborn, unrelated]:
            if process.returncode is None:
                process.kill()
            await process.wait()


def test_reused_root_identity_is_never_signalled(monkeypatch):
    from fast_agent.spawn.process_tree import SpawnProcessTree

    tree = SpawnProcessTree(__import__('os').getpid())
    root = SimpleNamespace(is_running=lambda: False, children=lambda **kw: pytest.fail('PID reused'))
    tree.root = root
    tree.terminate()
