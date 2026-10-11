"""Cancellation from a controller which does not own the worker's asyncio task."""
import asyncio
import json
import sys

import psutil
import pytest

from fast_agent.spawn.isolated_spawner import cancel_spawn
from fast_agent.spawn.registry_backends import SqliteBackend
from fast_agent.spawn.spawn_registry import SpawnRecord, SpawnRegistry


@pytest.mark.asyncio
async def test_registry_only_cancel_stops_owned_worker(tmp_path, monkeypatch):
    monkeypatch.setenv('SPAWN_REGISTRY_DB', str(tmp_path / 'registry.db'))
    registry = SpawnRegistry(tmp_path / 'registry.json')
    # Extra argv models the production uv launcher without starting an LLM.
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-c', 'import time;time.sleep(120)',
        '-m', 'fast_agent.spawn.isolated_runner', '--config', str(tmp_path / 'run_remote.json'),
        '--project-dir', str(tmp_path),
    )
    registry.register(SpawnRecord(
        run_id='remote', pid=process.pid, lifecycle='resumable',
        metadata={'spawn_pid_birth': psutil.Process(process.pid).create_time()},
    ))
    try:
        assert await cancel_spawn('remote', registry)
        await asyncio.wait_for(process.wait(), 5)
        assert registry.get('remote').status == 'cancelled'
        registry.update_status('remote', 'error', error='late exit')
        assert registry.get('remote').status == 'cancelled'
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


@pytest.mark.asyncio
async def test_wrong_registered_birth_never_signals_process(tmp_path):
    registry = SpawnRegistry(tmp_path / 'registry.json')
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', 'import time;time.sleep(120)')
    registry.register(SpawnRecord(run_id='wrong', pid=process.pid, metadata={'spawn_pid_birth': 1}))
    try:
        assert not await cancel_spawn('wrong', registry)
        assert process.returncode is None
    finally:
        process.kill()
        await process.wait()


def test_stale_sqlite_writer_preserves_cancelled_terminal_state(tmp_path):
    backend = SqliteBackend(tmp_path / 'registry.db')
    backend.save({'run': {'run_id': 'run', 'status': 'idle', 'metadata': {'note': 'kept'}}})
    stale = backend.load()
    backend.save({'run': {'run_id': 'run', 'status': 'cancelled', 'completed_at': 123}})
    stale['run']['status'] = 'error'
    backend.save(stale)
    record = backend.load()['run']
    assert record['status'] == 'cancelled'
    assert record['completed_at'] == 123
    assert json.dumps(record['metadata']) == '{"note": "kept"}'
