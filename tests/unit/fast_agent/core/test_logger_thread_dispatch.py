"""Regression tests for MCP stderr logging from a worker thread."""

import asyncio

import pytest

from fast_agent.core.logging.logger import Logger
from fast_agent.core.logging.transport import AsyncEventBus


@pytest.mark.asyncio
async def test_worker_thread_log_uses_running_bus_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = AsyncEventBus.get()
    await bus.start()
    main_loop = asyncio.get_running_loop()
    original_policy = asyncio.get_event_loop_policy()
    received = asyncio.Event()
    emitted_on: list[asyncio.AbstractEventLoop] = []

    async def capture_event(event: object) -> None:
        emitted_on.append(asyncio.get_running_loop())
        received.set()

    monkeypatch.setattr(bus, "emit", capture_event)
    try:
        await asyncio.to_thread(Logger("stderr-test").debug, "MCP stderr line")
        await asyncio.wait_for(received.wait(), timeout=2)
        assert emitted_on == [main_loop]
        assert asyncio.get_event_loop_policy() is original_policy
    finally:
        await bus.stop()
        AsyncEventBus.reset()
