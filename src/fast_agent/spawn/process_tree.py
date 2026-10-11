"""Bounded cleanup for owned launchers, runners and their MCP descendants."""
from __future__ import annotations

import time

import psutil


class SpawnProcessTree:
    """Keep a birth-checked root handle; never signal a reused PID.

    Descendants are captured before signalling the launcher, including MCP
    children which created their own OS sessions. Process groups alone cannot
    stop those children. psutil checks PID identity again on each signal.
    """

    def __init__(self, pid: int) -> None:
        self.root = psutil.Process(pid)
        self.root.create_time()

    def terminate(self, grace_seconds: float = 3.0) -> None:
        if not self.root.is_running():
            return
        try:
            descendants = self.root.children(recursive=True)
        except psutil.NoSuchProcess:
            return
        processes = [self.root, *descendants]
        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        # The asyncio child watcher owns waitpid(root); waiting for the root
        # here would steal its exit status. Grandchildren are safe to wait on.
        deadline = time.monotonic() + grace_seconds
        _, alive = psutil.wait_procs(descendants, timeout=grace_seconds)
        if self.root.is_running() and self.root.status() != psutil.STATUS_ZOMBIE:
            time.sleep(max(0.0, deadline - time.monotonic()))
        if self.root.is_running() and self.root.status() != psutil.STATUS_ZOMBIE:
            alive.append(self.root)
        # Reparented zombies cannot execute work and must be reaped by their
        # parent/init. Do not mistake them for running workers.
        alive = [p for p in alive if p.status() != psutil.STATUS_ZOMBIE]
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(
            [p for p in alive if p.pid != self.root.pid], timeout=grace_seconds
        )
        survivors = [p.pid for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
        if survivors:
            raise RuntimeError(f'Spawn cleanup did not stop owned processes: {survivors}')
