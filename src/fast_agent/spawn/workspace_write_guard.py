"""Coordinate filesystem MCP writes in a trusted shared team workspace.

Claims cover a tool batch, are reclaimed after a crashed runner, and never require
polling. This is coordination, not an OS sandbox for trusted shell commands.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_READS = {"read_file", "read_text_file", "read_multiple_files"}
_WRITES = {"write_file", "edit_file", "move_file", "delete_file"}


def _operation(name: str) -> str | None:
    for operation in _READS | _WRITES:
        if name == operation or any(
            name.endswith(separator + operation) for separator in ("-", "_", ".", "/")
        ):
            return operation
    return None


def _version(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class WorkspaceWriteGuard:
    def __init__(self, workspace: Path, owner: str):
        self.root = Path(workspace).resolve()
        self.owner = owner
        self._reads: dict[Path, Any] = {}
        self._pending: dict[str, tuple[str, list[Path], list[Any]]] = {}
        self._state = self.root / ".team-write-claims.json"
        self._lock = self.root / ".team-write-claims.lock"

    @contextmanager
    def _transaction(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = json.loads(self._state.read_text()) if self._state.exists() else {}
                # Reclaim a crashed writer on the next access, without a timer.
                # Never expire a live writer merely because its LLM is slow.
                data = {key: value for key, value in data.items() if _alive(value["pid"])}
                yield data
                temporary = self._state.with_suffix(".tmp")
                temporary.write_text(json.dumps(data))
                temporary.replace(self._state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def claims(self) -> dict:
        with self._transaction() as data:
            return dict(data)

    def _paths(self, operation: str, arguments: dict) -> list[Path]:
        values = (
            arguments.get("paths", [])
            if operation == "read_multiple_files"
            else [arguments.get("path")]
        )
        if operation == "move_file":
            values = [arguments.get("source"), arguments.get("destination")]
        result = []
        for value in values:
            if not isinstance(value, str):
                continue
            path = Path(value)
            path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
            if path.is_relative_to(self.root):
                result.append(path)
        return result

    def before(self, message) -> None:
        self._pending = {}
        writes = set()
        for call_id, call in (message.tool_calls or {}).items():
            operation = _operation(call.params.name)
            if operation is None:
                continue
            paths = self._paths(operation, call.params.arguments or {})
            self._pending[call_id] = (operation, paths, [_version(path) for path in paths])
            if operation in _WRITES:
                writes.update(paths)
        if not writes:
            return
        with self._transaction() as claims:
            for path in writes:
                if path in self._reads and self._reads[path] != _version(path):
                    raise RuntimeError(
                        f"{path.relative_to(self.root)} changed since your last read; read it again before editing"
                    )
                key = str(path.relative_to(self.root))
                for existing, claim in claims.items():
                    other = self.root / existing
                    if (claim["owner"] != self.owner or claim["pid"] != os.getpid()) and (
                        path == other or path.is_relative_to(other) or other.is_relative_to(path)
                    ):
                        raise RuntimeError(
                            f"{key} is being edited by {claim['owner']}; wait for their handoff, then read the file again"
                        )
            for path in writes:
                claims[str(path.relative_to(self.root))] = {"owner": self.owner, "pid": os.getpid()}

    def after(self, message) -> None:
        for call_id, result in (message.tool_results or {}).items():
            if result.isError or call_id not in self._pending:
                continue
            operation, paths, before = self._pending[call_id]
            for path, old in zip(paths, before):
                current = _version(path)
                if operation in _WRITES or current == old:
                    self._reads[path] = current
                else:
                    # A concurrent writer raced the read; force a fresh read.
                    self._reads[path] = old
        self._pending = {}

    def release(self) -> None:
        with self._transaction() as claims:
            for key in list(claims):
                if claims[key]["owner"] == self.owner and claims[key]["pid"] == os.getpid():
                    del claims[key]


def create_workspace_write_hooks(workspace: Path, owner: str):
    from fast_agent.agents.tool_runner import ToolRunnerHooks

    guard = WorkspaceWriteGuard(workspace, owner)

    async def before(_runner, message):
        guard.before(message)

    async def after(_runner, message):
        try:
            guard.after(message)
        finally:
            guard.release()

    async def complete(_runner, _message):
        guard.release()

    return ToolRunnerHooks(
        before_tool_call=before, after_tool_call=after, after_turn_complete=complete
    )
