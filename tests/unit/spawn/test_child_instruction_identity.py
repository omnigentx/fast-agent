"""A resumed role template must reach the child runtime with its real name."""

import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace

import pytest

from fast_agent.spawn import isolated_runner


@pytest.mark.asyncio
async def test_child_runtime_resolves_saved_identity_template(monkeypatch, tmp_path):
    captured = {}

    class FakeAgent:
        def __init__(self, *_args):
            pass

        def agent(self, **kwargs):
            captured.update(kwargs)
            return lambda fn: fn

        def run(self):
            class Context:
                async def __aenter__(self):
                    return SimpleNamespace(send=self.send)

                async def __aexit__(self, *_args):
                    return None

                async def send(self, _task):
                    return "done"

            return Context()

    fake_module = ModuleType("fast_agent")
    fake_module.FastAgent = FakeAgent
    fake_config = ModuleType("fast_agent.spawn.config_reader")
    fake_config.get_skills = lambda *_args: []
    monkeypatch.setitem(sys.modules, "fast_agent", fake_module)
    monkeypatch.setitem(sys.modules, "fast_agent.spawn.config_reader", fake_config)
    monkeypatch.setattr(isolated_runner, "create_child_config", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(isolated_runner, "emit_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(isolated_runner, "_install_tool_hooks", lambda *_args: None)

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(isolated_runner, "_save_agent_context_snapshot", noop)
    monkeypatch.setattr(isolated_runner, "_emit_mcp_status", noop)
    monkeypatch.setattr(isolated_runner, "_emit_runtime_config", lambda *_args: None)
    monkeypatch.delenv("TEAM_WORKSPACE", raising=False)
    monkeypatch.setenv("TEAM_MY_NAME", "Taylor [Dev]")
    result = await isolated_runner.run_child_agent({
        "task": "Continue work",
        "instruction": 'Your name is "{agent_name}". Send as {agent_name}.',
        "agent_name": "Taylor [Dev]",
        "servers": [],
    }, tmp_path)

    assert result["status"] == "completed"
    assert captured["name"] == "Taylor [Dev]"
    assert 'Your name is "Taylor [Dev]"' in captured["instruction"]
    assert "{agent_name}" not in captured["instruction"]


@pytest.mark.asyncio
async def test_cc_only_inbox_does_not_start_another_llm_turn(monkeypatch, tmp_path):
    sends = []
    done = []
    cc = SimpleNamespace(
        message_id="cc-1", message_type="notification",
        from_name="Bennett [PM]", content="FYI only",
    )

    class FakeAgent:
        def __init__(self, *_args):
            pass

        def agent(self, **_kwargs):
            return lambda fn: fn

        @asynccontextmanager
        async def run(self):
            async def send(task):
                sends.append(task)
                return "done"
            yield SimpleNamespace(send=send)

    class FakeChannel:
        def __init__(self, *_args):
            self.socket_path = tmp_path / "agent.sock"

        async def start_server(self):
            pass

        async def stop(self):
            pass

        async def listen(self, **_kwargs):
            raise RuntimeError("end test after idle inbox check")

    class FakeBus:
        def __init__(self, **_kwargs):
            pass

        def read_unread(self, _name):
            return [cc]

        def mark_done(self, _name, message_id):
            done.append(message_id)

    fake_module = ModuleType("fast_agent")
    fake_module.FastAgent = FakeAgent
    fake_config = ModuleType("fast_agent.spawn.config_reader")
    fake_config.get_skills = lambda *_args: []
    fake_channel = ModuleType("fast_agent.spawn.agent_channel")
    fake_channel.AgentChannel = FakeChannel
    fake_bus = ModuleType("fast_agent.spawn.message_bus")
    fake_bus.MessageBus = FakeBus
    for name, module in [
        ("fast_agent", fake_module),
        ("fast_agent.spawn.config_reader", fake_config),
        ("fast_agent.spawn.agent_channel", fake_channel),
        ("fast_agent.spawn.message_bus", fake_bus),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(isolated_runner, "create_child_config", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(isolated_runner, "emit_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(isolated_runner, "_install_tool_hooks", lambda *_args: None)
    monkeypatch.setattr(isolated_runner, "_install_termination_cleanup", lambda **_kwargs: None)

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(isolated_runner, "_save_agent_context_snapshot", noop)
    monkeypatch.setattr(isolated_runner, "_emit_mcp_status", noop)
    monkeypatch.setattr(isolated_runner, "_emit_runtime_config", lambda *_args: None)
    monkeypatch.setenv("TEAM_MY_NAME", "Taylor [Dev]")
    monkeypatch.setenv("TEAM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TEAM_MESSAGES_DIR", str(tmp_path))
    result = await isolated_runner.run_child_agent({
        "task": "Do first task", "agent_name": "Taylor [Dev]", "servers": [],
    }, tmp_path)

    assert result["status"] == "error"
    assert "end test after idle inbox check" in result["error"]
    assert sends == ["Do first task"]
    assert done == []  # CC remains available for the next actionable turn.
