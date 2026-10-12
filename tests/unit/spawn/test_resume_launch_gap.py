import json
from types import SimpleNamespace
from unittest.mock import Mock

from fast_agent.spawn.agent_channel import AgentChannel
from fast_agent.spawn.live_resume import queue_live_follow_up
from fast_agent.spawn.message_bus import MessageBus


def test_follow_up_during_socket_bind_gap_is_queued_without_respawning(monkeypatch, tmp_path):
    record = SimpleNamespace(
        agent_name="QE",
        session_id="team-a",
        run_id="old",
        original_config={"env_vars": {"TEAM_MESSAGES_DIR": str(tmp_path)}},
    )
    registry = SimpleNamespace(
        list_all=lambda: [record], has_running_resume=Mock(return_value=True)
    )
    monkeypatch.setattr(AgentChannel, "is_alive", lambda *a, **kw: False)
    result = queue_live_follow_up(registry, record, "old", "Check the checkpoint")
    assert result is not None
    data = json.loads(result)
    assert data["status"] == "queued"
    assert data["delivery"] == "queued_launching_process"
    registry.has_running_resume.assert_called_once_with("QE", "team-a", verify_process=True)
    assert MessageBus(str(tmp_path)).read_unread("QE")[0].content == "Check the checkpoint"


def test_dead_process_does_not_block_history_resume(monkeypatch, tmp_path):
    record = SimpleNamespace(
        agent_name="QE", session_id="team-a", run_id="dead", original_config={}
    )
    registry = SimpleNamespace(list_all=lambda: [record], has_running_resume=lambda *a, **kw: False)
    monkeypatch.setattr(AgentChannel, "is_alive", lambda *a, **kw: False)
    assert queue_live_follow_up(registry, record, "dead", "Continue") is None
