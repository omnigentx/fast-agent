import json
from types import SimpleNamespace
from unittest.mock import Mock

from fast_agent.spawn.servers import team_communicate_server as server


def test_informational_message_cannot_trigger_reply_wake(monkeypatch):
    monkeypatch.setenv("TEAM_MY_NAME", "Dev")
    bus = Mock()
    bus.read_inbox.return_value = [SimpleNamespace(message_id="fyi", context={"no_reply": True})]
    monkeypatch.setattr(server, "_get_bus", lambda: bus)
    wake = Mock()
    monkeypatch.setattr(server, "_auto_wake_if_idle", wake)
    result = json.loads(server.reply_to_message("PM", "Acknowledged", original_message_id="fyi"))
    assert result["status"] == "not_sent"
    bus.send.assert_not_called()
    wake.assert_not_called()
