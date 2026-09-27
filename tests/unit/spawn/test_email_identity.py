"""Sender identity must remain trustworthy in the team email tool."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

from fast_agent.spawn.servers import email_server


def test_placeholder_sender_cannot_create_or_self_deliver_email(monkeypatch):
    monkeypatch.setenv("TEAM_MY_NAME", "Bennett [PM]")
    bus = Mock()
    monkeypatch.setattr(email_server, "get_bus", lambda: bus)

    result = json.loads(
        email_server.send_email(
            to="Bennett [PM]",
            body="status",
            my_name="{agent_name}",
        )
    )

    assert "Impersonation refused" in result["error"]
    bus.send.assert_not_called()


def test_valid_sender_cannot_email_self_even_with_different_case(monkeypatch):
    monkeypatch.setenv("TEAM_MY_NAME", "Bennett [PM]")
    bus = Mock()
    monkeypatch.setattr(email_server, "get_bus", lambda: bus)
    monkeypatch.setattr(
        email_server,
        "get_team_config",
        lambda: {
            "pm": {"agent_name": "Bennett [PM]"},
            "dev": {"agent_name": "Taylor [Dev]"},
        },
    )

    result = json.loads(
        email_server.send_email(
            to="bennett [pm]",
            body="status",
            my_name="bennett [pm]",
        )
    )

    assert "Cannot send email to yourself" in result["error"]
    assert result["available_teammates"] == ["Taylor [Dev]"]
    bus.send.assert_not_called()


def test_auto_detected_sender_is_canonical_and_cc_self_is_removed(monkeypatch):
    monkeypatch.setenv("TEAM_MY_NAME", "Bennett [PM]")
    bus = Mock()
    bus.send.return_value = SimpleNamespace(message_id="msg-1")
    monkeypatch.setattr(email_server, "get_bus", lambda: bus)
    monkeypatch.setattr(email_server, "auto_wake_if_idle", lambda _name: None)

    result = json.loads(
        email_server.send_email(
            to="Taylor [Dev]",
            cc="BENNETT [PM]",
            body="Please review",
        )
    )

    assert result["from"] == "Bennett [PM]"
    assert "cc_sent" not in result
    bus.send.assert_called_once()
    assert bus.send.call_args.kwargs["from_name"] == "Bennett [PM]"
