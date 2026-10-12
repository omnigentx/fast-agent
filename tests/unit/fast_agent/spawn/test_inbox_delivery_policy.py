from fast_agent.spawn.inbox_delivery import build_inbox_prompt, needs_attention
from fast_agent.spawn.message_bus import AgentMessage


def test_cc_and_no_reply_do_not_start_idle_turn():
    assert not needs_attention(AgentMessage(context={"recipient_type": "cc"}))
    assert not needs_attention(AgentMessage(context={"no_reply": True}))
    assert needs_attention(AgentMessage(content="Fix the bug"))


def test_multimodal_delivery_keeps_recipient_and_image():
    message = AgentMessage(
        from_name="PM",
        to_name="Dev",
        content="Look",
        context={
            "attachments": [
                {"filename": "ui.png", "content_type": "image/png", "data_b64": "aW1hZ2U="}
            ]
        },
    )
    prompt = build_inbox_prompt([message])
    assert "to Dev" in prompt.content[0].text
    assert prompt.content[1].type == "image"
    assert prompt.content[1].data == "aW1hZ2U="


def test_cc_ownership_and_no_reply_are_visible():
    message = AgentMessage(
        from_name="PM",
        to_name="QE",
        content="Integrate art",
        context={
            "recipient_type": "cc",
            "primary_recipients": ["Dev"],
            "no_reply": True,
            "subject": "Integration",
        },
    )
    text = build_inbox_prompt([message]).content[0].text
    assert "Dev" in text and "FYI" in text and "Do not reply" in text


def test_no_reply_primary_and_cc_are_queued_without_wakes(tmp_path, monkeypatch):
    import json

    from fast_agent.spawn.message_bus import MessageBus
    from fast_agent.spawn.servers import email_server

    bus = MessageBus(tmp_path)
    wakes = []
    monkeypatch.setenv("TEAM_MY_NAME", "PM")
    monkeypatch.setattr(email_server, "get_bus", lambda: bus)
    monkeypatch.setattr(email_server, "assert_self_identity", lambda _name: ("PM", None))
    monkeypatch.setattr(email_server, "auto_wake_if_idle", wakes.append)
    result = json.loads(email_server.send_email("Dev", "Artifact ready", cc="QE", no_reply=True))
    assert result["status"] == "sent" and wakes == []
    assert len(bus.read_unread("Dev")) == len(bus.read_unread("QE")) == 1
    email_server.send_email("Dev", "Please integrate", cc="QE")
    assert wakes == ["Dev"]
