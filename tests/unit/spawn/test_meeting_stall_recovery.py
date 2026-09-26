"""Meeting stall recovery preserves identity and transcript in both stores."""

import json
import time

import pytest

from fast_agent.spawn.servers import meeting_room_server as room
from fast_agent.spawn.servers.meeting_storage import (
    JsonFileMeetingStorage,
    SqliteMeetingStorage,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("store_kind", ["json", "sqlite"])
async def test_only_chair_can_close_stalled_turn(store_kind, monkeypatch, tmp_path):
    storage = (
        JsonFileMeetingStorage(str(tmp_path)) if store_kind == "json"
        else SqliteMeetingStorage(str(tmp_path / "meetings.db"))
    )
    old_storage = room._storage
    room.configure_meeting_room(storage=storage)
    monkeypatch.setattr(room, "_notify_turn_agent", lambda *args: None)
    monkeypatch.setattr(room, "_notify_meeting_started", lambda *args: None)
    notified = []
    monkeypatch.setattr(room, "_notify_meeting_ended", lambda *args, **kwargs: notified.append((args, kwargs)))
    try:
        monkeypatch.setenv("TEAM_MY_NAME", "Bennett [PM]")
        created = json.loads(await room.create_meeting("Review", "Taylor [Dev]"))
        meeting_id = created["meeting_id"]
        assert json.loads(await room.speak(meeting_id, "Start review"))["next_speaker"] == "Taylor [Dev]"
        state = storage.get_state(meeting_id)
        state["turn_started_at"] = time.time() - 601
        storage.update_state(meeting_id, state)

        monkeypatch.setenv("TEAM_MY_NAME", "Parker [QE]")
        refused = json.loads(await room.end_meeting(meeting_id, "unresponsive"))
        assert "Only the meeting chair" in refused["error"]
        assert storage.get_state(meeting_id)["ended"] is False

        monkeypatch.setenv("TEAM_MY_NAME", "Bennett [PM]")
        closed = json.loads(await room.end_meeting(meeting_id, "speaker process exited"))
        assert closed["outcome"] == "stalled_turn_recovered"
        assert storage.get_state(meeting_id)["stalled_speaker"] == "Taylor [Dev]"
        assert storage.get_transcript(meeting_id)[-1]["type"] == "recovery"
        assert len(notified) == 2
        assert all(kwargs == {"wake": False} for _, kwargs in notified)
        assert "already ended" in json.loads(await room.end_meeting(meeting_id, "again"))["error"]
    finally:
        room.configure_meeting_room(storage=old_storage)


@pytest.mark.asyncio
async def test_chair_cannot_close_active_turn(monkeypatch, tmp_path):
    old_storage = room._storage
    room.configure_meeting_room(storage=SqliteMeetingStorage(str(tmp_path / "meetings.db")))
    monkeypatch.setattr(room, "_notify_turn_agent", lambda *args: None)
    monkeypatch.setattr(room, "_notify_meeting_started", lambda *args: None)
    monkeypatch.setenv("TEAM_MY_NAME", "Bennett [PM]")
    try:
        meeting_id = json.loads(await room.create_meeting("Review", "Taylor [Dev]"))["meeting_id"]
        assert "your turn" in json.loads(await room.end_meeting(meeting_id, "skip it"))["error"]
        await room.speak(meeting_id, "Start")
        result = json.loads(await room.end_meeting(meeting_id, "too soon"))
        assert result["recovery_after_seconds"] == room.STALL_RECOVERY_SECONDS
        assert room._storage.get_state(meeting_id)["ended"] is False
    finally:
        room.configure_meeting_room(storage=old_storage)
