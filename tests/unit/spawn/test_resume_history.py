from types import SimpleNamespace

import pytest

from fast_agent.spawn.resume_history import restore_history


def test_native_restore_preserves_summary_and_removes_private_file(tmp_path):
    path = tmp_path / "db_resume_test.json"
    path.write_text(
        '{"messages":[{"role":"user","content":[{"type":"text","text":"[COMPACTED_CONTEXT_SUMMARY] old decision"}]}]}'
    )
    agent = SimpleNamespace(
        message_history=[], usage_accumulator=None, llm=None, clear=lambda **kwargs: None
    )
    restore_history(agent, path)
    assert len(agent.message_history) == 1
    assert "old decision" in agent.message_history[0].content[0].text
    assert not path.exists()


@pytest.mark.parametrize("content", [None, "invalid", '{"messages":[]}'])
def test_invalid_resume_aborts_and_cleans_private_file(tmp_path, content):
    path = tmp_path / "db_resume_invalid.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(Exception):
        restore_history(
            SimpleNamespace(
                message_history=[], usage_accumulator=None, llm=None, clear=lambda **kwargs: None
            ),
            path,
        )
    assert not path.exists()


def test_user_history_is_retained(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('{"messages":[{"role":"user","content":[{"type":"text","text":"decision"}]}]}')
    restore_history(
        SimpleNamespace(
            message_history=[], usage_accumulator=None, llm=None, clear=lambda **kwargs: None
        ),
        path,
    )
    assert path.exists()
