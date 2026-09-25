"""Independent teams must retain each other's workspace and child configs."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_agent.spawn import team_spawner


@pytest.mark.asyncio
async def test_second_team_with_same_name_keeps_first_session_files(tmp_path, monkeypatch):
    stored = {}
    store = SimpleNamespace(upsert=lambda sid, data: stored.update({sid: data}))
    monkeypatch.setattr(team_spawner, "_get_store", lambda: store)
    monkeypatch.setattr(team_spawner, "load_team_template", lambda *_: {
        "name": "very_long_team_template_name_that_exceeds_the_workspace_limit",
        "orchestrator": "pm", "roles": {"pm": {"role_display": "PM"}},
    })
    monkeypatch.setattr(team_spawner, "_generate_unique_agent_name",
                        lambda *args, **kwargs: f"PM-{len(stored)}")
    async def spawn_stub(**kwargs):
        return kwargs["session"].agents[next(iter(kwargs["session"].agents))]["run_id"]
    monkeypatch.setattr(team_spawner, "_spawn_single_agent", spawn_stub)
    monkeypatch.setattr(team_spawner, "get_runtime_paths", lambda *_: {
        "workspaces": tmp_path / "workspaces", "tmp": tmp_path / "tmp",
    })

    first = await team_spawner.spawn_team(
        "long-template", "Build feature A", registry=object(),
        project_dir=tmp_path, team_name="shared-name",
    )
    marker = first.workspace / "work-in-progress.txt"
    marker.write_text("important work")
    configs = tmp_path / "tmp" / "child_configs"
    configs.mkdir(parents=True)
    (configs / "first-agent.yaml").write_text("still needed")

    second = await team_spawner.spawn_team(
        "long-template", "Build feature B", registry=object(),
        project_dir=tmp_path, team_name="shared-name",
    )
    assert first.workspace != second.workspace
    assert first.session_id in first.workspace.name
    assert second.session_id in second.workspace.name
    assert marker.read_text() == "important work"
    assert (configs / "first-agent.yaml").read_text() == "still needed"
