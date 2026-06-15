"""Regression tests for in-batch agent-name uniqueness.

Pin the bug surfaced in the PR #90 spawn review: ``_generate_unique_agent_name``
derives its taken-set from ``_collect_taken_names`` (live registry + DB team
sessions + ``agent_definitions``), but names assigned during a ``spawn_team``
pre-register loop live ONLY in ``session.agents`` until ``write_roster()`` runs
*after* the loop. So two roles sharing the same explicit ``role_display`` were
handed the SAME generated name, and the second silently overwrote the first in
the dict → one agent lost.

The fix threads an ``also_exclude`` set of names already reserved this batch.
These tests reproduce the loop and pin that distinct agents survive.
"""
from __future__ import annotations

import pytest

from fast_agent.spawn import team_spawner
from fast_agent.spawn.team_spawner import _generate_unique_agent_name


class _FakeStore:
    """TeamSessionStore stub — no cross-process siblings exist in the test."""

    def list_all(self):
        return []


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Empty, isolated registry + store; no SPAWN_REGISTRY_DB so the
    ``agent_definitions`` read is skipped. ``_collect_taken_names`` therefore
    returns the empty set and ``also_exclude`` is the only live constraint.
    """
    from fast_agent.spawn.spawn_registry import SpawnRegistry

    monkeypatch.delenv("SPAWN_REGISTRY_DB", raising=False)
    monkeypatch.setattr(team_spawner, "_team_store", _FakeStore())
    return SpawnRegistry(registry_file=tmp_path / "spawn_registry.json")


def test_also_exclude_is_honored(registry, monkeypatch):
    """Pin the pool to a single name so the ONLY way to stay unique is to
    honor ``also_exclude`` and fall through to the numbered fallback."""
    monkeypatch.setattr(team_spawner, "_AGENT_NAME_POOL", ["Robin"])

    first = _generate_unique_agent_name("ENG", registry)
    assert first == "Robin [ENG]"

    second = _generate_unique_agent_name("ENG", registry, also_exclude={first})
    assert second != first
    assert second.startswith("Robin") and "[ENG]" in second


def test_preregister_loop_keeps_both_same_display_roles(registry, monkeypatch):
    """Reproduce the spawn_team pre-register loop verbatim: two roles with the
    SAME explicit role_display must NOT collapse into one agent.

    Pool pinned to one name to force a collision absent ``also_exclude`` — this
    is exactly the state that lost an agent before the fix.
    """
    monkeypatch.setattr(team_spawner, "_AGENT_NAME_POOL", ["Robin"])
    roles = {
        "eng_a": {"role_display": "ENG"},
        "eng_b": {"role_display": "ENG"},  # same display on purpose
    }

    session_agents: dict[str, dict] = {}
    for role_name, role_config in roles.items():
        role_display = role_config.get("role_display", role_name.upper())
        agent_name = _generate_unique_agent_name(
            role_display, registry, also_exclude=set(session_agents)
        )
        session_agents[agent_name] = {"role": role_name, "agent_name": agent_name}

    assert len(session_agents) == 2, "both roles must survive as distinct agents"
    assert {v["role"] for v in session_agents.values()} == {"eng_a", "eng_b"}


def test_distinct_role_displays_unaffected(registry, monkeypatch):
    """Sanity: the fix must not perturb the common case where role_displays
    already differ — names stay distinct without leaning on the fallback."""
    monkeypatch.setattr(team_spawner, "_AGENT_NAME_POOL", ["Robin", "Sasha", "Toby"])

    session_agents: dict[str, dict] = {}
    for role_name in ("pm", "eng", "qa"):
        name = _generate_unique_agent_name(
            role_name.upper(), registry, also_exclude=set(session_agents)
        )
        session_agents[name] = {"role": role_name}

    assert len(session_agents) == 3


def test_every_team_call_site_passes_also_exclude():
    """Static guard: pin the ACTUAL fix, not a hand-copied loop.

    The loop-reproduction tests above validate `_generate_unique_agent_name`'s
    contract, but they'd stay green if a future edit dropped
    `also_exclude=set(session.agents)` from the real `spawn_team` /
    `_spawn_single_agent` / `spawn_team_members_for_session` call sites. Parse
    team_spawner.py and assert EVERY in-module call to the generator passes
    `also_exclude` — so deleting the kwarg breaks this test, not just review.

    Scope is team_spawner.py only: the standalone isolated-spawn call in
    agent_spawner_server.py has no session batch and intentionally omits it.
    """
    import ast
    import inspect

    src = inspect.getsource(team_spawner)
    tree = ast.parse(src)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_generate_unique_agent_name"
    ]
    assert len(calls) == 3, (
        f"expected 3 generator call sites in team_spawner.py, found {len(calls)} "
        "— update this guard if a call site was added/removed"
    )
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "also_exclude" in kwargs, (
            f"_generate_unique_agent_name call at line {call.lineno} dropped "
            "also_exclude — two same-role_display agents can collide again"
        )
