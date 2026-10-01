from pathlib import Path

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.registry import SkillRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("shell", [False, True])
async def test_scoped_skill_reader_preserves_existing_shell_access(tmp_path: Path, shell: bool):
    skill = tmp_path / "review"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: review\ndescription: Review\n---\nmarker")
    agent = McpAgent(AgentConfig(name="test", servers=[], skills=[], shell=shell), context=Context())
    before = agent.shell_runtime_enabled
    agent.set_skill_reader_preference(True)
    agent.set_skill_manifests(SkillRegistry.load_directory(tmp_path))
    assert agent.shell_runtime_enabled == before
    names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_skill" in names
    assert ("execute" in names) == before
