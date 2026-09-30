import pytest


@pytest.mark.asyncio
async def test_host_can_choose_scoped_skill_reader_with_restricted_filesystem(tmp_path):
    from mcp.types import CallToolResult, TextContent, Tool

    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.mcp_agent import McpAgent
    from fast_agent.context import Context
    from fast_agent.skills.registry import SkillRegistry

    class RestrictedFilesystem:
        tools = [Tool(name='read_text_file',description='Restricted filesystem',inputSchema={'type':'object'})]
        async def call_tool(self,name,arguments=None,tool_use_id=None,*,request_params=None):
            return CallToolResult(isError=True,content=[TextContent(type='text',text='Access denied')])
        def metadata(self):
            return {}
    skill = tmp_path / 'skill'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('---\nname: reviewed\ndescription: Reviewed\n---\nREVIEWED_MARKER')
    agent = McpAgent(AgentConfig(name='Reader',servers=[],skills=[]),context=Context(no_shell=True))
    agent.set_filesystem_runtime(RestrictedFilesystem())
    agent.set_skill_manifests(SkillRegistry(directories=[skill]).load_manifests())
    assert agent.skill_read_tool_name == 'read_text_file'
    agent.set_skill_reader_preference(True)
    assert agent.skill_read_tool_name == 'read_skill'
    assert 'read_skill' in [tool.name for tool in (await agent.list_tools()).tools]
    result = await agent.call_tool('read_skill',{'path':str(skill / 'SKILL.md')})
    assert not result.isError
    assert isinstance(result.content[0], TextContent)
    assert 'REVIEWED_MARKER' in result.content[0].text
    denied = await agent.call_tool('read_skill',{'path':str(tmp_path / 'outside-secret')})
    assert denied.isError
    assert not agent.shell_runtime_enabled
