"""A host hook can expose newly attached tools in the same running turn."""

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, ListToolsResult, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunner, ToolRunnerHooks
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


def warmup() -> str:
    return "continue"


class RecordingLLM(PassthroughLLM):
    def __init__(self) -> None:
        super().__init__()
        self.tools_seen: list[list[str]] = []

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.tools_seen.append([tool.name for tool in tools or []])
        if len(self.tools_seen) == 1:
            return Prompt.assistant(
                "warmup",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "first": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="warmup", arguments={}),
                    )
                },
            )
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class LiveToolsAgent(ToolAgent):
    enabled = False

    async def list_tools(self) -> ListToolsResult:
        result = await super().list_tools()
        if self.enabled:
            result.tools.append(
                Tool(
                    name="plugin_echo",
                    description="New live capability",
                    inputSchema={"type": "object"},
                )
            )
        return result

    def _tool_runner_hooks(self) -> ToolRunnerHooks:
        async def before_llm(runner: ToolRunner, messages: list[PromptMessageExtended]) -> None:
            if runner.iteration == 1:
                self.enabled = True
                await runner.refresh_tools()

        return ToolRunnerHooks(before_llm_call=before_llm)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_exposes_attached_tool_in_same_turn_without_restart() -> None:
    llm = RecordingLLM()
    agent = LiveToolsAgent(AgentConfig("live"), [warmup])
    agent._llm = llm
    result = await agent.generate("run")
    assert result.last_text() == "done"
    assert len(llm.tools_seen) == 2
    assert "plugin_echo" not in llm.tools_seen[0]
    assert "plugin_echo" in llm.tools_seen[1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_failure_preserves_previous_tool_snapshot() -> None:
    class FailingDiscoveryAgent(ToolAgent):
        fail_discovery = False

        async def list_tools(self) -> ListToolsResult:
            if self.fail_discovery:
                raise RuntimeError("discovery failed")
            return await super().list_tools()

    agent = FailingDiscoveryAgent(AgentConfig("live"), [warmup])
    runner = ToolRunner(agent=agent, messages=[])
    await runner.refresh_tools()
    previous = runner._tools
    agent.fail_discovery = True
    with pytest.raises(RuntimeError, match="discovery failed"):
        await runner.refresh_tools()
    assert runner._tools is previous


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("finalized_field", ["_done", "_deferred_structured_finalization_started"])
async def test_refresh_cannot_reopen_finalized_turn(finalized_field: str) -> None:
    agent = ToolAgent(AgentConfig("live"), [warmup])
    runner = ToolRunner(agent=agent, messages=[])
    setattr(runner, finalized_field, True)
    with pytest.raises(RuntimeError, match="finalization"):
        await runner.refresh_tools()
