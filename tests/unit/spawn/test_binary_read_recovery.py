import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.exceptions import RecoverableToolError
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.spawn.workspace_write_guard import create_workspace_write_hooks
from fast_agent.types.llm_stop_reason import LlmStopReason


@pytest.mark.asyncio
async def test_binary_rejection_can_recover_with_media_in_same_turn(tmp_path):
    path = tmp_path / "screen.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    executed = []

    def read_text_file(path: str) -> str:
        executed.append("text")
        return "binary should never reach this tool"

    def read_media_file(path: str) -> str:
        executed.append("media")
        return "media read"

    class RecoveryLLM(PassthroughLLM):
        turn = 0

        async def _apply_prompt_provider_specific(self, multipart_messages, *args, **kwargs):
            self.turn += 1
            if self.turn == 2:
                result = next(
                    m.tool_results["text"]
                    for m in multipart_messages
                    if m.tool_results and "text" in m.tool_results
                )
                assert result.isError
                assert "read_media_file" in result.content[0].text
            if self.turn <= 2:
                name, call_id = (
                    ("read_text_file", "text") if self.turn == 1 else ("read_media_file", "media")
                )
                return Prompt.assistant(
                    "",
                    stop_reason=LlmStopReason.TOOL_USE,
                    tool_calls={
                        call_id: CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(name=name, arguments={"path": str(path)}),
                        ),
                    },
                )
            return Prompt.assistant("Media inspected", stop_reason=LlmStopReason.END_TURN)

    class GuardedAgent(ToolAgent):
        def _tool_runner_hooks(self):
            return create_workspace_write_hooks(tmp_path, "QE")

    agent = GuardedAgent(AgentConfig("QE"), [read_text_file, read_media_file])
    agent._llm = RecoveryLLM()
    result = await agent.generate("inspect image")
    assert result.stop_reason == LlmStopReason.END_TURN
    assert result.last_text() == "Media inspected"
    assert executed == ["media"]


@pytest.mark.asyncio
async def test_execution_error_is_not_treated_as_safe_preflight_rejection():
    class FailingAgent(ToolAgent):
        async def run_tools(self, *args, **kwargs):
            raise RecoverableToolError("execution may have started")

    agent = FailingAgent(AgentConfig("QE"), [])
    from fast_agent.agents.tool_runner import ToolRunner

    runner = ToolRunner(agent=agent, messages=[])
    runner._pending_tool_request = Prompt.assistant(
        "",
        stop_reason=LlmStopReason.TOOL_USE,
        tool_calls={
            "execution": CallToolRequest(
                method="tools/call", params=CallToolRequestParams(name="write_file", arguments={})
            ),
        },
    )
    result = await runner.generate_tool_call_response()
    assert FAST_AGENT_ERROR_CHANNEL in result.channels
    assert "run_tools" in result.tool_results["execution"].content[0].text
