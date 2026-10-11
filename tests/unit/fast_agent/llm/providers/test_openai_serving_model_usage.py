from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletion

from fast_agent.context import Context
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.request_params import RequestParams


@pytest.mark.asyncio
async def test_gateway_alias_usage_records_response_model(monkeypatch):
    llm = OpenAILLM(context=Context(), model="gpt-4o")
    response = ChatCompletion(
        id="test",
        created=0,
        object="chat.completion",
        model="gpt-4o",
        choices=[
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
        usage={"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
    )
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=AsyncMock())

    @asynccontextmanager
    async def open_client():
        yield client

    monkeypatch.setattr(llm, "_openai_client", open_client)
    monkeypatch.setattr(llm, "_process_stream", AsyncMock(return_value=(response, [])))
    finalized = Mock()
    monkeypatch.setattr(llm, "_finalize_turn_usage", finalized)
    await llm._openai_completion(
        [{"role": "user", "content": "Continue"}], RequestParams(model="coding-agent")
    )
    assert client.chat.completions.create.call_args.kwargs["model"] == "coding-agent"
    assert finalized.call_args.args[0].model == "gpt-4o"


@pytest.mark.asyncio
async def test_manual_stream_preserves_gateway_serving_model():
    from openai.types.chat import ChatCompletionChunk

    llm = OpenAILLM(context=Context(), model="gpt-4o")

    async def chunks():
        yield ChatCompletionChunk(
            id="test",
            created=0,
            object="chat.completion.chunk",
            model="gpt-4o",
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        )

    response, _ = await llm._process_stream_manual(chunks(), model="coding-agent")
    assert response.model == "gpt-4o"
