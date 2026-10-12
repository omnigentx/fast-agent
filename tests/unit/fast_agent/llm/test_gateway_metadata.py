"""Gateway aliases retain attachments only when their endpoint declares support."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import ImageContent, TextContent

from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.llm.provider.openai.gateway_metadata import _CACHE, OpenAIGatewayMetadataMixin
from fast_agent.llm.provider_types import Provider


class Base:
    model_info = None


class Gateway(OpenAIGatewayMetadataMixin, Base):
    model_name = "coding-agent"
    provider = Provider.OPENAI
    resolved_model = SimpleNamespace(overlay=None, model_params=None)

    def _base_url(self):
        return "http://gateway.test/v1"

    def _api_key(self):
        return "test-key"

    def _default_headers(self):
        return {}


@pytest.fixture(autouse=True)
def clear_cache():
    _CACHE.clear()
    yield
    _CACHE.clear()


@pytest.mark.asyncio
async def test_declared_vision_survives_actual_agent_sanitizer(monkeypatch):
    response = httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": "coding-agent",
                    "capabilities": {"vision": True, "pdf": False},
                    "context_length": 272000,
                }
            ]
        },
    )
    get = AsyncMock(return_value=response)
    response.request = httpx.Request("GET", "http://gateway.test/v1/models")
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    llm = Gateway()
    await llm.refresh_model_info()
    agent = object.__new__(LlmDecorator)
    agent._llm = llm
    from fast_agent.types import PromptMessageExtended

    message = PromptMessageExtended(
        role="user",
        content=[
            TextContent(type="text", text="inspect"),
            ImageContent(type="image", mimeType="image/png", data="aGVsbG8="),
        ],
    )
    sanitized, removed = agent._sanitize_message_for_llm(message)
    assert [item.type for item in sanitized.content] == ["text", "image"]
    assert not removed
    assert llm.model_info.context_window == 272000
    await Gateway().refresh_model_info()
    assert get.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"data": [{"id": "other", "capabilities": {"vision": True}}]},
        {"data": [{"id": "coding-agent", "capabilities": {"vision": "true"}}]},
    ],
)
async def test_missing_or_invalid_metadata_does_not_enable_vision(monkeypatch, payload):
    response = httpx.Response(
        200, json=payload, request=httpx.Request("GET", "http://gateway.test/v1/models")
    )
    monkeypatch.setattr(httpx.AsyncClient, "get", AsyncMock(return_value=response))
    llm = Gateway()
    await llm.refresh_model_info()
    assert llm.model_info is None or not llm.model_info.supports_vision


@pytest.mark.asyncio
async def test_timeout_negative_cache_does_not_break_chat(monkeypatch):
    get = AsyncMock(side_effect=httpx.ReadTimeout("unavailable"))
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    await Gateway().refresh_model_info()
    await Gateway().refresh_model_info()
    assert get.await_count == 1


@pytest.mark.asyncio
async def test_user_overlay_is_not_overridden(monkeypatch):
    get = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    llm = Gateway()
    llm.resolved_model = SimpleNamespace(overlay=object(), model_params=None)
    await llm.refresh_model_info()
    get.assert_not_awaited()
