"""Bounded capability discovery for unknown models on custom OpenAI endpoints.

Only the configured endpoint is contacted. Explicit overlays and known models
keep their metadata; absent declarations never imply multimodal support.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from urllib.parse import urlparse

import httpx

from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider

_CACHE: OrderedDict[tuple[str, str, str], tuple[float, ModelInfo | None]] = OrderedDict()


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _parse_metadata(model: str, payload: object) -> ModelInfo | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None
    for entry in payload["data"]:
        if not isinstance(entry, dict) or entry.get("id") != model:
            continue
        caps = entry.get("capabilities")
        if not isinstance(caps, dict):
            return None
        tokenizes = ["text/plain"]
        if caps.get("vision") is True:
            tokenizes.extend(["image/jpeg", "image/png", "image/webp", "image/gif"])
        # Do not infer PDF/audio/video support from vision or provider name.
        return ModelInfo(
            name=model,
            provider=Provider.OPENAI,
            context_window=_positive_int(entry.get("context_length"))
            or _positive_int(caps.get("contextWindow")),
            max_output_tokens=_positive_int(entry.get("max_completion_tokens"))
            or _positive_int(caps.get("maxOutput")),
            tokenizes=tokenizes,
            json_mode=None,
            reasoning=None,
        )
    return None


class OpenAIGatewayMetadataMixin:
    @property
    def model_info(self):
        discovered = getattr(self, "_gateway_model_info", None)
        return discovered if discovered is not None else super().model_info

    async def refresh_model_info(self) -> None:
        # Keep provider subclasses, explicit user overlays, and known models
        # on their own discovery paths. This fills only missing alias metadata.
        if self.provider != Provider.OPENAI or self.resolved_model.overlay is not None:
            return
        if self.resolved_model.model_params is not None:
            return
        base = self._base_url()
        if not base or urlparse(str(base)).hostname in {"api.openai.com", None}:
            return
        model = self.model_name
        if not model:
            return
        try:
            secret = self._api_key()
            key = (str(base).rstrip("/"), hashlib.sha256(str(secret).encode()).hexdigest(), model)
            now = time.monotonic()
            cached = _CACHE.get(key)
            if cached and now - cached[0] < 300:
                self._gateway_model_info = cached[1]
                _CACHE.move_to_end(key)
                return
            headers = dict(self._default_headers() or {})
            headers["Authorization"] = f"Bearer {secret}"
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
                response = await client.get(key[0] + "/models", headers=headers)
                response.raise_for_status()
                info = _parse_metadata(model, response.json())
        except (httpx.HTTPError, ValueError):
            info = None
            if "key" not in locals():
                return
        self._gateway_model_info = info
        _CACHE[key] = (time.monotonic(), info)
        _CACHE.move_to_end(key)
        while len(_CACHE) > 32:
            _CACHE.popitem(last=False)
