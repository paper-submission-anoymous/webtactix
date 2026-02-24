# webtactix/llm/openrouter_client.py
"""
Drop-in replacement for OpenAICompatClient that calls OpenRouter
via plain `requests`, wrapped in a thread-executor so it stays
compatible with the async agents (chat_text / chat_json interface).

Usage
-----
from webtactix.llm.openrouter_client import OpenRouterClient, OpenRouterConfig

cfg = OpenRouterConfig(api_key="sk-or-v1-...", model="deepseek/deepseek-v3.2")
llm = OpenRouterClient(cfg)

# inside an async context:
obj, usage = await llm.chat_json(system="...", user="...")
"""
from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import requests

# Module-level thread pool so all clients share it
_EXECUTOR = ThreadPoolExecutor(max_workers=8)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# Free models to rotate through when a model returns an empty response
FREE_MODELS: list[str] = [
    "deepseek/deepseek-r1-0528:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-120b:free",
    "arcee-ai/trinity-large-preview:free",
    "qwen/qwen3-coder:free",

    "meta-llama/llama-3.3-70b-instruct:free",
]


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    model: str = "deepseek/deepseek-r1-0528:free"
    fallback_models: tuple[str, ...] = tuple(FREE_MODELS[1:])
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_s: float = 120.0

    # Optional: shown in OpenRouter usage dashboard
    site_url: str = ""
    site_name: str = ""


class OpenRouterClient:
    """
    Async-compatible OpenRouter client.

    Implements the same interface as OpenAICompatClient:
        chat_text(system, user)  -> (str,  usage_dict)
        chat_json(system, user)  -> (dict|list, usage_dict)

    All agents that accept an `OpenAICompatClient` can accept this class
    without any other changes.
    """

    def __init__(self, cfg: OpenRouterConfig) -> None:
        self.cfg = cfg
        self._headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            **({"HTTP-Referer": cfg.site_url} if cfg.site_url else {}),
            **({"X-Title": cfg.site_name} if cfg.site_name else {}),
        }

    # ------------------------------------------------------------------
    # internal: synchronous call (runs in thread pool)
    # ------------------------------------------------------------------
    def _call_sync(
        self,
        system: str,
        user: str,
        temperature: float,
    ) -> Tuple[str, Dict[str, Any]]:
        models_to_try = [self.cfg.model, *self.cfg.fallback_models]

        last_exc: Exception = RuntimeError("No attempts made")
        for model in models_to_try:
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": self.cfg.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            }
            for attempt in range(3):
                try:
                    resp = requests.post(
                        _OPENROUTER_URL,
                        headers=self._headers,
                        json=payload,
                        timeout=self.cfg.timeout_s,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    # extract text
                    output_text: str = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        or ""
                    ).strip()

                    if not output_text:
                        print(f"[OPENROUTER] model={model!r} returned empty response, switching to next fallback")
                        last_exc = RuntimeError(f"model={model!r} returned empty response")
                        break  # move to next fallback model immediately

                    # extract usage
                    raw_usage = data.get("usage") or {}
                    usage: Dict[str, Any] = {
                        "prompt_tokens":     int(raw_usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
                        "total_tokens":      int(raw_usage.get("total_tokens", 0)),
                        "estimated":         False,
                        "model":             model,
                    }
                    return output_text, usage

                except Exception as exc:
                    last_exc = exc
                    print(f"[OPENROUTER] model={model!r} attempt {attempt + 1} failed: {exc}")

                    # On 429 (rate limit): sleep 60s and retry the same model
                    if isinstance(exc, requests.exceptions.HTTPError):
                        if exc.response is not None and exc.response.status_code == 429:
                            print(f"[OPENROUTER] Rate limit hit on {model!r}, sleeping 60s before retry")
                            time.sleep(60.0)


        raise ValueError(
            f"All models exhausted with no valid response. Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # public async interface (matches OpenAICompatClient exactly)
    # ------------------------------------------------------------------
    async def chat_text(
        self,
        *,
        system: str,
        user: str,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Return (raw_text, usage_dict)."""
        temp = self.cfg.temperature if temperature is None else float(temperature)
        loop = asyncio.get_event_loop()
        output_text, usage = await loop.run_in_executor(
            _EXECUTOR, self._call_sync, system, user, temp
        )
        return output_text, usage

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: Optional[float] = None,
    ) -> Tuple[Union[Dict[str, Any], list], Dict[str, Any]]:
        """Return (parsed_json, usage_dict).  Strips ```json fences automatically."""
        text, usage = await self.chat_text(system=system, user=user, temperature=temperature)

        s = text.strip()
        print(s)

        # strip <think>...</think> blocks (DeepSeek R1 / reasoning models)
        import re as _re
        s = _re.sub(r"<think>.*?</think>", "", s, flags=_re.DOTALL).strip()

        # strip ``` fences  (```json ... ``` or ``` ... ```)
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        if not s:
            raise ValueError(
                f"LLM returned an empty response (model={self.cfg.model!r}). "
                "Check rate limits or model availability."
            )

        obj = json.loads(s)
        if not isinstance(obj, (dict, list)):
            raise ValueError(f"JSON root must be dict or list, got {type(obj)}")

        return obj, usage
