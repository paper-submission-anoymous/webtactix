# webtactix/llm/vllm_client.py
"""
Client for a locally hosted vLLM server using the official OpenAI Python SDK.

vLLM exposes the same OpenAI-compatible API at http://localhost:8000/v1, so
we simply point the OpenAI client there. This gives us ALL OpenAI SDK features:
    - chat.completions.create()
    - completions.create()
    - embeddings.create()
    - models.list()
    - streaming
    - tool/function calling
    - async support
    - etc.

Start the vLLM server first:
    vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --trust-remote-code \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 16384 \
        --dtype bfloat16

Usage
-----
from webtactix.llm.vllm_client import VLLMClient, VLLMConfig

cfg = VLLMConfig(model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
llm = VLLMClient(cfg)

# High-level interface (same as all other clients in the codebase):
obj, usage = await llm.chat_json(system="...", user="...")
text, usage = await llm.chat_text(system="...", user="...")

# Or use the raw OpenAI client directly for full API access:
response = await llm.async_client.chat.completions.create(...)
response = await llm.async_client.embeddings.create(...)
models    = await llm.async_client.models.list()
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI

_VLLM_BASE_URL = "http://localhost:8080/v1"


@dataclass(frozen=True)
class VLLMConfig:
    model: str = "Qwen/Qwen3.5-0.8B"
    base_url: str = _VLLM_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_s: float = 120.0
    # vLLM doesn't require a real API key, but the OpenAI SDK requires a non-empty string
    api_key: str = "EMPTY"


class VLLMClient:
    """
    Async-compatible client for a locally hosted vLLM server.

    Uses the official OpenAI Python SDK pointed at http://localhost:8080/v1,
    giving full access to ALL OpenAI-compatible endpoints vLLM supports:
    chat completions, text completions, embeddings, streaming, tool calling, etc.

    High-level interface (matches OpenAICompatClient):
        chat_text(system, user)  -> (str, usage_dict)
        chat_json(system, user)  -> (dict|list, usage_dict)
        chat_stream(system, user) -> AsyncIterator[str]   (streaming)

    Raw OpenAI SDK access:
        llm.client        -> openai.OpenAI        (sync)
        llm.async_client  -> openai.AsyncOpenAI   (async)
    """

    def __init__(self, cfg: VLLMConfig) -> None:
        self.cfg = cfg

        # Sync client — for simple scripts or blocking contexts
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
        )

        # Async client — for use inside async agents
        self.async_client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
        )

    # ------------------------------------------------------------------
    # High-level async interface (matches OpenAICompatClient exactly)
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

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(3):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )

                output_text = (response.choices[0].message.content or "").strip()
                if not output_text:
                    raise RuntimeError(
                        f"[VLLM] model={self.cfg.model!r} returned empty response"
                    )

                usage: Dict[str, Any] = {
                    "prompt_tokens":     response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens":      response.usage.total_tokens if response.usage else 0,
                    "estimated":         False,
                    "model":             self.cfg.model,
                }
                return output_text, usage

            except Exception as exc:
                last_exc = exc
                print(f"[VLLM] model={self.cfg.model!r} attempt {attempt + 1} failed: {exc}")

        raise ValueError(
            f"vLLM request failed after 3 attempts. Last error: {last_exc}"
        )

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: Optional[float] = None,
    ) -> Tuple[Union[Dict[str, Any], list], Dict[str, Any]]:
        """Return (parsed_json, usage_dict). Strips ```json fences and <think> blocks."""
        text, usage = await self.chat_text(system=system, user=user, temperature=temperature)

        s = text.strip()
        print(s)

        # strip <think>...</think> blocks (reasoning models like DeepSeek R1)
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

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
                "Check that the vLLM server is running and the model is loaded."
            )

        obj = json.loads(s)
        if not isinstance(obj, (dict, list)):
            raise ValueError(f"JSON root must be dict or list, got {type(obj)}")

        return obj, usage

    async def chat_stream(
        self,
        *,
        system: str,
        user: str,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they stream in from vLLM."""
        temp = self.cfg.temperature if temperature is None else float(temperature)

        async with await self.async_client.chat.completions.create(
            model=self.cfg.model,
            temperature=temp,
            max_tokens=self.cfg.max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta