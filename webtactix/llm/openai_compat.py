# webtactix/llm/openai_compat
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import tiktoken
from openai import AsyncOpenAI  # ✅ 用异步 client


@dataclass(frozen=True)
class OpenAICompatConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout_s: float = 60.0


class OpenAICompatClient:
    """
    Standard OpenAI Python SDK client with custom base_url.
    Works for OpenAI, Qwen, DeepSeek, and other OpenAI-compatible services.
    """

    def __init__(self, cfg: OpenAICompatConfig) -> None:
        self.cfg = cfg

        self._client = AsyncOpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.timeout_s,
        )

    async def chat_text(self, *, system: str, user: str, temperature: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
        temp = self.cfg.temperature if temperature is None else float(temperature)
        model = self.cfg.model

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for _ in range(3):
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    temperature=temp,
                    messages=messages,
                )
            except Exception as e:
                print('[LLM ERR]', e)
                continue

        output_text = (resp.choices[0].message.content or "").strip()

        usage: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated": False,
            "model": model,
        }

        # 1) 优先用官方 usage
        if hasattr(resp, "usage") and resp.usage is not None:
            usage["prompt_tokens"] = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            usage["completion_tokens"] = int(getattr(resp.usage, "completion_tokens", 0) or 0)
            usage["total_tokens"] = int(getattr(resp.usage, "total_tokens", 0) or 0)
            usage["estimated"] = False
            return output_text, usage

        # 2) 没有 usage 用 tiktoken 估算
        try:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")

            prompt_tokens = sum(len(enc.encode(m["content"])) for m in messages)
            completion_tokens = len(enc.encode(output_text))
            total_tokens = prompt_tokens + completion_tokens

            usage["prompt_tokens"] = int(prompt_tokens)
            usage["completion_tokens"] = int(completion_tokens)
            usage["total_tokens"] = int(total_tokens)
            usage["estimated"] = True
        except Exception:
            # 极端兜底，保证不崩
            usage["estimated"] = True

        return output_text, usage

    async def chat_json(self, *, system: str, user: str, temperature: Optional[float] = None) -> Tuple[
        Union[Dict[str, Any], list], Dict[str, Any]]:
        text, usage = await self.chat_text(system=system, user=user, temperature=temperature)

        s = text.strip()

        # strip ```json fences
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        obj = json.loads(s)
        if not isinstance(obj, (dict, list)):
            raise ValueError(f"JSON root must be dict or list, got {type(obj)}")

        return obj, usage
