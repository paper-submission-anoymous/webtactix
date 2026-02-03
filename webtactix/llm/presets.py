# webtactix/llm/presets
from __future__ import annotations

from webtactix.llm.openai_compat import OpenAICompatConfig


def preset_qwen32b(key_num: int = 0) -> OpenAICompatConfig:
    """
    OpenAI-compatible endpoint for qwen3-32b.
    Supports parallel runs by choosing api_key via key_num.
    """
    base_url = "xxx"
    api_key_list = ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"]
    api_key = api_key_list[key_num % len(api_key_list)]
    model = "qwen3-32b"
    return OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model)


def preset_deepseek_chat(key_num: int = 0) -> OpenAICompatConfig:
    """
    DeepSeek official OpenAI-compatible endpoint.
    Supports parallel runs by choosing api_key via key_num.
    """
    base_url_list = ["https://api.deepseek.com/v1", "https://api.siliconflow.cn/v1"]
    api_key_list = [
        "sk-xxx",
        "sk-xxx",
    ]
    model_list = ["deepseek-chat", "deepseek-ai/DeepSeek-V3.2"]

    api_key = api_key_list[key_num % len(api_key_list)]
    base_url = base_url_list[key_num % len(base_url_list)]
    model = model_list[key_num % len(model_list)]
    return OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model)

def preset_chatgpt(key_num: int = 0) -> OpenAICompatConfig:
    """
    SiliconFlow OpenAI-compatible endpoint (ChatGPT-like models).
    Supports parallel runs by choosing api_key via key_num.
    """
    base_url = "https://oneai.evanora.top/v1"
    api_key_list = [
        "sk-xxx",
    ]
    api_key = api_key_list[key_num % len(api_key_list)]
    model = "gpt-4o"
    return OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model)