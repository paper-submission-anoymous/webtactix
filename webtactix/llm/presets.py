# webtactix/llm/presets
from __future__ import annotations

from webtactix.llm.openai_compat import OpenAICompatConfig
from webtactix.llm.openrouter_client import OpenRouterClient, OpenRouterConfig


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


def preset_openrouter(
    model: str = "deepseek/deepseek-r1-0528:free",
    key_num: int = 0,
) -> OpenRouterClient:
    """
    OpenRouter client.  Returns an OpenRouterClient that has the same
    chat_text / chat_json interface as OpenAICompatClient, so it can be
    passed to PlannerAgent, DataExtractionAgent, DecisionAgent, etc.

    Example
    -------
    llm = preset_openrouter()                          # default model
    llm = preset_openrouter("openai/gpt-4o")           # different model
    """
    api_key_list = [
        "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",
    ]
    api_key = api_key_list[key_num % len(api_key_list)]
    cfg = OpenRouterConfig(api_key=api_key, model=model)
    return OpenRouterClient(cfg)


def preset_deepseek_chat(key_num: int = 0) -> OpenAICompatConfig:
    """
    DeepSeek official OpenAI-compatible endpoint.
    Supports parallel runs by choosing api_key via key_num.
    """
    base_url_list = ["https://api.deepseek.com/v1", "xxx"]
    api_key_list = [
        "sk-xxx",
        "sk-xxx",
    ]
    model_list = ["deepseek-chat", "xxx"]

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