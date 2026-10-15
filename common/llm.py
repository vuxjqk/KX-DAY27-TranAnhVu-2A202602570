"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — copy .env.example to .env")
    # LLM_TEMPERATURE overrides the argument; "default" omits it entirely, for
    # reasoning models that only accept their built-in temperature.
    override = os.environ.get("LLM_TEMPERATURE", "").strip()
    if override:
        temperature = None if override.lower() == "default" else float(override)
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
    )
