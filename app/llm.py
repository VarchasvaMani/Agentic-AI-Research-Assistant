"""
app/llm.py
──────────
Central LLM factory — returns the correct LangChain chat model based on
the LLM_BACKEND setting in your .env file.

  LLM_BACKEND=anthropic  →  ChatAnthropic (Claude API, requires ANTHROPIC_API_KEY)
  LLM_BACKEND=ollama     →  ChatOllama    (local Ollama, free, no API key needed)

All other modules call get_llm() from here — never instantiate a model directly.
This makes swapping backends a one-line .env change.

Ollama quick-start
──────────────────
1. Install:  https://ollama.com
2. Pull a model:
     ollama pull llama3      # best quality  (~5 GB RAM)
     ollama pull mistral     # balanced      (~4 GB RAM)
     ollama pull phi3        # lightest      (~2 GB RAM)
3. Set in .env:
     LLM_BACKEND=ollama
     OLLAMA_MODEL=llama3
4. Start your server as normal — no API key needed.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)


def get_llm(temperature: float | None = None) -> BaseChatModel:
    """
    Return a LangChain-compatible chat model for the configured backend.

    Args:
        temperature: Override the default temperature if needed.
                     Defaults to 0.1 for the agent (factual), 0.2 for summarization.

    Returns:
        A BaseChatModel instance — either ChatAnthropic or ChatOllama,
        both share the same LangChain interface so callers need not care.
    """
    backend = settings.llm_backend.lower()

    if backend == "ollama":
        return _build_ollama(temperature)

    # Default: Anthropic Claude
    return _build_anthropic(temperature)


def _build_anthropic(temperature: float | None) -> BaseChatModel:
    """Build a ChatAnthropic instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "langchain-anthropic is not installed.\n"
            "Run: pip install langchain-anthropic\n"
            "Or switch to the free local option by setting LLM_BACKEND=ollama in .env"
        ) from exc

    logger.info("LLM backend: Anthropic Claude (%s)", settings.claude_model)
    return ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=4096,
        temperature=temperature if temperature is not None else 0.1,
    )


def _build_ollama(temperature: float | None) -> BaseChatModel:
    """Build a ChatOllama instance."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is not installed.\n"
            "Run: pip install langchain-ollama\n"
            "Then make sure Ollama is running: https://ollama.com"
        ) from exc

    logger.info(
        "LLM backend: Ollama (%s) at %s",
        settings.ollama_model,
        settings.ollama_base_url,
    )
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature if temperature is not None else 0.1,
    )


def get_backend_info() -> dict:
    """Return a dict describing the active LLM backend — used by /health endpoint."""
    if settings.llm_backend == "ollama":
        return {
            "backend": "ollama",
            "model": settings.ollama_model,
            "base_url": settings.ollama_base_url,
            "api_key_required": False,
        }
    return {
        "backend": "anthropic",
        "model": settings.claude_model,
        "api_key_required": True,
    }
