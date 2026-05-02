"""
app/config.py
─────────────
Centralised settings loaded from environment variables / .env file.
All other modules import `settings` from here — never os.getenv directly.

LLM backend selection
─────────────────────
Set LLM_BACKEND in your .env to switch between providers:

  LLM_BACKEND=anthropic   → Claude API (default, requires ANTHROPIC_API_KEY)
  LLM_BACKEND=ollama      → Local Ollama (free, no API key needed)
"""

from functools import lru_cache
from typing import Literal
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM backend ────────────────────────────────────────────────────────
    # "anthropic" = Claude API  |  "ollama" = local Ollama (no key needed)
    llm_backend: Literal["anthropic", "ollama"] = "anthropic"

    # ── Anthropic (used when llm_backend=anthropic) ────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    # ── Ollama (used when llm_backend=ollama) ─────────────────────────────
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"

    # ── ChromaDB ───────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "research_docs"

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 5

    # ── Agent ──────────────────────────────────────────────────────────────
    agent_max_iterations: int = 8

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_backend_config(self) -> "Settings":
        """Ensure the chosen backend has the required config."""
        if self.llm_backend == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_BACKEND=anthropic.\n"
                "Either set your API key, or switch to the free local option:\n"
                "  LLM_BACKEND=ollama"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — call this everywhere."""
    return Settings()


settings = get_settings()
