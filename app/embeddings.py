"""
app/embeddings.py
─────────────────
Returns a LangChain-compatible embedding function.

Strategy (in priority order):
  1. If VOYAGE_API_KEY is set, use Anthropic's voyage-3 embeddings via
     langchain-voyageai (best quality, requires separate key).
  2. Otherwise fall back to sentence-transformers/all-MiniLM-L6-v2 via
     langchain-community's HuggingFaceEmbeddings (free, runs locally).

The returned object is passed to Chroma() as `embedding_function`.
All other modules call get_embedding_function() — never instantiate directly.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_function():
    """
    Returns a cached embedding function.
    Cached so we don't reload the model on every request.
    """
    voyage_key = os.getenv("VOYAGE_API_KEY")

    if voyage_key:
        try:
            from langchain_voyageai import VoyageAIEmbeddings  # type: ignore

            logger.info("Using Voyage AI embeddings (voyage-3)")
            return VoyageAIEmbeddings(
                voyage_api_key=voyage_key,
                model="voyage-3",
            )
        except ImportError:
            logger.warning(
                "langchain-voyageai not installed — falling back to HuggingFace embeddings. "
                "Run: pip install langchain-voyageai"
            )

    # Default: local sentence-transformers model (no API key required)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Using HuggingFace embeddings: %s", model_name)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        logger.warning(
            "langchain-community / sentence-transformers not available. "
            "Using Chroma's built-in default embeddings."
        )

    # Last resort: Chroma default (all-MiniLM-L6-v2 via chromadb's bundled copy)
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    class _ChromaEmbeddingAdapter:
        """Wraps Chroma's DefaultEmbeddingFunction to match LangChain's interface."""

        def __init__(self):
            self._fn = DefaultEmbeddingFunction()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._fn(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._fn([text])[0]

    logger.info("Using Chroma built-in default embeddings")
    return _ChromaEmbeddingAdapter()
