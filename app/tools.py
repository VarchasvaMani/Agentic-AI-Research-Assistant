"""
app/tools.py
────────────
Defines the LangChain tools that the ReAct agent can call:

  1. retrieve_documents  — semantic search over ChromaDB, returns ranked chunks.
  2. summarize_document  — ask Claude to distil a long passage into key points.
  3. list_sources        — list all document sources currently in the vector store.

Each tool is decorated with @tool so LangChain auto-generates the JSON schema
that is passed to Claude as its tool manifest (MCP-compatible format).
"""

from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_chroma import Chroma

from app.config import settings
from app.embeddings import get_embedding_function
from app.llm import get_llm as _get_llm_from_factory

logger = logging.getLogger(__name__)

# ── Shared vector store instance ──────────────────────────────────────────────

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.chroma_collection,
            persist_directory=settings.chroma_persist_dir,
            embedding_function=get_embedding_function(),
        )
    return _vectorstore


def reset_vectorstore() -> None:
    """Force re-initialisation (call after ingesting new documents)."""
    global _vectorstore
    _vectorstore = None


# ── Shared LLM instance (for summarize tool) ──────────────────────────────────

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        _llm = _get_llm_from_factory(temperature=0.2)
    return _llm


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def retrieve_documents(
    query: Annotated[str, "The search query to find relevant research document chunks."],
    top_k: Annotated[int, "Number of chunks to retrieve (1–10). Default 5."] = 5,
) -> str:
    """
    Perform semantic similarity search over the ingested research documents
    stored in ChromaDB. Returns the most relevant text chunks along with
    their source file names and relevance scores.

    Use this tool whenever you need factual information from the research corpus.
    Call it multiple times with different query phrasings if the first result
    is insufficient.
    """
    top_k = max(1, min(top_k, 10))
    vs = get_vectorstore()

    try:
        results = vs.similarity_search_with_relevance_scores(query, k=top_k)
    except Exception as exc:
        logger.error("ChromaDB retrieval failed: %s", exc)
        return f"Retrieval error: {exc}"

    if not results:
        return "No relevant documents found for that query."

    parts: list[str] = []
    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (page {page})" if page != "" else ""
        parts.append(
            f"[{rank}] Source: {source}{page_info}  |  Score: {score:.3f}\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(parts)


@tool
def summarize_document(
    text: Annotated[str, "The text passage to summarize."],
    focus: Annotated[str, "Optional aspect to focus the summary on (e.g. 'methodology', 'results')."] = "",
) -> str:
    """
    Use Claude to produce a concise summary of a long text passage retrieved
    from the research corpus. Optionally focus the summary on a specific aspect
    such as 'key findings', 'methodology', 'limitations', or 'conclusions'.

    Use this after retrieve_documents when the raw chunks are too verbose
    and you need a crisp synthesis before composing your final answer.
    """
    focus_clause = f" Focus specifically on: {focus}." if focus else ""
    prompt = (
        f"Summarize the following research text concisely and accurately.{focus_clause}\n\n"
        f"TEXT:\n{text[:6000]}"  # guard against context overflow
    )
    try:
        response = get_llm().invoke(prompt)
        return response.content
    except Exception as exc:
        logger.error("Summarize tool failed: %s", exc)
        return f"Summarization error: {exc}"


@tool
def list_sources() -> str:
    """
    List all document sources (file names) currently stored in the ChromaDB
    vector store. Use this to understand what research material is available
    before deciding what to retrieve.
    """
    vs = get_vectorstore()
    try:
        data = vs.get(include=["metadatas"])
        sources = sorted({
            m.get("source", "unknown")
            for m in (data.get("metadatas") or [])
        })
        if not sources:
            return "No documents have been ingested yet."
        lines = "\n".join(f"  • {s}" for s in sources)
        return f"Available sources ({len(sources)}):\n{lines}"
    except Exception as exc:
        logger.error("list_sources failed: %s", exc)
        return f"Error listing sources: {exc}"


# ── Tool registry (imported by agent.py) ─────────────────────────────────────

ALL_TOOLS = [retrieve_documents, summarize_document, list_sources]
