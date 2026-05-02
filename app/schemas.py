"""
app/schemas.py
──────────────
Pydantic v2 models for all FastAPI request and response bodies.

These also serve as the MCP-compatible tool schema surface — FastAPI
auto-generates OpenAPI (JSON Schema) from them, which external orchestrators
can consume to discover and call the agent endpoint programmatically.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── Ingest endpoints ──────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Returned after successfully ingesting documents."""
    message: str = Field(description="Human-readable status message.")
    chunks_stored: int = Field(description="Total number of vector chunks now in the store.")
    sources_added: list[str] = Field(description="List of file names that were processed.")


# ── Query / ask endpoint ──────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """
    MCP-compatible input schema for the /ask endpoint.
    External tools can introspect this via GET /openapi.json.
    """
    question: str = Field(
        description="The research question to answer.",
        min_length=3,
        max_length=4000,
        examples=["What are the key findings of the study on transformer attention?"],
    )
    chat_history: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Optional prior conversation turns for multi-turn sessions. "
            "Each entry: {\"role\": \"human\" | \"assistant\", \"content\": \"...\"}"
        ),
    )
    include_steps: bool = Field(
        default=False,
        description="If true, include the agent's intermediate reasoning steps in the response.",
    )


class AgentStep(BaseModel):
    """One iteration of the agent's ReAct loop."""
    tool: str = Field(description="Name of the tool that was called.")
    input: Any = Field(description="Arguments passed to the tool.")
    output: str = Field(description="Truncated result returned by the tool.")


class AskResponse(BaseModel):
    """Full response from the /ask endpoint."""
    answer: str = Field(description="Claude's final synthesized answer.")
    sources: list[str] = Field(description="Document sources referenced in the answer.")
    iterations: int = Field(description="Number of ReAct loop iterations performed.")
    steps: list[AgentStep] | None = Field(
        default=None,
        description="Intermediate reasoning steps (only present if include_steps=true).",
    )


# ── Sources endpoint ──────────────────────────────────────────────────────────

class SourcesResponse(BaseModel):
    """List of all ingested document sources."""
    sources: list[str] = Field(description="Document file names available in the vector store.")
    total: int = Field(description="Total number of unique source documents.")


# ── Health endpoint ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """API health check response."""
    status: str = Field(description="'ok' if all systems are operational.")
    llm_backend: str = Field(description="Active LLM backend: 'anthropic' or 'ollama'.")
    model: str = Field(description="LLM model in use.")
    collection: str = Field(description="ChromaDB collection name.")
    vector_count: int = Field(description="Number of vectors currently stored.")
