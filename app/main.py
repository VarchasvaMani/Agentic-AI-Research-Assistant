"""
app/main.py
───────────
FastAPI application exposing the agentic RAG assistant via HTTP.

Endpoints
─────────
  GET  /health              — liveness check, reports vector count
  GET  /sources             — list all ingested document sources
  POST /ask                 — submit a research question to the agent
  POST /ingest/file         — upload a PDF or text file for ingestion
  POST /ingest/directory    — ingest all docs in a server-side directory
  DELETE /collection        — wipe the ChromaDB collection (dev/reset)

MCP compatibility
─────────────────
The OpenAPI schema at GET /openapi.json describes every endpoint's input/output
using JSON Schema — the same format MCP uses for tool manifests.  An external
orchestrator can point its tool registry at this URL and immediately discover
how to call the agent.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import run_agent
from app.config import settings
from app.ingestion import ingest_directory, ingest_file, get_vectorstore
from app.llm import get_backend_info
from app.schemas import (
    AgentStep,
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestResponse,
    SourcesResponse,
)
from app.tools import reset_vectorstore

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up the embedding model and ChromaDB connection."""
    logger.info("Starting Agentic RAG Research Assistant...")
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        logger.info(
            "ChromaDB ready — collection '%s' has %d vectors",
            settings.chroma_collection,
            count,
        )
    except Exception as exc:
        logger.warning("ChromaDB warm-up skipped: %s", exc)
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG Research Assistant",
    description=(
        "A multi-step AI agent built with LangChain and Claude that autonomously "
        "retrieves, summarizes, and answers questions over research documents stored "
        "in ChromaDB.  All endpoints follow MCP-compatible tool schema patterns."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Returns API status and current vector store size."""
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
    except Exception:
        count = -1
    info = get_backend_info()
    return HealthResponse(
        status="ok",
        llm_backend=info["backend"],
        model=info["model"],
        collection=settings.chroma_collection,
        vector_count=count,
    )


# ── Sources ───────────────────────────────────────────────────────────────────

@app.get(
    "/sources",
    response_model=SourcesResponse,
    summary="List ingested document sources",
    tags=["Documents"],
)
async def list_sources() -> SourcesResponse:
    """Returns all document file names currently stored in ChromaDB."""
    try:
        vs = get_vectorstore()
        data = vs.get(include=["metadatas"])
        sources = sorted({
            m.get("source", "unknown")
            for m in (data.get("metadatas") or [])
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return SourcesResponse(sources=sources, total=len(sources))


# ── Ask ───────────────────────────────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a research question",
    tags=["Agent"],
)
async def ask(body: AskRequest) -> AskResponse:
    """
    Submit a research question to the LangChain ReAct agent.

    The agent will:
    1. List available sources.
    2. Retrieve relevant document chunks from ChromaDB.
    3. Optionally summarize long passages.
    4. Synthesize a grounded answer citing sources.

    Set `include_steps: true` to receive the full reasoning chain.
    """
    logger.info("Question received: %s", body.question[:120])

    result = run_agent(
        question=body.question,
        chat_history=body.chat_history,
    )

    steps = None
    if body.include_steps:
        steps = [
            AgentStep(tool=s["tool"], input=s["input"], output=s["output"])
            for s in result["steps"]
        ]

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        iterations=result["iterations"],
        steps=steps,
    )


# ── Ingest: file upload ───────────────────────────────────────────────────────

@app.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    tags=["Documents"],
)
async def ingest_uploaded_file(
    file: Annotated[UploadFile, File(description="PDF or plain-text file to ingest.")],
) -> IngestResponse:
    """
    Upload a PDF (.pdf) or text (.txt / .md) file.
    The file is chunked, embedded, and stored in ChromaDB.
    Re-uploading the same file name replaces the previous version.
    """
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    # Write to a temp file so the loader can read it by path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        vs = ingest_file(tmp_path)
        # Rename metadata source back to the original filename
        count = vs._collection.count()
        reset_vectorstore()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        message=f"Successfully ingested '{file.filename}'.",
        chunks_stored=count,
        sources_added=[file.filename or tmp_path],
    )


# ── Ingest: server-side directory ─────────────────────────────────────────────

@app.post(
    "/ingest/directory",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest all docs in a server-side directory",
    tags=["Documents"],
)
async def ingest_dir(
    path: Annotated[str, Query(description="Absolute or relative path to the document directory.")] = "./sample_docs",
) -> IngestResponse:
    """
    Ingest all .pdf, .txt, and .md files found (recursively) in `path`.
    Useful for bulk loading a corpus at startup.
    """
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: '{path}'")

    try:
        vs = ingest_directory(path)
        count = vs._collection.count()
        data = vs.get(include=["metadatas"])
        sources = sorted({
            m.get("source", "unknown")
            for m in (data.get("metadatas") or [])
        })
        reset_vectorstore()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        message=f"Ingested documents from '{path}'.",
        chunks_stored=count,
        sources_added=sources,
    )


# ── Collection reset ──────────────────────────────────────────────────────────

@app.delete(
    "/collection",
    summary="Wipe the ChromaDB collection",
    tags=["System"],
)
async def delete_collection() -> JSONResponse:
    """
    Delete all vectors from the ChromaDB collection.
    Use during development / testing to start fresh.
    """
    try:
        vs = get_vectorstore()
        vs.delete_collection()
        reset_vectorstore()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse({"message": f"Collection '{settings.chroma_collection}' wiped."})
