"""
tests/test_all.py
─────────────────
Comprehensive test suite for the Agentic RAG Research Assistant.

Run with:
    pytest tests/ -v

Tests are organised into four groups:
  1. Config — settings loading and validation
  2. Ingestion — document loading, chunking, and vector store operations
  3. Tools — individual tool invocation
  4. API — FastAPI endpoint integration tests (uses TestClient, no live server needed)

Note: Tests that hit the real Anthropic API or ChromaDB are marked with
@pytest.mark.integration and skipped by default unless --integration flag is passed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """Write a small text file and return its path."""
    f = tmp_path / "sample.txt"
    f.write_text(
        "The Transformer architecture uses self-attention to process sequences in parallel. "
        "Unlike RNNs, it has no recurrence and can be trained much faster on modern hardware. "
        "Multi-head attention allows the model to attend to information from different subspaces. "
        "Positional encodings are added to give the model sequence order information.",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a small directory of text files."""
    (tmp_path / "doc1.txt").write_text("RAG combines retrieval with generation.", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("ReAct interleaves reasoning and acting.", encoding="utf-8")
    (tmp_path / "doc3.md").write_text("# Agents\nLLMs can use tools to solve tasks.", encoding="utf-8")
    return tmp_path


# ── 1. Config tests ───────────────────────────────────────────────────────────


class TestConfig:
    def test_settings_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        monkeypatch.setenv("RETRIEVAL_TOP_K", "7")

        # Clear the lru_cache so fresh settings are loaded
        from app.config import get_settings
        get_settings.cache_clear()
        from app.config import Settings
        s = Settings()

        assert s.anthropic_api_key == "sk-ant-test-key"
        assert s.claude_model == "claude-sonnet-4-20250514"
        assert s.retrieval_top_k == 7

    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from app.config import Settings
        s = Settings()
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 150
        assert s.agent_max_iterations == 8


# ── 2. Ingestion tests ────────────────────────────────────────────────────────


class TestIngestion:
    def test_load_document_txt(self, sample_txt):
        from app.ingestion import load_document
        docs = load_document(sample_txt)
        assert len(docs) >= 1
        assert "self-attention" in docs[0].page_content
        assert docs[0].metadata["source"] == "sample.txt"

    def test_load_document_unsupported(self, tmp_path):
        from app.ingestion import load_document
        bad = tmp_path / "file.xyz"
        bad.write_text("content")
        docs = load_document(bad)
        assert docs == []

    def test_load_directory(self, sample_dir):
        from app.ingestion import load_directory
        docs = load_directory(sample_dir)
        assert len(docs) >= 3

    def test_split_documents(self, sample_txt):
        from app.ingestion import load_document, split_documents
        docs = load_document(sample_txt)
        chunks = split_documents(docs)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 1200  # chunk_size + some tolerance

    @pytest.mark.integration
    def test_ingest_and_retrieve(self, sample_dir, monkeypatch, tmp_path):
        """Integration test: full ingest → retrieval round-trip."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("CHROMA_COLLECTION", "test_col")

        from app.ingestion import ingest_directory
        vs = ingest_directory(sample_dir)
        results = vs.similarity_search("retrieval generation", k=2)
        assert len(results) >= 1
        texts = " ".join(r.page_content for r in results)
        assert len(texts) > 0


# ── 3. Tools tests ────────────────────────────────────────────────────────────


class TestTools:
    def _mock_vectorstore(self, monkeypatch):
        """Patch get_vectorstore to return a mock."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                MagicMock(
                    page_content="Transformers use self-attention mechanisms.",
                    metadata={"source": "paper.txt", "page": 1},
                ),
                0.92,
            )
        ]
        mock_vs.get.return_value = {
            "metadatas": [{"source": "paper.txt"}, {"source": "rag.txt"}]
        }
        monkeypatch.setattr("app.tools.get_vectorstore", lambda: mock_vs)
        return mock_vs

    def test_retrieve_documents(self, monkeypatch):
        self._mock_vectorstore(monkeypatch)
        from app.tools import retrieve_documents
        result = retrieve_documents.invoke({"query": "self-attention", "top_k": 1})
        assert "self-attention" in result
        assert "paper.txt" in result
        assert "0.92" in result

    def test_retrieve_documents_empty(self, monkeypatch):
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = []
        monkeypatch.setattr("app.tools.get_vectorstore", lambda: mock_vs)
        from app.tools import retrieve_documents
        result = retrieve_documents.invoke({"query": "nonexistent topic"})
        assert "No relevant documents" in result

    def test_list_sources(self, monkeypatch):
        self._mock_vectorstore(monkeypatch)
        from app.tools import list_sources
        result = list_sources.invoke({})
        assert "paper.txt" in result
        assert "rag.txt" in result

    def test_summarize_document(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Summary: attention is key.")
        monkeypatch.setattr("app.tools.get_llm", lambda: mock_llm)
        from app.tools import summarize_document
        result = summarize_document.invoke({
            "text": "Attention is a mechanism that allows a model to...",
            "focus": "key findings",
        })
        assert "Summary" in result


# ── 4. API integration tests ──────────────────────────────────────────────────


class TestAPI:
    @pytest.fixture
    def client(self, monkeypatch):
        """Create a FastAPI test client with mocked dependencies."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        # Mock the agent so we don't hit the real API
        mock_result = {
            "answer": "Transformers use self-attention to process sequences in parallel.",
            "sources": ["attention_is_all_you_need.txt"],
            "iterations": 2,
            "steps": [
                {"tool": "list_sources", "input": {}, "output": "paper.txt"},
                {"tool": "retrieve_documents", "input": {"query": "transformer"}, "output": "chunk text"},
            ],
        }
        monkeypatch.setattr("app.main.run_agent", lambda **kwargs: mock_result)

        # Mock get_vectorstore for health and sources endpoints
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 42
        mock_vs.get.return_value = {"metadatas": [{"source": "paper.txt"}]}
        monkeypatch.setattr("app.main.get_vectorstore", lambda: mock_vs)

        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["vector_count"] == 42

    def test_sources(self, client):
        r = client.get("/sources")
        assert r.status_code == 200
        data = r.json()
        assert "paper.txt" in data["sources"]

    def test_ask_basic(self, client):
        r = client.post("/ask", json={"question": "What is self-attention?"})
        assert r.status_code == 200
        data = r.json()
        assert "self-attention" in data["answer"]
        assert data["iterations"] == 2
        assert data["steps"] is None  # include_steps defaults to False

    def test_ask_with_steps(self, client):
        r = client.post("/ask", json={
            "question": "What is self-attention?",
            "include_steps": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["steps"] is not None
        assert len(data["steps"]) == 2
        assert data["steps"][0]["tool"] == "list_sources"

    def test_ask_empty_question(self, client):
        r = client.post("/ask", json={"question": "ab"})
        assert r.status_code == 422  # validation error: min_length=3

    def test_ingest_unsupported_file(self, client):
        r = client.post(
            "/ingest/file",
            files={"file": ("bad.xlsx", b"data", "application/octet-stream")},
        )
        assert r.status_code == 422

    def test_ingest_directory_not_found(self, client):
        r = client.post("/ingest/directory?path=/nonexistent/path/xyz")
        assert r.status_code == 404
