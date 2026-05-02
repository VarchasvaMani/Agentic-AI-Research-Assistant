"""
app/ingestion.py
────────────────
Document ingestion pipeline:
  1. Load PDF / plain-text files from disk.
  2. Split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
  3. Embed each chunk with Anthropic's voyage-style embeddings via langchain-anthropic
     (falls back to a lightweight sentence-transformers model if voyage is unavailable).
  4. Persist vectors into ChromaDB.

Usage (CLI):
    python -m app.ingestion --docs ./sample_docs

Usage (Python):
    from app.ingestion import ingest_directory
    ingest_directory("./sample_docs")
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic

# We use Chroma's default embedding (sentence-transformers/all-MiniLM-L6-v2)
# unless you swap in a custom one.  This keeps the dependency footprint small.
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.config import settings
from app.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


# ── Loaders ───────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_document(file_path: str | Path) -> List[Document]:
    """Load a single file into a list of LangChain Documents."""
    path = Path(file_path)
    ext = path.suffix.lower()
    loader_cls = SUPPORTED_EXTENSIONS.get(ext)
    if loader_cls is None:
        logger.warning("Unsupported file type %s — skipping %s", ext, path.name)
        return []
    logger.info("Loading %s", path.name)
    loader = loader_cls(str(path))
    docs = loader.load()
    # Attach source metadata
    for doc in docs:
        doc.metadata.setdefault("source", path.name)
    return docs


def load_directory(directory: str | Path) -> List[Document]:
    """Recursively load all supported documents from a directory."""
    directory = Path(directory)
    all_docs: List[Document] = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in directory.rglob(f"*{ext}"):
            all_docs.extend(load_document(file_path))
    logger.info("Loaded %d raw document pages/sections", len(all_docs))
    return all_docs


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_documents(docs: List[Document]) -> List[Document]:
    """Chunk documents for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        "Split %d documents → %d chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks


# ── Vector store ──────────────────────────────────────────────────────────────

def get_vectorstore(embedding_function=None) -> Chroma:
    """Return (or create) the persisted ChromaDB vector store."""
    ef = embedding_function or get_embedding_function()
    return Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=ef,
    )


def ingest_documents(docs: List[Document]) -> Chroma:
    """
    Embed and store a list of Documents into ChromaDB.
    Deduplicates by document source so re-ingesting the same file is safe.
    """
    chunks = split_documents(docs)
    if not chunks:
        logger.warning("No chunks produced — nothing to ingest.")
        return get_vectorstore()

    ef = get_embedding_function()
    vectorstore = Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=ef,
    )

    # Deduplicate: remove existing chunks from the same sources before re-adding
    sources = {c.metadata.get("source") for c in chunks if c.metadata.get("source")}
    for source in sources:
        try:
            existing = vectorstore.get(where={"source": source})
            if existing and existing["ids"]:
                vectorstore.delete(ids=existing["ids"])
                logger.info(
                    "Removed %d stale chunks for source '%s'",
                    len(existing["ids"]),
                    source,
                )
        except Exception:
            pass  # Collection may be empty on first run

    vectorstore.add_documents(chunks)
    logger.info("Ingested %d chunks into ChromaDB collection '%s'", len(chunks), settings.chroma_collection)
    return vectorstore


def ingest_directory(directory: str | Path) -> Chroma:
    """Convenience function: load → split → embed → store."""
    docs = load_directory(directory)
    if not docs:
        raise ValueError(f"No supported documents found in '{directory}'.")
    return ingest_documents(docs)


def ingest_file(file_path: str | Path) -> Chroma:
    """Ingest a single file."""
    docs = load_document(file_path)
    if not docs:
        raise ValueError(f"Could not load '{file_path}'.")
    return ingest_documents(docs)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Ingest research documents into ChromaDB.")
    parser.add_argument("--docs", default="./sample_docs", help="Directory of documents to ingest")
    args = parser.parse_args()

    store = ingest_directory(args.docs)
    col = store._collection
    print(f"\nDone. ChromaDB collection '{settings.chroma_collection}' now has {col.count()} vectors.")
