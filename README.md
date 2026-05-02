# Agentic RAG Research Assistant

A production-ready multi-step AI agent that autonomously retrieves, summarises,
and answers questions over research documents.

**Stack:** LangChain · Anthropic Claude API · ChromaDB · FastAPI · Python 3.12

---

## Architecture

```
User / Client
    │
    ▼  HTTP POST /ask
┌─────────────────────────────┐
│         FastAPI             │  ← MCP-compatible tool schema
│   (request validation,      │
│    OpenAPI docs at /docs)   │
└────────────┬────────────────┘
             │ invoke
             ▼
┌─────────────────────────────┐
│     LangChain ReAct Agent   │  ← Orchestrates tool calls
│                             │
│  Thought → Act → Observe    │  ← ReAct loop (max 8 iterations)
│     ↑_________________________│
└──────┬──────────────┬────────┘
       │              │
       ▼              ▼
┌──────────┐   ┌──────────────────┐
│ Claude   │   │    ChromaDB      │
│   API    │   │  (vector store)  │
│ (Claude  │   │                  │
│ Sonnet)  │   │ top-k semantic   │
│          │   │ similarity search│
└──────────┘   └──────────────────┘
                       ▲
               ┌───────┴────────┐
               │  Doc Ingestion │
               │ chunk→embed→   │
               │    store       │
               └────────────────┘
                       ▲
               PDFs, .txt, .md files
```

### How tool selection works

The agent receives all tool schemas (name + description + JSON input schema)
in its system prompt.  Claude reads these and autonomously decides:

1. Which tool to call (based on tool descriptions vs. current need)
2. What arguments to pass (from the JSON schema)
3. Whether to loop again after observing the result

This is the **ReAct** (Reason + Act) pattern — the agent interleaves
reasoning ("Thought") with action ("Action") and observation ("Observation")
until it has enough information to emit a final answer.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- An Anthropic API key from [console.anthropic.com](https://console.anthropic.com)

### 2. Install

```bash
git clone <repo>
cd rag_agent

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure — choose your LLM backend

**Option A — Anthropic Claude** (best quality, requires API key)
```bash
cp .env.example .env
# Set LLM_BACKEND=anthropic and your ANTHROPIC_API_KEY in .env
```

**Option B — Ollama** (free, local, no API key needed)
```bash
cp .env.example .env
# Set LLM_BACKEND=ollama in .env

# Install Ollama from https://ollama.com then pull a model:
ollama pull llama3      # best quality (~5 GB RAM)
ollama pull mistral     # balanced    (~4 GB RAM)
ollama pull phi3        # lightest    (~2 GB RAM)

# Also install the LangChain Ollama package:
pip install langchain-ollama
```

### 4. Ingest sample documents

```bash
python -m app.ingestion --docs ./sample_docs
```

This loads the three included research summaries (Attention Is All You Need,
RAG paper, LLM Agents survey) into ChromaDB.

### 5. Start the API server

```bash
python run.py
# or: uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs
Health:   http://localhost:8000/health

### 6. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does multi-head attention work?", "include_steps": true}'
```

Or use the CLI:

```bash
python scripts/query_cli.py "What are the key findings of the RAG paper?"
python scripts/query_cli.py --interactive --steps
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check, vector count |
| GET | `/sources` | List ingested document sources |
| POST | `/ask` | Submit a research question |
| POST | `/ingest/file` | Upload a PDF / text file |
| POST | `/ingest/directory` | Ingest a server-side directory |
| DELETE | `/collection` | Wipe the ChromaDB collection |

### POST /ask — Request body

```json
{
  "question": "What is the ReAct framework?",
  "include_steps": false,
  "chat_history": null
}
```

### POST /ask — Response

```json
{
  "answer": "The ReAct framework interleaves reasoning traces with...",
  "sources": ["llm_agents_survey.txt"],
  "iterations": 3,
  "steps": null
}
```

With `include_steps: true`, `steps` contains each tool call:

```json
"steps": [
  {"tool": "list_sources",        "input": {},                        "output": "..."},
  {"tool": "retrieve_documents",  "input": {"query": "ReAct", "top_k": 5}, "output": "..."},
  {"tool": "summarize_document",  "input": {"text": "...", "focus": ""}, "output": "..."}
]
```

---

## MCP Compatibility

Every endpoint's input/output is described via JSON Schema in the OpenAPI spec
at `GET /openapi.json`.  Any MCP-aware orchestrator can point its tool registry
at this URL to discover and call the agent programmatically.

Each `@tool` in `app/tools.py` also generates a standalone JSON schema
(name + description + parameters) that is passed to Claude as its tool manifest,
following the same MCP format.

---

## Adding Your Own Documents

**Via file upload:**
```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/your/paper.pdf"
```

**Via directory:**
```bash
curl -X POST "http://localhost:8000/ingest/directory?path=/path/to/docs"
```

**Supported formats:** `.pdf`, `.txt`, `.md`

---

## Adding Custom Tools

Create a new tool in `app/tools.py`:

```python
from langchain_core.tools import tool

@tool
def search_pubmed(
    query: Annotated[str, "PubMed search query"],
    max_results: Annotated[int, "Maximum results to return"] = 10,
) -> str:
    """Search PubMed for biomedical research papers matching the query."""
    # ... implementation
    return formatted_results

# Add to ALL_TOOLS list:
ALL_TOOLS = [retrieve_documents, summarize_document, list_sources, search_pubmed]
```

The agent will automatically discover the new tool and use it when appropriate.

---

## Docker

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY
docker-compose up -d
```

ChromaDB data is persisted in a named volume (`chroma_data`) across restarts.

---

## Tests

```bash
pip install pytest httpx
pytest tests/ -v

# Run integration tests (requires real API key + ChromaDB):
pytest tests/ -v -m integration
```

---

## Project Structure

```
rag_agent/
├── app/
│   ├── __init__.py
│   ├── config.py          # Settings (pydantic-settings, .env)
│   ├── embeddings.py      # Embedding function (HuggingFace / Voyage AI)
│   ├── ingestion.py       # Load → chunk → embed → store pipeline
│   ├── tools.py           # LangChain @tool definitions
│   ├── agent.py           # ReAct agent builder + run_agent()
│   ├── schemas.py         # Pydantic request/response models
│   └── main.py            # FastAPI app + all endpoints
├── scripts/
│   └── query_cli.py       # Interactive CLI (no server required)
├── tests/
│   └── test_all.py        # Unit + integration tests
├── sample_docs/           # Example research documents
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── run.py                 # Uvicorn entry point
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `anthropic` | `anthropic` or `ollama`. Switch to Ollama for free local inference. |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_BACKEND=anthropic`. Leave blank for Ollama. |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model (used when `LLM_BACKEND=anthropic`). |
| `OLLAMA_MODEL` | `llama3` | Ollama model name (used when `LLM_BACKEND=ollama`). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence path. |
| `CHROMA_COLLECTION` | `research_docs` | Collection name inside ChromaDB. |
| `CHUNK_SIZE` | `1000` | Characters per document chunk. |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks. |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per retrieval call. |
| `AGENT_MAX_ITERATIONS` | `8` | Max ReAct loop iterations. |
| `HOST` | `0.0.0.0` | Server bind address. |
| `PORT` | `8000` | Server port. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |
| `VOYAGE_API_KEY` | — | Optional. Uses Voyage AI embeddings if set. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model name (if no Voyage key). |
