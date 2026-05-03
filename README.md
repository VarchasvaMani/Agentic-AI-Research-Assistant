# 🔬 Agentic AI Research Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.2.6-green?logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-0.5.3-orange" />
  <img src="https://img.shields.io/badge/Claude_API-Anthropic-blueviolet?logo=anthropic&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-black?logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" />
</p>

<p align="center">
  A production-ready <strong>multi-step AI agent</strong> that autonomously retrieves, summarises,
  and answers questions over research documents using LangChain's ReAct loop,
  ChromaDB vector store, and your choice of LLM backend.
</p>

---

## ✨ Features

- 🤖 **Agentic ReAct loop** — Claude autonomously decides which tools to call, in what order, and when to stop
- 📚 **RAG pipeline** — documents are chunked, embedded, and stored in ChromaDB for grounded, citation-backed answers
- 🔀 **Dual LLM backend** — switch between Anthropic Claude (cloud) and Ollama (free, local) with a single `.env` change
- 🌐 **FastAPI REST API** — MCP-compatible OpenAPI schema at `/openapi.json`
- 📄 **Multi-format ingestion** — supports `.pdf`, `.txt`, and `.md` files
- 🐳 **Docker ready** — one command to spin up with persistent ChromaDB volume
- ✅ **Test suite** — 15 unit + integration tests with pytest

---

## 🏗️ Architecture

```
User / Client
     │
     ▼  HTTP POST /ask
┌──────────────────────────────┐
│          FastAPI             │  ← MCP-compatible tool schema
│  (validation + OpenAPI docs) │
└─────────────┬────────────────┘
              │ invoke
              ▼
┌──────────────────────────────┐
│    LangChain ReAct Agent     │  ← Orchestrates tool calls
│                              │
│   Thought → Act → Observe    │  ← Loop up to 8 iterations
│      ↑_______________________|
└───────┬──────────────┬───────┘
        │              │
        ▼              ▼
┌────────────┐  ┌─────────────────┐
│ Claude API │  │    ChromaDB     │
│    or      │  │  (vector store) │
│   Ollama   │  │  top-k search   │
└────────────┘  └────────┬────────┘
                         ▲
                ┌────────┴────────┐
                │  Doc Ingestion  │
                │ chunk→embed→    │
                │    store        │
                └─────────────────┘
                         ▲
                PDFs · TXT · MD files
```

### How tool selection works

The agent receives all tool schemas (name + description + JSON input schema) in its system prompt. The LLM reads these and autonomously decides:

1. **Which tool to call** — based on tool descriptions vs. current need
2. **What arguments to pass** — from the tool's JSON schema
3. **Whether to loop again** — after observing the tool's output

This is the **ReAct** (Reason + Act) pattern — the agent interleaves reasoning (`Thought`) with action (`Action`) and observation (`Observation`) until it has enough information to emit a final answer.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- One of:
  - Anthropic API key → [console.anthropic.com](https://console.anthropic.com)
  - Ollama installed → [ollama.com](https://ollama.com) *(free, no key needed)*

### 1. Clone & Install

```bash
git clone https://github.com/VarchasvaMani/Agentic-AI-Research-Assistant.git
cd Agentic-AI-Research-Assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure — choose your LLM backend

Copy the example config:
```bash
cp .env.example .env
```

**Option A — Anthropic Claude** *(best quality, requires API key)*
```env
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
CLAUDE_MODEL=claude-sonnet-4-20250514
```

**Option B — Ollama** *(free, runs locally, no API key needed)*
```env
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

Then pull a model with Ollama:
```bash
ollama pull llama3      # best quality  (~5 GB RAM)
ollama pull mistral     # balanced      (~4 GB RAM)
ollama pull phi3        # lightest      (~2 GB RAM)

pip install langchain-ollama   # one extra package for Ollama
```

### 3. Ingest documents

```bash
python -m app.ingestion --docs ./sample_docs
```

This loads the three included research papers into ChromaDB.

### 4. Start the server

```bash
python run.py
# or: uvicorn app.main:app --reload
```

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Interactive API docs (Swagger UI) |
| http://localhost:8000/health | Health check + vector count |
| http://localhost:8000/redoc | ReDoc API reference |

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does multi-head attention work?", "include_steps": true}'
```

Or use the interactive CLI (no server needed):
```bash
python scripts/query_cli.py "What are the key findings of the RAG paper?"
python scripts/query_cli.py --interactive --steps
```

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check, active backend, vector count |
| `GET` | `/sources` | List all ingested document sources |
| `POST` | `/ask` | Submit a research question to the agent |
| `POST` | `/ingest/file` | Upload a PDF / TXT / MD file |
| `POST` | `/ingest/directory` | Ingest all docs in a server-side directory |
| `DELETE` | `/collection` | Wipe the ChromaDB collection |

### POST /ask

**Request:**
```json
{
  "question": "What is the ReAct framework?",
  "include_steps": false,
  "chat_history": null
}
```

**Response:**
```json
{
  "answer": "The ReAct framework interleaves reasoning traces with tool calls...",
  "sources": ["llm_agents_survey.txt"],
  "iterations": 3,
  "steps": null
}
```

With `include_steps: true`, the full reasoning chain is returned:
```json
"steps": [
  {"tool": "list_sources",       "input": {},                             "output": "..."},
  {"tool": "retrieve_documents", "input": {"query": "ReAct", "top_k": 5}, "output": "..."},
  {"tool": "summarize_document", "input": {"text": "...", "focus": ""},   "output": "..."}
]
```

---

## 🧰 Agent Tools

The ReAct agent has access to three tools. The LLM reads their schemas and autonomously decides when and how to call them.

| Tool | Description |
|------|-------------|
| `retrieve_documents` | Semantic similarity search over ChromaDB — returns top-k chunks with scores |
| `summarize_document` | Asks the LLM to distil a long passage, with optional focus (e.g. "key findings") |
| `list_sources` | Lists all document file names in the vector store — always called first |

---

## 📂 Project Structure

```
Agentic-AI-Research-Assistant/
├── app/
│   ├── __init__.py
│   ├── config.py          # Settings via pydantic-settings + .env
│   ├── llm.py             # LLM factory — Claude API or Ollama
│   ├── embeddings.py      # Embedding function (HuggingFace / Voyage AI)
│   ├── ingestion.py       # Load → chunk → embed → store pipeline
│   ├── tools.py           # LangChain @tool definitions (3 tools)
│   ├── agent.py           # ReAct AgentExecutor + run_agent()
│   ├── schemas.py         # Pydantic v2 request/response models
│   └── main.py            # FastAPI app + all 6 endpoints
├── scripts/
│   └── query_cli.py       # Interactive CLI / REPL (no server needed)
├── tests/
│   ├── __init__.py
│   └── test_all.py        # 15 unit + integration tests
├── sample_docs/
│   ├── attention_is_all_you_need.txt
│   ├── retrieval_augmented_generation.txt
│   └── llm_agents_survey.txt
├── .env.example           # Config template — copy to .env
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py                 # Uvicorn entry point
```

---

## 📥 Adding Your Own Documents

**Upload via API:**
```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/your/paper.pdf"
```

**Ingest a directory:**
```bash
curl -X POST "http://localhost:8000/ingest/directory?path=/path/to/docs"
```

**Supported formats:** `.pdf` · `.txt` · `.md`

Re-uploading the same file name automatically replaces the old version.

---

## 🔧 Adding Custom Tools

Add a new `@tool` to `app/tools.py`:

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def search_arxiv(
    query: Annotated[str, "ArXiv search query"],
    max_results: Annotated[int, "Maximum results to return"] = 5,
) -> str:
    """Search ArXiv for recent papers matching the query."""
    # ... your implementation
    return formatted_results

# Register it:
ALL_TOOLS = [retrieve_documents, summarize_document, list_sources, search_arxiv]
```

The agent automatically discovers the new tool from its docstring and type annotations — no other changes needed.

---

## 🐳 Docker

```bash
cp .env.example .env    # fill in LLM_BACKEND + API key (if using Claude)
docker-compose up -d
```

ChromaDB data is persisted in a named Docker volume (`chroma_data`) and survives container restarts.

---

## 🧪 Tests

```bash
pip install pytest httpx
pytest tests/ -v

# Integration tests (requires real API key + running ChromaDB):
pytest tests/ -v -m integration
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `anthropic` | `anthropic` or `ollama` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_BACKEND=anthropic` |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model name |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence path |
| `CHROMA_COLLECTION` | `research_docs` | Collection name |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per retrieval call |
| `AGENT_MAX_ITERATIONS` | `8` | Max ReAct loop iterations |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `VOYAGE_API_KEY` | — | Optional — Voyage AI embeddings |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |

---

## 🤝 MCP Compatibility

Every endpoint is described via JSON Schema in the OpenAPI spec at `GET /openapi.json`. Any MCP-aware orchestrator can point its tool registry at this URL to discover and call the agent programmatically.

Each `@tool` in `app/tools.py` generates a standalone JSON schema (name + description + parameters) that matches the MCP tool manifest format exactly.

---

<p align="center">Built with LangChain · Anthropic Claude · ChromaDB · FastAPI</p>
