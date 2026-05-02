"""
app/agent.py
────────────
Builds and runs the LangChain ReAct agent.

How tool selection works
────────────────────────
LangChain passes the tool schemas (name + description + JSON input schema) to
Claude as part of the system prompt.  Claude reads the user's question and
autonomously decides:

  1. Which tool to call next (or whether to emit a final answer).
  2. What arguments to pass to that tool.
  3. Whether to loop again after observing the tool's output.

This is the standard ReAct (Reason + Act) loop:
  Thought → Action (tool call) → Observation (tool result) → Thought → …

The agent stops when Claude emits a final answer or max_iterations is reached.

Key design decisions
────────────────────
• AgentExecutor wraps the agent and handles the ReAct loop automatically.
• handle_parsing_errors=True prevents crashes if Claude produces malformed output.
• return_intermediate_steps=True lets the API surface the reasoning chain.
• The system prompt steers Claude to use retrieve_documents before answering,
  and to cite sources explicitly.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import settings
from app.llm import get_llm
from app.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert research assistant with access to a corpus of scientific and academic documents stored in a vector database.

Your goal is to answer questions accurately and completely, grounding every claim in the retrieved documents.

## Workflow

1. **Start by listing available sources** using the `list_sources` tool so you know what material exists.
2. **Retrieve relevant chunks** using `retrieve_documents`. Use specific, targeted queries. Call this tool multiple times with different phrasings if needed.
3. **Summarize long passages** using `summarize_document` when raw retrieved text is verbose.
4. **Synthesize** the retrieved information into a clear, well-structured answer.
5. **Cite your sources** — always mention the document name and page (if available) for each key claim.

## Rules

- Never answer from memory alone — always retrieve first.
- If retrieval returns no results, say so honestly and suggest rephrasing.
- If the answer spans multiple documents, compare and contrast them.
- Be concise but complete. Use bullet points or numbered lists where helpful.
- If you are uncertain, say so explicitly.

## Output format

- Lead with a direct answer to the question.
- Follow with supporting evidence from the retrieved documents, citing [Source: filename].
- End with a brief note on confidence and any caveats.
"""

# ── Agent factory ─────────────────────────────────────────────────────────────

def build_agent_executor() -> AgentExecutor:
    """
    Construct and return a LangChain AgentExecutor backed by Claude.

    The executor is NOT cached — create one per request so conversations
    remain stateless and thread-safe.
    """
    llm = get_llm(temperature=0.1)  # low temperature for factual grounding

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # create_tool_calling_agent uses Claude's native tool-use API
    # (which is MCP-compatible — each tool's schema is auto-generated
    #  from the @tool decorator's type annotations and docstring).
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        max_iterations=settings.agent_max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        verbose=True,  # logs the ReAct loop to stdout; set False in prod
    )
    return executor


# ── Run helper ────────────────────────────────────────────────────────────────

def run_agent(question: str, chat_history: list | None = None) -> dict[str, Any]:
    """
    Run the agent against a question and return a structured result dict:

    {
        "answer":    str,           # Claude's final answer
        "steps":     list[dict],    # intermediate tool calls + observations
        "sources":   list[str],     # unique source files referenced
        "iterations": int,          # number of ReAct loop iterations
    }
    """
    executor = build_agent_executor()
    inputs: dict[str, Any] = {"input": question}
    if chat_history:
        inputs["chat_history"] = chat_history

    try:
        result = executor.invoke(inputs)
    except Exception as exc:
        logger.error("Agent execution failed: %s", exc, exc_info=True)
        return {
            "answer": f"Agent error: {exc}",
            "steps": [],
            "sources": [],
            "iterations": 0,
        }

    answer: str = result.get("output", "")
    raw_steps: list = result.get("intermediate_steps", [])

    # Parse intermediate steps into a serialisable format
    steps: list[dict] = []
    sources: set[str] = set()

    for action, observation in raw_steps:
        step: dict[str, Any] = {
            "tool": action.tool,
            "input": action.tool_input,
            "output": str(observation)[:2000],  # truncate for API response
        }
        steps.append(step)

        # Extract source file names from retrieve_documents output
        if action.tool == "retrieve_documents":
            for line in str(observation).split("\n"):
                if line.startswith("Source:") or "Source:" in line:
                    src = line.split("Source:")[-1].split("|")[0].strip()
                    if src:
                        sources.add(src)

    return {
        "answer": answer,
        "steps": steps,
        "sources": sorted(sources),
        "iterations": len(raw_steps),
    }
