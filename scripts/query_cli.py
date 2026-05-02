#!/usr/bin/env python3
"""
scripts/query_cli.py
────────────────────
Interactive CLI for querying the agent without starting the FastAPI server.
Useful for development, debugging, and quick experiments.

Usage:
    # Single question
    python scripts/query_cli.py "What is the ReAct framework?"

    # Interactive REPL
    python scripts/query_cli.py --interactive

    # Show reasoning steps
    python scripts/query_cli.py --steps "How does multi-head attention work?"
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def print_divider(char: str = "─", width: int = 72) -> None:
    print(char * width)


def ask(question: str, show_steps: bool = False) -> None:
    from app.agent import run_agent

    print(f"\nQuestion: {question}")
    print_divider()
    print("Thinking...\n")

    result = run_agent(question=question)

    print("\n" + "═" * 72)
    print("ANSWER")
    print("═" * 72)
    print(result["answer"])

    if result["sources"]:
        print(f"\nSources: {', '.join(result['sources'])}")

    print(f"Iterations: {result['iterations']}")

    if show_steps and result["steps"]:
        print("\n" + "─" * 72)
        print("REASONING STEPS")
        print_divider()
        for i, step in enumerate(result["steps"], 1):
            print(f"\n[Step {i}] Tool: {step['tool']}")
            print(f"  Input:  {json.dumps(step['input'], indent=4)}")
            print(f"  Output: {step['output'][:400]}{'...' if len(step['output']) > 400 else ''}")


def interactive_repl(show_steps: bool = False) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║  Agentic RAG Research Assistant CLI  ║")
    print("╚══════════════════════════════════════╝")
    print("Type your question and press Enter.  'quit' or Ctrl+C to exit.\n")

    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        from app.agent import run_agent
        result = run_agent(question=question, chat_history=history or None)

        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print(f"[Sources: {', '.join(result['sources'])}]")

        if show_steps and result["steps"]:
            for i, step in enumerate(result["steps"], 1):
                print(f"  [Step {i}] {step['tool']} → {step['output'][:120]}...")

        print()
        history.append({"role": "human", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI for the Agentic RAG Research Assistant.")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode).")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive REPL.")
    parser.add_argument("--steps", "-s", action="store_true", help="Show reasoning steps.")
    args = parser.parse_args()

    if args.interactive or not args.question:
        interactive_repl(show_steps=args.steps)
    else:
        ask(args.question, show_steps=args.steps)


if __name__ == "__main__":
    main()
