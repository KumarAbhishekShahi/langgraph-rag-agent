"""
main.py
=======
Interactive CLI entry point for the LangGraph Enterprise RAG Agent.

Run:
    python main.py               # interactive menu loop
    python main.py --ingest      # run ingestion only and exit
    python main.py --demo        # demo mode (no API keys needed)

Menu:
  1. Ingest          — load all sources, build ChromaDB
  2. Analyze         — deep-analyze a requirement (15 sections)
  3. Gherkin         — generate Gherkin + auto-create Jira Story
  4. Code            — generate code for a Jira issue (dual RAG)
  5. History         — show this session's conversation history
  0. Exit
"""

import argparse
import sys
from typing import Optional

from app.state import AgentState
from app.graph import build_graph
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Build the LangGraph once at startup ──────────────────────────────────────
graph = build_graph()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║    LangGraph Enterprise RAG Agent  v1.0            ║")
    print("║    Jira · Confluence · KB · PDF/CSV · Codebase     ║")
    print("╚" + "═" * 58 + "╝")


def _menu():
    print("\n┌─ MENU " + "─" * 50 + "┐")
    print("│  1  Ingest all sources → build ChromaDB           │")
    print("│  2  Analyze requirement                            │")
    print("│  3  Generate Gherkin + create Jira Story           │")
    print("│  4  Generate code for a Jira issue                 │")
    print("│  5  Show conversation history                      │")
    print("│  0  Exit                                           │")
    print("└" + "─" * 56 + "┘")
    return input("  Choose [0-5]: ").strip()


def _run(intent: str, user_input: str, history: list) -> list:
    """Invoke the LangGraph with the given intent and return updated history."""
    initial_state: AgentState = {
        "user_input":           user_input,
        "intent":               intent,
        "retrieved_context":    [],
        "final_answer":         "",
        "conversation_history": history,
        "jira_issue_created":   None,
        "error":                None,
    }

    result = graph.invoke(initial_state)

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")
        logger.error(result["error"])

    return result.get("conversation_history", history)


def _show_history(history: list):
    if not history:
        print("\n[INFO] No conversation history yet.")
        return
    print("\n" + "═" * 60)
    print("  CONVERSATION HISTORY")
    print("═" * 60)
    for i, turn in enumerate(history):
        role = turn.get("role", "?").upper()
        text = turn.get("content", "")[:300]
        print(f"\n[{i+1}] {role}:")
        print(f"  {text}...")
    print("═" * 60)


def _get_input(prompt: str) -> Optional[str]:
    """Get multi-line input (end with blank line or single '.')."""
    print(f"\n{prompt}")
    print("(Press Enter twice or type '.' on a new line to submit)")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None
        if line.strip() == ".":
            break
        if not line.strip() and lines:
            break
        lines.append(line)
    result = "\n".join(lines).strip()
    return result if result else None


# ── Demo mode ─────────────────────────────────────────────────────────────────

def _run_demo():
    """Quick demo using sample KB articles — no API keys needed."""
    print("\n[DEMO MODE] Running ingestion on sample KB articles...")
    history = _run("ingest", "", [])

    sample_req = (
        "We need to add OAuth2 Single Sign-On (SSO) to our customer portal. "
        "Users should be able to log in with their company Microsoft Azure AD account. "
        "The system must support PKCE flow, refresh tokens, and automatic logout after 8 hours."
    )

    print("\n[DEMO MODE] Analyzing sample requirement...")
    print(f"Requirement: {sample_req[:80]}...")
    history = _run("analyze", sample_req, history)

    print("\n[DEMO] ✅ Demo complete. Check outputs/ folder for saved results.")
    sys.exit(0)


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LangGraph Enterprise RAG Agent")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion and exit")
    parser.add_argument("--demo",   action="store_true", help="Run demo mode and exit")
    args = parser.parse_args()

    _banner()

    if args.demo:
        _run_demo()
        return

    if args.ingest:
        print("\n[CLI] Running ingestion...")
        _run("ingest", "", [])
        print("[CLI] Ingestion complete. Exiting.")
        return

    # ── Interactive loop ──────────────────────────────────────────────────
    history: list = []
    print("\n[INFO] Welcome! Run option 1 first to ingest your sources.")
    print("[INFO] Ollama must be running: ollama serve")
    print("[INFO] Default model: llama3.2 (change LLM_MODEL in .env)")

    while True:
        try:
            choice = _menu()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Interrupted. Goodbye!")
            break

        if choice == "0":
            print("\n[INFO] Goodbye!")
            break

        elif choice == "1":
            history = _run("ingest", "", history)

        elif choice == "2":
            req = _get_input("Enter the requirement to analyze:")
            if req:
                history = _run("analyze", req, history)
            else:
                print("[WARN] Empty input — skipped.")

        elif choice == "3":
            req = _get_input("Enter the feature request for Gherkin generation:")
            if req:
                history = _run("gherkin", req, history)
            else:
                print("[WARN] Empty input — skipped.")

        elif choice == "4":
            req = _get_input(
                "Enter a Jira issue key (e.g. PROJ-148) or describe what to code:"
            )
            if req:
                history = _run("code", req, history)
            else:
                print("[WARN] Empty input — skipped.")

        elif choice == "5":
            _show_history(history)

        else:
            print("[WARN] Invalid choice. Enter 0–5.")


if __name__ == "__main__":
    main()
