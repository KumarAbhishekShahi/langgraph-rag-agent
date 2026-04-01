"""
code_agent.py
=============
LangGraph node: generates implementation code for a Jira issue using
dual-context RAG — fetches both the issue details AND relevant source files.

Three-step flow:
  Step 1 — Fetch the Jira issue (summary + description + acceptance criteria)
  Step 2 — Dual RAG retrieval:
              a. KB / Confluence context (for patterns, API docs)
              b. Codebase context        (for existing code, structure)
  Step 3 — Stream code generation from LLM with full context

State reads : user_input  (should contain Jira issue key, e.g. "PROJ-148")
State writes: retrieved_context, final_answer, conversation_history
"""

import re
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import config
from app.state import AgentState
from app.prompts import CODE_PROMPT
from app.rag.vectorstore import load_vectorstore
from app.rag.retriever import retrieve_context, retrieve_codebase_context
from app.utils.logger import get_logger, format_history, strip_think_block
from app.utils.file_saver import save_output

logger = get_logger(__name__)


def _get_llm():
    backend = config.LLM_BACKEND.lower()
    if backend == "openai":
        return ChatOpenAI(model=config.OPENAI_MODEL,
                          api_key=config.OPENAI_API_KEY, temperature=0.15)
    elif backend == "anthropic":
        return ChatAnthropic(model=config.ANTHROPIC_MODEL,
                             api_key=config.ANTHROPIC_API_KEY, temperature=0.15)
    else:
        return ChatOllama(model=config.LLM_MODEL,
                          base_url=config.OLLAMA_BASE_URL, temperature=0.15)


def _extract_issue_key(text: str) -> str:
    """Extract Jira issue key from user input (e.g. PROJ-148)."""
    match = re.search(r"\b([A-Z][A-Z0-9]+-\d+)\b", text.upper())
    return match.group(1) if match else ""


def _fetch_issue_from_vectorstore(vectorstore, issue_key: str) -> str:
    """
    Retrieve the Jira issue content from the vectorstore using its key.
    Falls back to similarity search on the key if metadata filter fails.
    """
    if not issue_key:
        return "No issue key provided."

    try:
        # Try metadata filter first
        docs = vectorstore.similarity_search(
            query=issue_key,
            k=3,
            filter={"issue_key": issue_key}
        )
        if docs:
            return "\n\n".join(d.page_content for d in docs)
    except Exception:
        pass

    # Fallback: search by issue key string
    docs = vectorstore.similarity_search(query=issue_key, k=3)
    if docs:
        return "\n\n".join(d.page_content for d in docs)

    return f"Issue {issue_key} not found in knowledge base. Run ingestion first."


def code_node(state: AgentState) -> AgentState:
    """Fetch issue context + codebase context → generate implementation code."""

    user_input = state.get("user_input", "")
    history    = state.get("conversation_history", [])

    logger.info(f"Code node — input: {user_input[:80]}")
    print("\n" + "=" * 60)
    print("  CODE GENERATION AGENT")
    print("=" * 60)

    # ── Step 1: Extract issue key ─────────────────────────────────────────
    issue_key = _extract_issue_key(user_input)
    if issue_key:
        print(f"[INFO] Jira issue detected : {issue_key}")
    else:
        print(f"[INFO] No issue key in input — treating as free-form code request")
        print(f"[INFO] Tip: include a key like 'PROJ-123' to load exact issue details")

    try:
        print("\n[STEP 1] Loading knowledge base...")
        vectorstore = load_vectorstore()

        # ── Step 2a: Retrieve issue context ───────────────────────────────
        if issue_key:
            print(f"[STEP 2a] Fetching issue details for {issue_key}...")
            issue_context = _fetch_issue_from_vectorstore(vectorstore, issue_key)
        else:
            issue_context = user_input

        # ── Step 2b: Retrieve KB / documentation context ──────────────────
        print("[STEP 2b] Retrieving KB / documentation context (MMR)...")
        kb_chunks    = retrieve_context(vectorstore, user_input, k=config.RETRIEVAL_K)
        kb_context   = "\n\n---\n\n".join(kb_chunks)
        print(f"[OK]      Retrieved {len(kb_chunks)} KB chunks")

        # ── Step 2c: Retrieve codebase context ────────────────────────────
        print("[STEP 2c] Retrieving codebase context...")
        code_query   = f"{issue_key} {user_input}" if issue_key else user_input
        code_chunks  = retrieve_codebase_context(vectorstore, code_query, k=8)
        code_context = "\n\n---\n\n".join(code_chunks) if code_chunks else "No codebase indexed."
        print(f"[OK]      Retrieved {len(code_chunks)} codebase chunks")

        history_text = format_history(history)

        # ── Step 3: Stream code generation ───────────────────────────────
        print(f"\n[STEP 3] Generating code (model: {config.LLM_MODEL})...")
        print("-" * 60)

        llm   = _get_llm()
        chain = CODE_PROMPT | llm

        answer = ""
        for chunk in chain.stream({
            "issue_context":   issue_context,
            "kb_context":      kb_context,
            "codebase_context": code_context,
            "user_input":      user_input,
            "history":         history_text,
        }):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            print(text, end="", flush=True)
            answer += text

        print("\n" + "-" * 60)
        answer = strip_think_block(answer)

        # ── Save output ───────────────────────────────────────────────────
        saved = save_output(
            intent="code",
            user_input=user_input,
            answer=answer,
            issue_key=issue_key or None,
        )
        if saved:
            print(f"\n[SAVED] {saved}")

        # ── Update conversation history ───────────────────────────────────
        all_context  = kb_chunks + code_chunks
        updated_hist = history + [
            {"role": "user",      "content": user_input[:500]},
            {"role": "assistant", "content": answer[:500]},
        ]

        logger.info("Code node completed")
        return {
            **state,
            "retrieved_context":    all_context,
            "final_answer":         answer,
            "conversation_history": updated_hist,
            "error":                None,
        }

    except Exception as e:
        logger.error(f"Code node failed: {e}")
        return {**state, "final_answer": "", "error": f"Code generation failed: {str(e)}"}
