"""
gherkin_agent.py
================
LangGraph node: generates a Gherkin issue from a requirement,
then creates it as a new Jira Story via the Atlassian API.

Two-phase operation:
  Phase 1 — Generate: RAG retrieval → prompt → LLM → Gherkin text
  Phase 2 — Create:   Parse Gherkin → POST to Jira → return issue key + URL

State reads : user_input, conversation_history
State writes: retrieved_context, final_answer,
              jira_issue_created, conversation_history
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import config
from app.state import AgentState
from app.prompts import GHERKIN_PROMPT
from app.rag.vectorstore import load_vectorstore
from app.rag.retriever import retrieve_context
from app.tools.jira_create_tool import create_jira_issue_from_gherkin
from app.utils.logger import get_logger, format_history, strip_think_block
from app.utils.file_saver import save_output

logger = get_logger(__name__)


def _get_llm():
    backend = config.LLM_BACKEND.lower()
    if backend == "openai":
        return ChatOpenAI(model=config.OPENAI_MODEL,
                          api_key=config.OPENAI_API_KEY, temperature=0.2)
    elif backend == "anthropic":
        return ChatAnthropic(model=config.ANTHROPIC_MODEL,
                             api_key=config.ANTHROPIC_API_KEY, temperature=0.2)
    else:
        return ChatOllama(model=config.LLM_MODEL,
                          base_url=config.OLLAMA_BASE_URL, temperature=0.2)


def gherkin_node(state: AgentState) -> AgentState:
    """Phase 1: generate Gherkin. Phase 2: create Jira issue."""

    user_input = state.get("user_input", "")
    history    = state.get("conversation_history", [])

    logger.info(f"Gherkin node — requirement: {user_input[:80]}")
    print("\n" + "=" * 60)
    print("  GHERKIN ISSUE GENERATOR AGENT")
    print("=" * 60)
    print(f"[INFO] Feature request: {user_input[:100]}")

    try:
        # ── PHASE 1: Generate Gherkin text ────────────────────────────────
        print("\n[PHASE 1] Generating Gherkin specification...")

        print("[STEP 1] Loading knowledge base...")
        vectorstore = load_vectorstore()

        print("[STEP 2] Retrieving relevant context (MMR search)...")
        context_pieces = retrieve_context(vectorstore, user_input)
        context_text   = "\n\n---\n\n".join(context_pieces)
        history_text   = format_history(history)
        print(f"[OK]     Retrieved {len(context_pieces)} context chunks")

        print(f"\n[STEP 3] Generating Gherkin (model: {config.LLM_MODEL})...")
        print("-" * 60)

        llm   = _get_llm()
        chain = GHERKIN_PROMPT | llm

        gherkin_text = ""
        for chunk in chain.stream({
            "context":    context_text,
            "user_input": user_input,
            "history":    history_text,
        }):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            print(text, end="", flush=True)
            gherkin_text += text

        print("\n" + "-" * 60)
        gherkin_text = strip_think_block(gherkin_text)

    except Exception as e:
        logger.error(f"Gherkin generation (Phase 1) failed: {e}")
        return {**state, "final_answer": "", "jira_issue_created": None,
                "error": f"Gherkin generation failed: {str(e)}"}

    # ── PHASE 2: Create Jira issue ────────────────────────────────────────
    print("\n[PHASE 2] Creating Jira issue from Gherkin...")

    jira_result = create_jira_issue_from_gherkin(
        gherkin_text=gherkin_text,
        user_input=user_input,
        issue_type="Story",
        priority="Medium",
    )

    if jira_result["success"]:
        print(f"[OK] Jira issue created : {jira_result['issue_key']}")
        print(f"     URL               : {jira_result['issue_url']}")
        jira_footer = (
            f"\n\n---\n"
            f"**Jira Issue Created:** [{jira_result['issue_key']}]({jira_result['issue_url']})\n"
            f"**Summary:** {jira_result['summary']}\n"
        )
    else:
        print(f"[WARN] Jira issue NOT created: {jira_result['error']}")
        jira_footer = (
            f"\n\n---\n"
            f"**Note:** Jira issue was not created automatically.\n"
            f"**Reason:** {jira_result['error']}\n"
            f"Copy the Gherkin above and create the issue manually.\n"
        )

    final_answer = gherkin_text + jira_footer

    # ── Save output ───────────────────────────────────────────────────────
    saved = save_output(
        intent="gherkin",
        user_input=user_input,
        answer=final_answer,
        jira_result=jira_result,
    )
    if saved:
        print(f"\n[SAVED] {saved}")

    # ── Update conversation history ───────────────────────────────────────
    updated_history = history + [
        {"role": "user",      "content": user_input[:500]},
        {"role": "assistant", "content": gherkin_text[:500]},
    ]

    logger.info("Gherkin node completed")
    return {
        **state,
        "retrieved_context":    context_pieces,
        "final_answer":         final_answer,
        "jira_issue_created":   jira_result,
        "conversation_history": updated_history,
        "error":                None,
    }
