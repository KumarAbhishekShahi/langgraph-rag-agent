"""
requirement_agent.py
====================
LangGraph node: deep-analyzes a new requirement using RAG context.

Flow:
  1. Retrieve top-k relevant chunks from ChromaDB (MMR search)
  2. Build prompt with context + requirement
  3. Stream LLM response word-by-word (Step 11)
  4. Save output to outputs/ folder (Step 12)
  5. Append turn to conversation history (Step 14)

State reads : user_input, conversation_history
State writes: retrieved_context, final_answer, conversation_history
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import config
from app.state import AgentState
from app.prompts import ANALYSIS_PROMPT
from app.rag.vectorstore import load_vectorstore
from app.rag.retriever import retrieve_context
from app.utils.logger import get_logger, format_history, strip_think_block
from app.utils.file_saver import save_output

logger = get_logger(__name__)


def _get_llm():
    """Return the configured LLM instance."""
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


def analysis_node(state: AgentState) -> AgentState:
    """Retrieve context and run 15-section requirement analysis."""

    user_input = state.get("user_input", "")
    history    = state.get("conversation_history", [])

    logger.info(f"Analysis node — requirement: {user_input[:80]}")
    print("\n" + "=" * 60)
    print("  REQUIREMENT ANALYSIS AGENT")
    print("=" * 60)
    print(f"[INFO] Requirement: {user_input[:100]}")

    try:
        # ── Step 1: Retrieve context ──────────────────────────────────────
        print("\n[STEP 1] Loading knowledge base...")
        vectorstore = load_vectorstore()

        print("[STEP 2] Retrieving relevant context (MMR search)...")
        context_pieces = retrieve_context(vectorstore, user_input)
        context_text   = "\n\n---\n\n".join(context_pieces)
        history_text   = format_history(history)

        print(f"[OK]     Retrieved {len(context_pieces)} context chunks")

        # ── Step 2: Stream LLM response ───────────────────────────────────
        print(f"\n[STEP 3] Generating analysis (model: {config.LLM_MODEL})...")
        print("-" * 60)

        llm   = _get_llm()
        chain = ANALYSIS_PROMPT | llm

        answer = ""
        for chunk in chain.stream({
            "context":    context_text,
            "user_input": user_input,
            "history":    history_text,
        }):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            print(text, end="", flush=True)
            answer += text

        print("\n" + "-" * 60)

        # Strip DeepSeek-R1 think blocks from stored answer
        answer = strip_think_block(answer)

        # ── Step 3: Save output ───────────────────────────────────────────
        saved = save_output(intent="analyze", user_input=user_input, answer=answer)
        if saved:
            print(f"\n[SAVED] {saved}")

        # ── Step 4: Update conversation history ───────────────────────────
        updated_history = history + [
            {"role": "user",      "content": user_input[:500]},
            {"role": "assistant", "content": answer[:500]},
        ]

        logger.info("Analysis node completed")
        return {
            **state,
            "retrieved_context":   context_pieces,
            "final_answer":        answer,
            "conversation_history": updated_history,
            "error":               None,
        }

    except Exception as e:
        logger.error(f"Analysis node failed: {e}")
        return {**state, "final_answer": "", "error": f"Analysis failed: {str(e)}"}
