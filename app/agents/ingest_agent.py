"""
ingest_agent.py
===============
LangGraph node: loads all documents from all sources and builds ChromaDB.

This is always the FIRST thing you run before using any other agent.
It:
  1. Calls all 5 source loaders (Jira, Confluence, KB, files, codebase)
  2. Chunks every document with RecursiveCharacterTextSplitter
  3. Generates embeddings with all-MiniLM-L6-v2 (local, offline)
  4. Persists vectors to ChromaDB at CHROMA_PERSIST_DIR

State changes:
  final_answer  → ingestion summary string
  error         → set if ingestion fails completely
"""

from app.state import AgentState
from app.rag.loader import load_all_documents
from app.rag.vectorstore import build_vectorstore
from app.utils.logger import get_logger
from app.utils.file_saver import save_output

logger = get_logger(__name__)


def ingest_node(state: AgentState) -> AgentState:
    """Run full ingestion pipeline and persist to ChromaDB."""

    logger.info("Starting ingestion node")
    print("\n" + "=" * 60)
    print("  INGESTION AGENT")
    print("=" * 60)

    try:
        # ── Load from all sources ─────────────────────────────────────────
        documents = load_all_documents()

        if not documents:
            msg = (
                "No documents found from any source.\n"
                "Check your .env settings:\n"
                "  - JIRA_URL, CONFLUENCE_URL (leave empty to skip)\n"
                "  - KB_ARTICLES_PATH (default: ./kb_articles)\n"
                "  - SAMPLE_DATA_PATH (default: ./sample_data)\n"
                "  - CODEBASE_PATH (leave empty to skip)"
            )
            logger.warning(msg)
            print(f"\n[WARN] {msg}")
            return {**state, "final_answer": msg, "error": None}

        # ── Build vectorstore ─────────────────────────────────────────────
        build_vectorstore(documents)

        summary = (
            f"Ingestion complete.\n"
            f"Total documents loaded : {len(documents)}\n"
            f"ChromaDB location      : {__import__('config').CHROMA_PERSIST_DIR}\n"
            f"Collection             : {__import__('config').CHROMA_COLLECTION_NAME}\n\n"
            f"You can now use:\n"
            f"  Option 2 — Analyze a requirement\n"
            f"  Option 3 — Generate a Gherkin issue\n"
            f"  Option 4 — Generate code for a Jira issue"
        )

        print("\n[OK] Ingestion complete.")
        logger.info("Ingestion node completed successfully")

        # Save ingestion summary to outputs/
        save_output(intent="ingest", user_input="", answer=summary)

        return {**state, "final_answer": summary, "error": None}

    except Exception as e:
        logger.error(f"Ingestion node failed: {e}")
        return {
            **state,
            "final_answer": "",
            "error": f"Ingestion failed: {str(e)}"
        }
