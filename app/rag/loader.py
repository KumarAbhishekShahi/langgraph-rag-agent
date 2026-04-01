"""
loader.py
=========
Orchestrates all document sources and returns a unified list of Documents
ready for embedding and storage in ChromaDB.

Sources (in order):
  1. Jira issues          → app/tools/jira_tool.py
  2. Confluence pages     → app/tools/confluence_tool.py
  3. KB articles          → app/tools/kb_tool.py
  4. Local sample files   → app/tools/file_tool.py
  5. Codebase source files → app/tools/codebase_tool.py

Each source loader returns List[Document] with normalised metadata:
  source_type    : "jira" | "confluence" | "kb_article" | "local_file" | "codebase"
  source_system  : e.g. "jira_cloud", "confluence_cloud", "local_kb"
  title          : document/page/file title
  url            : browser URL (None for local files)
  issue_key      : Jira issue key (Jira source only)

Usage:
    from app.rag.loader import load_all_documents
    documents = load_all_documents()
    # Returns: List[Document] — all sources merged
"""

from typing import List

from langchain_core.documents import Document

from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_all_documents() -> List[Document]:
    """
    Run all five source loaders and return a combined document list.

    Each loader is called independently. If one source fails (e.g. Jira
    credentials not configured), it logs a warning and returns an empty
    list — other sources continue unaffected.

    Returns:
        List[Document] — merged from all configured sources.
        May be empty if no sources are configured.
    """

    # Import tools here (not at top) to allow individual tool failures
    # without crashing the entire loader on import
    from app.tools.jira_tool       import fetch_jira_issues
    from app.tools.confluence_tool import fetch_confluence_pages
    from app.tools.kb_tool         import fetch_kb_articles
    from app.tools.file_tool       import fetch_local_files
    from app.tools.codebase_tool   import fetch_codebase_files

    logger.info("=" * 55)
    logger.info("Starting document ingestion from all sources")
    logger.info("=" * 55)

    all_docs: List[Document] = []

    # ── Source 1: Jira ────────────────────────────────────────────────────────
    print("\n[SOURCE 1/5] Jira Issues")
    print("-" * 40)
    jira_docs = fetch_jira_issues()
    all_docs.extend(jira_docs)
    print(f"[OK] Loaded {len(jira_docs)} Jira issues")

    # ── Source 2: Confluence ──────────────────────────────────────────────────
    print("\n[SOURCE 2/5] Confluence Pages")
    print("-" * 40)
    conf_docs = fetch_confluence_pages()
    all_docs.extend(conf_docs)
    print(f"[OK] Loaded {len(conf_docs)} Confluence pages")

    # ── Source 3: KB Articles ─────────────────────────────────────────────────
    print("\n[SOURCE 3/5] KB Articles")
    print("-" * 40)
    kb_docs = fetch_kb_articles()
    all_docs.extend(kb_docs)
    print(f"[OK] Loaded {len(kb_docs)} KB articles")

    # ── Source 4: Local Files ─────────────────────────────────────────────────
    print("\n[SOURCE 4/5] Local Files (PDF, HTML, TXT, CSV)")
    print("-" * 40)
    file_docs = fetch_local_files()
    all_docs.extend(file_docs)
    print(f"[OK] Loaded {len(file_docs)} local files")

    # ── Source 5: Codebase ────────────────────────────────────────────────────
    print("\n[SOURCE 5/5] Codebase Source Files")
    print("-" * 40)
    code_docs = fetch_codebase_files()
    all_docs.extend(code_docs)
    print(f"[OK] Loaded {len(code_docs)} codebase files")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  TOTAL DOCUMENTS LOADED: {len(all_docs)}")
    print(f"    Jira        : {len(jira_docs)}")
    print(f"    Confluence  : {len(conf_docs)}")
    print(f"    KB Articles : {len(kb_docs)}")
    print(f"    Local Files : {len(file_docs)}")
    print(f"    Codebase    : {len(code_docs)}")
    print("=" * 55)

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs
