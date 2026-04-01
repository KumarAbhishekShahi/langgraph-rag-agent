"""
config.py
=========
Centralised configuration for the LangGraph Enterprise RAG Agent.
All settings are loaded from environment variables (.env file).
Never hardcode credentials here.

Usage:
    import config
    print(config.LLM_MODEL)
    print(config.JIRA_URL)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Backend ───────────────────────────────────────────────────────────────
LLM_BACKEND     = os.getenv("LLM_BACKEND",     "ollama")
LLM_MODEL       = os.getenv("LLM_MODEL",       "deepseek-r1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "gpt-4o")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL",   "claude-3-5-sonnet-20241022")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR     = os.getenv("CHROMA_PERSIST_DIR",     "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_agent")

# ── Text Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Jira ──────────────────────────────────────────────────────────────────────
JIRA_URL         = os.getenv("JIRA_URL",         "")
JIRA_USERNAME    = os.getenv("JIRA_USERNAME",    "")
JIRA_API_TOKEN   = os.getenv("JIRA_API_TOKEN",   "")
JIRA_JQL         = os.getenv("JIRA_JQL",         "project = MYPROJECT AND updated >= -30d")
JIRA_MAX_RESULTS = int(os.getenv("JIRA_MAX_RESULTS", "100"))
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "")

# ── Confluence ────────────────────────────────────────────────────────────────
CONFLUENCE_URL       = os.getenv("CONFLUENCE_URL",       "")
CONFLUENCE_USERNAME  = os.getenv("CONFLUENCE_USERNAME",  "")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "")
CONFLUENCE_MAX_PAGES = int(os.getenv("CONFLUENCE_MAX_PAGES", "50"))

# ── Local Paths ───────────────────────────────────────────────────────────────
KB_ARTICLES_PATH = os.getenv("KB_ARTICLES_PATH", "./kb_articles")
SAMPLE_DATA_PATH = os.getenv("SAMPLE_DATA_PATH", "./sample_data")
CODEBASE_PATH    = os.getenv("CODEBASE_PATH",    "")

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K           = int(os.getenv("RETRIEVAL_K",           "5"))
RETRIEVAL_FETCH_K     = int(os.getenv("RETRIEVAL_FETCH_K",     "20"))
RETRIEVAL_LAMBDA_MULT = float(os.getenv("RETRIEVAL_LAMBDA_MULT", "0.5"))

# ── Scheduler ─────────────────────────────────────────────────────────────────
SCHEDULE_INTERVAL_HOURS = int(os.getenv("SCHEDULE_INTERVAL_HOURS", "6"))

# ── Gradio UI ─────────────────────────────────────────────────────────────────
GRADIO_PORT  = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
