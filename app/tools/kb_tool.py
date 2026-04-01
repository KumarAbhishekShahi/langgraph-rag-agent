"""
kb_tool.py
==========
Reads local Knowledge Base articles from the kb_articles/ folder.

Supported file types: .md  .html  .txt
Sub-folders are scanned recursively.

Config (from .env):
  KB_ARTICLES_PATH — path to kb_articles folder (default: ./kb_articles)

Each Document contains:
  page_content : file content (HTML stripped to plain text for .html)
  metadata     : source_type="kb_article", title, path, content_type
"""

from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from langchain_core.documents import Document

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_KB_EXTENSIONS = {".md", ".html", ".htm", ".txt"}


def fetch_kb_articles() -> List[Document]:
    """
    Read all KB article files from KB_ARTICLES_PATH.
    Returns empty list if path is not configured or does not exist.
    """
    kb_path = Path(config.KB_ARTICLES_PATH)

    if not kb_path.exists():
        logger.warning(f"KB articles path not found: {kb_path}. Skipping.")
        print(f"[SKIP] KB articles folder not found: {kb_path}")
        return []

    files = [
        f for f in kb_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_KB_EXTENSIONS
    ]

    logger.info(f"Found {len(files)} KB article files in {kb_path}")
    print(f"[INFO] Found {len(files)} KB articles in: {kb_path}")

    documents = []
    for filepath in sorted(files):
        doc = _file_to_document(filepath, kb_path)
        if doc:
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} KB article Documents")
    return documents


def _file_to_document(filepath: Path, root: Path) -> Document:
    """Read a single KB file and return a LangChain Document."""
    try:
        raw = filepath.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            return None

        ext = filepath.suffix.lower()
        if ext in (".html", ".htm"):
            soup    = BeautifulSoup(raw, "lxml")
            content = soup.get_text(separator="\n", strip=True)
        else:
            content = raw

        rel_path = str(filepath.relative_to(root))
        title    = filepath.stem.replace("-", " ").replace("_", " ").title()

        page_content = f"KB Article: {title}\nFile: {rel_path}\n\n{content[:3000]}"

        metadata = {
            "source_type":   "kb_article",
            "source_system": "local_kb",
            "title":         title,
            "path":          rel_path,
            "url":           None,
            "issue_key":     None,
            "page_id":       None,
            "author":        None,
            "labels":        "",
            "updated_at":    "",
            "content_type":  f"kb_{ext.strip('.')}",
        }

        return Document(page_content=page_content, metadata=metadata)

    except Exception as e:
        logger.error(f"Failed to read KB file {filepath}: {e}")
        return None
