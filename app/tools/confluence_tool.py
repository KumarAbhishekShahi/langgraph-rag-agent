"""
confluence_tool.py
==================
Fetches Confluence pages from a space and converts them to LangChain Documents.

Config (from .env):
  CONFLUENCE_URL        — https://yourcompany.atlassian.net
  CONFLUENCE_USERNAME   — your.email@company.com
  CONFLUENCE_API_TOKEN  — API token
  CONFLUENCE_SPACE_KEY  — e.g. "MYSPACE" or "~username" for personal space
  CONFLUENCE_MAX_PAGES  — max pages to fetch (default 50)

Each Document contains:
  page_content : page title + plain-text body (HTML stripped)
  metadata     : source_type, page_id, title, url, author, updated_at
"""

from typing import List

from atlassian import Confluence
from bs4 import BeautifulSoup
from langchain_core.documents import Document

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_confluence_pages() -> List[Document]:
    """
    Fetch Confluence pages from the configured space.
    Returns empty list if Confluence is not configured or connection fails.
    """
    if not config.CONFLUENCE_URL or not config.CONFLUENCE_API_TOKEN:
        logger.warning("Confluence credentials not configured. Skipping.")
        print("[SKIP] Confluence not configured (CONFLUENCE_URL / CONFLUENCE_API_TOKEN missing)")
        return []

    logger.info(f"Connecting to Confluence: {config.CONFLUENCE_URL}")
    print(f"[INFO] Connecting to Confluence space: {config.CONFLUENCE_SPACE_KEY}")

    try:
        confluence = Confluence(
            url=config.CONFLUENCE_URL,
            username=config.CONFLUENCE_USERNAME,
            password=config.CONFLUENCE_API_TOKEN,
            cloud=True
        )

        results = confluence.get_all_pages_from_space(
            space=config.CONFLUENCE_SPACE_KEY,
            start=0,
            limit=config.CONFLUENCE_MAX_PAGES,
            expand="body.storage,version,history"
        )

        logger.info(f"Fetched {len(results)} Confluence pages")
        print(f"[INFO] Fetched {len(results)} pages from Confluence")

        documents = []
        for page in results:
            doc = _page_to_document(page, confluence)
            if doc:
                documents.append(doc)

        logger.info(f"Converted {len(documents)} Confluence pages to Documents")
        return documents

    except Exception as e:
        logger.error(f"Confluence ingestion failed: {e}")
        print(f"[WARN] Confluence ingestion failed: {e}")
        print("[INFO] Continuing without Confluence data.")
        return []


def _page_to_document(page: dict, confluence: Confluence) -> Document:
    """Convert a single Confluence page dict to a LangChain Document."""
    try:
        page_id    = page.get("id", "")
        title      = page.get("title", "Untitled")
        space_key  = page.get("space", {}).get("key", config.CONFLUENCE_SPACE_KEY)
        url        = f"{config.CONFLUENCE_URL}/wiki/spaces/{space_key}/pages/{page_id}"
        updated_at = page.get("version", {}).get("when", "")
        author     = page.get("history", {}).get("createdBy", {}).get("displayName", "")

        # Extract HTML body and strip tags to plain text
        html_body = (
            page.get("body", {})
                .get("storage", {})
                .get("value", "")
        )
        if html_body:
            soup      = BeautifulSoup(html_body, "lxml")
            body_text = soup.get_text(separator="\n", strip=True)
        else:
            body_text = ""

        if not body_text.strip():
            logger.debug(f"Skipping empty Confluence page: {title}")
            return None

        page_content = (
            f"Confluence Page: {title}\n"
            f"Space: {space_key}\n"
            f"Author: {author}\n"
            f"\n{body_text[:3000]}"
        )

        metadata = {
            "source_type":   "confluence",
            "source_system": "confluence_cloud",
            "page_id":       page_id,
            "title":         title,
            "url":           url,
            "author":        author,
            "updated_at":    updated_at,
            "issue_key":     None,
            "labels":        "",
            "content_type":  "confluence_page",
        }

        return Document(page_content=page_content, metadata=metadata)

    except Exception as e:
        logger.error(f"Failed to convert Confluence page to Document: {e}")
        return None
