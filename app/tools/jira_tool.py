"""
jira_tool.py
============
Fetches Jira issues using the Atlassian Python API and converts them
into LangChain Documents for embedding into ChromaDB.

Config (from .env):
  JIRA_URL         — https://yourcompany.atlassian.net
  JIRA_USERNAME    — your.email@company.com
  JIRA_API_TOKEN   — API token from https://id.atlassian.com/manage-profile/security
  JIRA_JQL         — JQL query to select issues (e.g. project=MYPROJECT AND updated>=-30d)
  JIRA_MAX_RESULTS — max issues to fetch (default 100)

Each Document contains:
  page_content : summary + description + labels + acceptance criteria
  metadata     : source_type, issue_key, title, url, status, labels, updated_at
"""

from typing import List

from atlassian import Jira
from langchain_core.documents import Document

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_jira_issues() -> List[Document]:
    """
    Fetch Jira issues via JQL and return as LangChain Documents.
    Returns empty list if Jira is not configured or connection fails.
    """
    if not config.JIRA_URL or not config.JIRA_API_TOKEN:
        logger.warning("Jira credentials not configured. Skipping Jira ingestion.")
        print("[SKIP] Jira not configured (JIRA_URL / JIRA_API_TOKEN missing in .env)")
        return []

    logger.info(f"Connecting to Jira: {config.JIRA_URL}")
    print(f"[INFO] Connecting to Jira: {config.JIRA_URL}")
    print(f"[INFO] JQL: {config.JIRA_JQL}")

    try:
        jira = Jira(
            url=config.JIRA_URL,
            username=config.JIRA_USERNAME,
            password=config.JIRA_API_TOKEN,
            cloud=True
        )

        results = jira.jql(
            jql=config.JIRA_JQL,
            limit=config.JIRA_MAX_RESULTS,
            fields=["summary", "description", "labels", "status",
                    "issuetype", "priority", "updated", "assignee",
                    "reporter", "comment"]
        )

        issues = results.get("issues", [])
        logger.info(f"Fetched {len(issues)} Jira issues")
        print(f"[INFO] Fetched {len(issues)} issues from Jira")

        documents = []
        for issue in issues:
            doc = _issue_to_document(issue)
            if doc:
                documents.append(doc)

        logger.info(f"Converted {len(documents)} Jira issues to Documents")
        return documents

    except Exception as e:
        logger.error(f"Jira ingestion failed: {e}")
        print(f"[WARN] Jira ingestion failed: {e}")
        print("[INFO] Continuing without Jira data.")
        return []


def _issue_to_document(issue: dict) -> Document:
    """Convert a single Jira issue dict to a LangChain Document."""
    try:
        fields      = issue.get("fields", {})
        issue_key   = issue.get("key", "")
        summary     = fields.get("summary", "")
        description = fields.get("description", "") or ""
        labels      = fields.get("labels", [])
        status      = fields.get("status", {}).get("name", "")
        issue_type  = fields.get("issuetype", {}).get("name", "")
        priority    = fields.get("priority", {}).get("name", "")
        updated_at  = fields.get("updated", "")

        # Extract comments (first 3) for richer context
        comments_raw = fields.get("comment", {}).get("comments", [])[:3]
        comments_text = ""
        for c in comments_raw:
            author  = c.get("author", {}).get("displayName", "Unknown")
            body    = c.get("body", "")[:300]
            comments_text += f"\nComment by {author}: {body}"

        page_content = (
            f"Jira Issue: {issue_key}\n"
            f"Type: {issue_type} | Status: {status} | Priority: {priority}\n"
            f"Summary: {summary}\n"
            f"Labels: {', '.join(labels) if labels else 'none'}\n"
            f"Description:\n{description[:2000]}\n"
        )
        if comments_text:
            page_content += f"\nComments:{comments_text}"

        metadata = {
            "source_type":   "jira",
            "source_system": "jira_cloud",
            "issue_key":     issue_key,
            "title":         f"{issue_key}: {summary}",
            "url":           f"{config.JIRA_URL}/browse/{issue_key}",
            "labels":        ", ".join(labels),
            "status":        status,
            "issue_type":    issue_type,
            "updated_at":    updated_at,
            "page_id":       None,
            "author":        None,
            "content_type":  "jira_issue",
        }

        return Document(page_content=page_content, metadata=metadata)

    except Exception as e:
        logger.error(f"Failed to convert issue to Document: {e}")
        return None
