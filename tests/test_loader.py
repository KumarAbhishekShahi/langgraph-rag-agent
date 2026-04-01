"""
test_loader.py
==============
Unit tests for the RAG document loading pipeline.

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # concise tracebacks
"""

import csv
import io
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document


# ── test: kb_tool ─────────────────────────────────────────────────────────────

class TestKbTool:
    """Tests for app/tools/kb_tool.py"""

    def test_loads_markdown_files(self, tmp_path):
        """KB tool should load .md files as Documents."""
        md_file = tmp_path / "guide.md"
        md_file.write_text("# OAuth2 Guide\n\nThis is a test guide.", encoding="utf-8")

        import config
        original = config.KB_ARTICLES_PATH
        config.KB_ARTICLES_PATH = str(tmp_path)

        from app.tools.kb_tool import fetch_kb_articles
        docs = fetch_kb_articles()

        config.KB_ARTICLES_PATH = original

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "OAuth2" in docs[0].page_content
        assert docs[0].metadata["source_type"] == "kb_article"

    def test_loads_html_files(self, tmp_path):
        """KB tool should strip HTML tags from .html files."""
        html_file = tmp_path / "faq.html"
        html_file.write_text(
            "<html><body><h1>FAQ</h1><p>How do I reset?</p></body></html>",
            encoding="utf-8"
        )

        import config
        original = config.KB_ARTICLES_PATH
        config.KB_ARTICLES_PATH = str(tmp_path)

        from importlib import reload
        import app.tools.kb_tool as kb_mod
        reload(kb_mod)
        docs = kb_mod.fetch_kb_articles()

        config.KB_ARTICLES_PATH = original

        assert len(docs) == 1
        assert "<html>" not in docs[0].page_content
        assert "FAQ" in docs[0].page_content or "reset" in docs[0].page_content

    def test_skips_empty_files(self, tmp_path):
        """KB tool should skip files with no content."""
        (tmp_path / "empty.md").write_text("   \n\n   ", encoding="utf-8")
        (tmp_path / "real.md").write_text("# Real Content\n\nSome text.", encoding="utf-8")

        import config
        original = config.KB_ARTICLES_PATH
        config.KB_ARTICLES_PATH = str(tmp_path)

        from importlib import reload
        import app.tools.kb_tool as kb_mod
        reload(kb_mod)
        docs = kb_mod.fetch_kb_articles()

        config.KB_ARTICLES_PATH = original
        assert len(docs) == 1

    def test_returns_empty_if_path_missing(self):
        """KB tool should return [] if KB_ARTICLES_PATH does not exist."""
        import config
        original = config.KB_ARTICLES_PATH
        config.KB_ARTICLES_PATH = "/nonexistent/path/12345"

        from importlib import reload
        import app.tools.kb_tool as kb_mod
        reload(kb_mod)
        docs = kb_mod.fetch_kb_articles()

        config.KB_ARTICLES_PATH = original
        assert docs == []


# ── test: file_tool ───────────────────────────────────────────────────────────

class TestFileTool:
    """Tests for app/tools/file_tool.py"""

    def test_loads_txt_file(self, tmp_path):
        (tmp_path / "requirements.txt").write_text(
            "The system shall support 10,000 concurrent users.", encoding="utf-8"
        )
        import config
        original = config.SAMPLE_DATA_PATH
        config.SAMPLE_DATA_PATH = str(tmp_path)

        from importlib import reload
        import app.tools.file_tool as ft
        reload(ft)
        docs = ft.fetch_local_files()

        config.SAMPLE_DATA_PATH = original
        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "local_file"
        assert "concurrent" in docs[0].page_content

    def test_loads_csv_file(self, tmp_path):
        csv_content = "id,name,price\n1,Widget,9.99\n2,Gadget,19.99\n"
        (tmp_path / "products.csv").write_text(csv_content, encoding="utf-8")

        import config
        original = config.SAMPLE_DATA_PATH
        config.SAMPLE_DATA_PATH = str(tmp_path)

        from importlib import reload
        import app.tools.file_tool as ft
        reload(ft)
        docs = ft.fetch_local_files()

        config.SAMPLE_DATA_PATH = original
        assert len(docs) == 1
        assert "Widget" in docs[0].page_content
        assert docs[0].metadata["content_type"] == "csv"

    def test_returns_empty_if_path_missing(self):
        import config
        original = config.SAMPLE_DATA_PATH
        config.SAMPLE_DATA_PATH = "/nonexistent/sample/data"

        from importlib import reload
        import app.tools.file_tool as ft
        reload(ft)
        docs = ft.fetch_local_files()

        config.SAMPLE_DATA_PATH = original
        assert docs == []


# ── test: logger utils ────────────────────────────────────────────────────────

class TestLoggerUtils:
    """Tests for app/utils/logger.py helpers."""

    def test_format_history_empty(self):
        from app.utils.logger import format_history
        result = format_history([])
        assert result == "No prior conversation."

    def test_format_history_recent_turns(self):
        from app.utils.logger import format_history
        history = [
            {"role": "user",      "content": "Add SSO"},
            {"role": "assistant", "content": "Here is the analysis..."},
            {"role": "user",      "content": "Generate Gherkin"},
        ]
        result = format_history(history, max_turns=4)
        assert "USER" in result
        assert "ASSISTANT" in result
        assert "SSO" in result

    def test_strip_think_block(self):
        from app.utils.logger import strip_think_block
        raw = "<think>\nI need to think about this...\n</think>\nHere is my answer."
        result = strip_think_block(raw)
        assert "<think>" not in result
        assert "Here is my answer." in result

    def test_strip_think_block_no_tag(self):
        from app.utils.logger import strip_think_block
        raw = "Simple answer without think block."
        assert strip_think_block(raw) == raw


# ── test: jira_create_tool helpers ───────────────────────────────────────────

class TestJiraCreateHelpers:
    """Unit tests for parsing helpers in jira_create_tool.py"""

    def test_extract_feature_name(self):
        from app.tools.jira_create_tool import _extract_feature_name
        gherkin = "Feature: OAuth2 SSO Login\n\nScenario: User logs in"
        assert _extract_feature_name(gherkin) == "OAuth2 SSO Login"

    def test_extract_labels_auth(self):
        from app.tools.jira_create_tool import _extract_labels
        labels = _extract_labels("OAuth2 SSO login with JWT token", "user login")
        assert "auth" in labels

    def test_extract_labels_payment(self):
        from app.tools.jira_create_tool import _extract_labels
        labels = _extract_labels("Stripe payment billing invoice", "pay now")
        assert "payment" in labels

    def test_jira_create_skipped_if_not_configured(self):
        """create_jira_issue_from_gherkin should fail gracefully if no credentials."""
        import config
        orig_url   = config.JIRA_URL
        orig_token = config.JIRA_API_TOKEN
        config.JIRA_URL       = ""
        config.JIRA_API_TOKEN = ""

        from app.tools.jira_create_tool import create_jira_issue_from_gherkin
        result = create_jira_issue_from_gherkin("Feature: Test\n", "test input")

        config.JIRA_URL       = orig_url
        config.JIRA_API_TOKEN = orig_token

        assert result["success"] is False
        assert result["issue_key"] is None
