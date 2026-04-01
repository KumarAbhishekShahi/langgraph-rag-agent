"""
test_retriever.py
=================
Unit tests for retrieve_context (mock vectorstore — no ChromaDB needed).
"""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestRetriever:

    def _make_vectorstore(self, docs):
        """Build a minimal mock vectorstore that returns given docs."""
        vs = MagicMock()
        retriever = MagicMock()
        retriever.invoke.return_value = docs
        vs.as_retriever.return_value  = retriever
        vs.similarity_search.return_value = docs
        return vs

    def test_returns_formatted_strings(self):
        from app.rag.retriever import retrieve_context
        docs = [
            Document(
                page_content="OAuth2 requires PKCE for public clients.",
                metadata={"title": "SSO Guide", "source_type": "kb_article",
                          "source_system": "local_kb", "issue_key": ""}
            )
        ]
        vs     = self._make_vectorstore(docs)
        result = retrieve_context(vs, "OAuth2 SSO", k=1)
        assert len(result) == 1
        assert "Source 1" in result[0]
        assert "OAuth2" in result[0]

    def test_empty_query_returns_fallback(self):
        from app.rag.retriever import retrieve_context
        vs     = self._make_vectorstore([])
        result = retrieve_context(vs, "   ", k=3)
        assert result == ["No relevant context found."]

    def test_no_docs_returns_fallback(self):
        from app.rag.retriever import retrieve_context
        vs     = self._make_vectorstore([])
        result = retrieve_context(vs, "some query", k=5)
        assert result == ["No relevant context found."]

    def test_multiple_chunks_returned(self):
        from app.rag.retriever import retrieve_context
        docs = [
            Document(page_content=f"Chunk {i}",
                     metadata={"title": f"Doc{i}", "source_type": "jira",
                                "source_system": "jira_cloud", "issue_key": f"PROJ-{i}"})
            for i in range(5)
        ]
        vs     = self._make_vectorstore(docs)
        result = retrieve_context(vs, "payment processing", k=5)
        assert len(result) == 5
        for i, chunk in enumerate(result):
            assert f"Source {i+1}" in chunk
