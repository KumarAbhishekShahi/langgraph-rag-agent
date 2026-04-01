"""
retriever.py
============
Retrieves the most relevant document chunks from ChromaDB for a given query.

Uses MMR (Maximal Marginal Relevance) search to:
  - Return chunks that are relevant to the query
  - AND diverse from each other (avoids returning the same info 5 times)

Config:
  RETRIEVAL_K           — number of chunks to return (default 5)
  RETRIEVAL_FETCH_K     — candidates to consider before MMR filtering (default 20)
  RETRIEVAL_LAMBDA_MULT — balance relevance vs diversity 0.0–1.0 (default 0.5)

Usage:
    from app.rag.retriever import retrieve_context

    chunks = retrieve_context(vectorstore, "OAuth2 SSO integration requirements")
    # Returns: list of text strings (one per chunk)
"""

from typing import List

from langchain_chroma import Chroma

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_context(
    vectorstore: Chroma,
    query: str,
    k: int = None,
    source_filter: dict = None,
) -> List[str]:
    """
    Run MMR search and return top-k chunk texts as a list of strings.

    Each returned string includes a source header:
        [Source 1: <title> | <source_type> | <system>]
        <chunk text>

    Args:
        vectorstore   : loaded Chroma instance
        query         : search query text
        k             : number of chunks (default: config.RETRIEVAL_K)
        source_filter : optional ChromaDB metadata filter dict
                        e.g. {"source_type": "codebase"}

    Returns:
        List of formatted context strings, ready for prompt injection.
        Returns ["No relevant context found."] if nothing matches.
    """
    k = k or config.RETRIEVAL_K

    if not query or not query.strip():
        logger.warning("Empty query passed to retrieve_context")
        return ["No relevant context found."]

    logger.info(f"Retrieving context — query: '{query[:60]}...' k={k}")

    search_kwargs = {
        "k":           k,
        "fetch_k":     config.RETRIEVAL_FETCH_K,
        "lambda_mult": config.RETRIEVAL_LAMBDA_MULT,
    }
    if source_filter:
        search_kwargs["filter"] = source_filter

    try:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )
        docs = retriever.invoke(query)

    except Exception as e:
        logger.error(f"MMR retrieval failed: {e}. Falling back to similarity search.")
        try:
            docs = vectorstore.similarity_search(query, k=k)
        except Exception as e2:
            logger.error(f"Similarity search also failed: {e2}")
            return ["No relevant context found."]

    if not docs:
        logger.info("No documents retrieved for query")
        return ["No relevant context found."]

    context_pieces = []
    for i, doc in enumerate(docs):
        meta        = doc.metadata or {}
        title       = meta.get("title",       "Unknown")
        source_type = meta.get("source_type", "unknown")
        system      = meta.get("source_system", "")
        issue_key   = meta.get("issue_key",   "")

        # Build a header so the LLM knows where each chunk came from
        header_parts = [f"Source {i+1}: {title}", source_type]
        if system:
            header_parts.append(system)
        if issue_key:
            header_parts.append(f"issue={issue_key}")

        header = "[" + " | ".join(header_parts) + "]"
        context_pieces.append(f"{header}\n{doc.page_content}")

    logger.info(f"Retrieved {len(context_pieces)} context chunks")
    return context_pieces


def retrieve_codebase_context(
    vectorstore: Chroma,
    query: str,
    k: int = 8,
) -> List[str]:
    """
    Retrieve only codebase source files (source_type = "codebase").
    Used by code_agent to find relevant source code files.

    Args:
        vectorstore : loaded Chroma instance
        query       : search query (issue summary + description)
        k           : number of codebase chunks to retrieve (default 8)

    Returns:
        List of formatted code context strings.
    """
    logger.info(f"Retrieving codebase context — query: '{query[:60]}...'")

    try:
        return retrieve_context(
            vectorstore=vectorstore,
            query=query,
            k=k,
            source_filter={"source_type": "codebase"},
        )
    except Exception:
        # Fallback: retrieve without filter, then manually filter
        logger.warning("Metadata filter not supported. Filtering manually.")
        all_chunks = retrieve_context(vectorstore=vectorstore, query=query, k=k * 3)
        code_chunks = [c for c in all_chunks if "source_type: codebase" in c.lower()
                       or "// FILE:" in c or "// LANGUAGE:" in c]
        return code_chunks[:k] if code_chunks else []
