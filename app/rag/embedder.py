"""
embedder.py
===========
Provides the embedding model used to convert text chunks into vectors.

Default: HuggingFace all-MiniLM-L6-v2 (runs fully offline, no API key needed).
Override: set EMBEDDING_MODEL in .env to any HuggingFace model name.

Why all-MiniLM-L6-v2?
  - Fast on CPU (no GPU needed)
  - Small download (~80 MB)
  - Good retrieval quality for English enterprise text
  - Free, no rate limits, no API key

Usage:
    from app.rag.embedder import get_embeddings
    embeddings = get_embeddings()
"""

from langchain_huggingface import HuggingFaceEmbeddings

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Cache the embedding model so it is only loaded once per session
_embeddings_cache = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the HuggingFace embedding model (cached after first call).

    The model is downloaded from HuggingFace Hub on first use and
    cached locally by the sentence-transformers library.
    Subsequent calls return the in-memory cached instance.
    """
    global _embeddings_cache

    if _embeddings_cache is not None:
        logger.debug("Using cached embedding model")
        return _embeddings_cache

    model_name = config.EMBEDDING_MODEL
    logger.info(f"Loading embedding model: {model_name}")
    print(f"[INFO] Loading embedding model: {model_name}")
    print("[INFO] First run downloads ~80 MB. Subsequent runs use local cache.")

    _embeddings_cache = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded successfully")
    print(f"[OK] Embedding model ready: {model_name}")
    return _embeddings_cache
