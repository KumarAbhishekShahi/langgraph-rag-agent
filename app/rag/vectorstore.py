"""
vectorstore.py
==============
Manages the ChromaDB vector store — building it from documents and loading it.

Two operations:
  build_vectorstore(documents)  — chunk → embed → persist to disk
  load_vectorstore()            — load existing ChromaDB from disk

ChromaDB stores vectors in CHROMA_PERSIST_DIR (default: ./chroma_db).
This folder is gitignored — rebuild it after cloning the repo.

Usage:
    from app.rag.vectorstore import build_vectorstore, load_vectorstore

    # Build (during ingestion)
    docs = load_all_documents()
    build_vectorstore(docs)

    # Load (during query/analysis/gherkin/code)
    vectorstore = load_vectorstore()
"""

from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from app.rag.embedder import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_vectorstore(documents: List[Document]) -> Chroma:
    """
    Chunk all documents, generate embeddings, and persist to ChromaDB.

    Steps:
      1. Split each Document into overlapping chunks
         (CHUNK_SIZE / CHUNK_OVERLAP from config)
      2. Generate vector embeddings for each chunk
      3. Store in ChromaDB at CHROMA_PERSIST_DIR
      4. Return the Chroma instance

    Args:
        documents : list of LangChain Document objects from all sources

    Returns:
        Chroma vectorstore instance
    """
    if not documents:
        logger.warning("build_vectorstore called with empty document list")
        print("[WARN] No documents to embed. Skipping vectorstore build.")
        return None

    logger.info(f"Building vectorstore from {len(documents)} documents")
    print(f"\n[STEP] Chunking {len(documents)} documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks "
                f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    print(f"[OK] Created {len(chunks)} chunks from {len(documents)} documents")

    print("[STEP] Generating embeddings and storing in ChromaDB...")
    print(f"[INFO] Persist path: {config.CHROMA_PERSIST_DIR}")
    print("[INFO] This may take a few minutes on first run...")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    logger.info(f"Vectorstore built and persisted at: {config.CHROMA_PERSIST_DIR}")
    print(f"[OK] ChromaDB vectorstore saved to: {config.CHROMA_PERSIST_DIR}")
    print(f"[OK] Collection: {config.CHROMA_COLLECTION_NAME}")
    print(f"[OK] Total vectors stored: {len(chunks)}")

    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Load an existing ChromaDB vectorstore from disk.

    Raises:
        RuntimeError if the persist directory does not exist
        (run ingestion first to build the vectorstore).

    Returns:
        Chroma vectorstore instance ready for similarity search
    """
    import os

    persist_dir = config.CHROMA_PERSIST_DIR

    if not os.path.exists(persist_dir):
        raise RuntimeError(
            f"ChromaDB not found at: {persist_dir}\n"
            f"Run ingestion first: choose option 1 in main.py"
        )

    logger.info(f"Loading vectorstore from: {persist_dir}")

    embeddings  = get_embeddings()
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    count = vectorstore._collection.count()
    logger.info(f"Vectorstore loaded — {count} vectors in collection")
    print(f"[OK] ChromaDB loaded: {count} vectors in '{config.CHROMA_COLLECTION_NAME}'")

    return vectorstore


def add_documents(vectorstore: Chroma, documents: List[Document]) -> int:
    """
    Add new documents to an existing vectorstore without rebuilding it.
    Used for incremental ingestion (Step 15).

    Args:
        vectorstore : existing Chroma instance
        documents   : new LangChain Documents to add

    Returns:
        Number of new chunks added
    """
    if not documents:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    new_chunks = splitter.split_documents(documents)
    vectorstore.add_documents(new_chunks)
    logger.info(f"Added {len(new_chunks)} new chunks to vectorstore")
    return len(new_chunks)
