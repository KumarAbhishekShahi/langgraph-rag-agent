"""
file_tool.py
============
Reads local files (PDF, HTML, TXT, CSV, MD) from the sample_data/ folder.

Config (from .env):
  SAMPLE_DATA_PATH — path to sample_data folder (default: ./sample_data)

Supported extensions:
  .pdf   — extracted via PyPDF
  .html  — HTML stripped to plain text via BeautifulSoup
  .htm   — same as .html
  .txt   — plain text
  .md    — Markdown as plain text
  .csv   — tabular data formatted as readable rows

Each Document contains:
  page_content : file content
  metadata     : source_type="local_file", title, path, content_type
"""

import csv
import io
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from langchain_core.documents import Document

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FILE_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md", ".csv"}


def fetch_local_files() -> List[Document]:
    """
    Read all supported files from SAMPLE_DATA_PATH.
    Returns empty list if path is not configured or does not exist.
    """
    data_path = Path(config.SAMPLE_DATA_PATH)

    if not data_path.exists():
        logger.warning(f"Sample data path not found: {data_path}. Skipping.")
        print(f"[SKIP] Sample data folder not found: {data_path}")
        return []

    files = [
        f for f in data_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_FILE_EXTENSIONS
    ]

    logger.info(f"Found {len(files)} local files in {data_path}")
    print(f"[INFO] Found {len(files)} local files in: {data_path}")

    documents = []
    for filepath in sorted(files):
        docs = _load_file(filepath, data_path)
        documents.extend(docs)

    logger.info(f"Loaded {len(documents)} Documents from local files")
    return documents


def _load_file(filepath: Path, root: Path) -> List[Document]:
    """Dispatch to the correct loader based on file extension."""
    ext = filepath.suffix.lower()
    try:
        if ext == ".pdf":
            return _load_pdf(filepath, root)
        elif ext in (".html", ".htm"):
            return _load_html(filepath, root)
        elif ext == ".csv":
            return _load_csv(filepath, root)
        else:  # .txt, .md
            return _load_text(filepath, root)
    except Exception as e:
        logger.error(f"Failed to load file {filepath}: {e}")
        return []


def _make_metadata(filepath: Path, root: Path, content_type: str) -> dict:
    rel_path = str(filepath.relative_to(root))
    title    = filepath.stem.replace("-", " ").replace("_", " ").title()
    return {
        "source_type":   "local_file",
        "source_system": "local_files",
        "title":         title,
        "path":          rel_path,
        "url":           None,
        "issue_key":     None,
        "page_id":       None,
        "author":        None,
        "labels":        "",
        "updated_at":    "",
        "content_type":  content_type,
    }


def _load_pdf(filepath: Path, root: Path) -> List[Document]:
    from pypdf import PdfReader
    reader   = PdfReader(str(filepath))
    metadata = _make_metadata(filepath, root, "pdf")
    docs     = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            meta = {**metadata, "page_number": i + 1}
            docs.append(Document(
                page_content=f"PDF: {filepath.name} | Page {i+1}\n\n{text[:2000]}",
                metadata=meta
            ))
    return docs


def _load_html(filepath: Path, root: Path) -> List[Document]:
    raw      = filepath.read_text(encoding="utf-8", errors="ignore")
    soup     = BeautifulSoup(raw, "lxml")
    text     = soup.get_text(separator="\n", strip=True)
    if not text.strip():
        return []
    metadata = _make_metadata(filepath, root, "html")
    return [Document(
        page_content=f"HTML File: {filepath.name}\n\n{text[:3000]}",
        metadata=metadata
    )]


def _load_text(filepath: Path, root: Path) -> List[Document]:
    text     = filepath.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    ext      = filepath.suffix.lower().strip(".")
    metadata = _make_metadata(filepath, root, ext)
    return [Document(
        page_content=f"File: {filepath.name}\n\n{text[:3000]}",
        metadata=metadata
    )]


def _load_csv(filepath: Path, root: Path) -> List[Document]:
    """Load CSV — each row becomes a readable sentence."""
    raw      = filepath.read_text(encoding="utf-8", errors="ignore")
    reader   = csv.DictReader(io.StringIO(raw))
    rows     = list(reader)
    if not rows:
        return []

    lines = [f"CSV file: {filepath.name}"]
    lines.append(f"Columns: {', '.join(rows[0].keys())}\n")
    for i, row in enumerate(rows[:100]):   # cap at 100 rows
        row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
        lines.append(f"Row {i+1}: {row_text}")

    metadata = _make_metadata(filepath, root, "csv")
    return [Document(page_content="\n".join(lines), metadata=metadata)]
