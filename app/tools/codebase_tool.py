"""
codebase_tool.py
================
Scans a local source code repository and loads source files as LangChain Documents.

Supported extensions:
  .java        — Spring Boot / enterprise Java
  .py          — Python services
  .js  .ts     — JavaScript / TypeScript
  .xml         — Spring configs, Maven POM
  .yml .yaml   — Kubernetes, application configs
  .properties  — Spring Boot application.properties
  .sql         — Flyway / Liquibase migrations

For each file the tool:
  1. Reads raw source code (skips files > 100KB)
  2. Extracts class/method names as metadata (enables filtered retrieval)
  3. Adds a header comment block so the LLM knows which file it is reading
  4. Tags Document with source_type="codebase" for filtered MMR search

Config (from .env):
  CODEBASE_PATH — path to source root (e.g. C:/projects/my-app/src)
  Leave empty to skip codebase scanning.

Usage:
    from app.tools.codebase_tool import fetch_codebase_files
    docs = fetch_codebase_files()
"""

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {
    ".java", ".py", ".js", ".ts",
    ".xml", ".yml", ".yaml",
    ".properties", ".sql"
}

SKIP_FOLDERS = {
    "node_modules", ".git", "__pycache__", "target",
    "build", "dist", ".venv", "venv", ".idea",
    ".mvn", "generated-sources", "generated-test-sources"
}

MAX_FILE_SIZE_BYTES = 100_000   # 100 KB

LANG_MAP = {
    ".java": "java", ".py": "python",
    ".js": "javascript", ".ts": "typescript",
    ".xml": "xml", ".yml": "yaml", ".yaml": "yaml",
    ".properties": "properties", ".sql": "sql"
}


# ── Symbol extractors ─────────────────────────────────────────────────────────

def _extract_java_symbols(src: str) -> dict:
    classes  = re.findall(
        r"(?:public|private|protected)?\s*(?:abstract\s+)?(?:class|interface|enum|record)\s+(\w+)",
        src)
    methods  = re.findall(
        r"(?:public|protected|private)[^;{]*\s+(\w+)\s*\([^)]*\)\s*(?:throws[^{]*)?\s*\{",
        src)
    anns     = re.findall(r"@(\w+)", src)
    return {"classes": list(set(classes)), "methods": list(set(methods[:20])),
            "annotations": list(set(anns[:15]))}


def _extract_python_symbols(src: str) -> dict:
    classes  = re.findall(r"^class\s+(\w+)", src, re.MULTILINE)
    funcs    = re.findall(r"^def\s+(\w+)", src, re.MULTILINE)
    decs     = re.findall(r"^@(\w+)", src, re.MULTILINE)
    return {"classes": list(set(classes)), "methods": list(set(funcs[:20])),
            "annotations": list(set(decs[:10]))}


def _extract_js_symbols(src: str) -> dict:
    classes  = re.findall(r"(?:class|interface)\s+(\w+)", src)
    fns      = re.findall(
        r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
        src)
    fn_names = [f[0] or f[1] for f in fns if f[0] or f[1]]
    exports  = re.findall(r"export\s+(?:default\s+)?(?:class|function|const)\s+(\w+)", src)
    return {"classes": list(set(classes)), "methods": list(set(fn_names[:20])),
            "annotations": list(set(exports[:10]))}


def _extract_xml_symbols(src: str) -> dict:
    beans = re.findall(r'<bean[^>]+id="([^"]+)"', src)
    comps = re.findall(r'<component[^>]+class="([^"]+)"', src)
    return {"classes": list(set(comps[:10])), "methods": [],
            "annotations": list(set(beans[:10]))}


def _get_symbols(src: str, ext: str) -> dict:
    if ext == ".java":    return _extract_java_symbols(src)
    if ext == ".py":      return _extract_python_symbols(src)
    if ext in (".js", ".ts"): return _extract_js_symbols(src)
    if ext == ".xml":     return _extract_xml_symbols(src)
    return {"classes": [], "methods": [], "annotations": []}


# ── Main function ─────────────────────────────────────────────────────────────

def fetch_codebase_files() -> List[Document]:
    """
    Scan CODEBASE_PATH recursively and return source files as LangChain Documents.
    Returns empty list if CODEBASE_PATH is not set or does not exist.
    """
    codebase_path = getattr(config, "CODEBASE_PATH", "").strip()

    if not codebase_path:
        logger.info("CODEBASE_PATH not set. Skipping codebase ingestion.")
        print("[SKIP] CODEBASE_PATH not set in .env — skipping codebase scan")
        return []

    root = Path(codebase_path)
    if not root.exists() or not root.is_dir():
        logger.warning(f"Codebase path not found: {root}")
        print(f"[WARN] Codebase path not found: {root}")
        return []

    logger.info(f"Scanning codebase: {root}")
    print(f"[INFO] Scanning codebase: {root}")

    # Collect all matching files
    all_files = [
        f for f in root.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(skip in f.parts for skip in SKIP_FOLDERS)
        and f.stat().st_size <= MAX_FILE_SIZE_BYTES
    ]

    logger.info(f"Found {len(all_files)} source files (after exclusions)")
    print(f"[INFO] Found {len(all_files)} source files (excluding build/generated folders)")

    documents = []
    skipped   = 0

    for filepath in sorted(all_files):
        try:
            source = filepath.read_text(encoding="utf-8", errors="ignore")
            if not source.strip():
                skipped += 1
                continue

            ext      = filepath.suffix.lower()
            language = LANG_MAP.get(ext, "text")
            symbols  = _get_symbols(source, ext)

            try:
                rel_path = str(filepath.relative_to(root))
            except ValueError:
                rel_path = str(filepath)

            content = (
                f"// FILE: {rel_path}\n"
                f"// LANGUAGE: {language}\n"
                f"// CLASSES: {', '.join(symbols['classes']) or 'none'}\n"
                f"// METHODS: {', '.join(symbols['methods'][:10]) or 'none'}\n"
                f"\n{source}"
            )

            metadata = {
                "source_type":    "codebase",
                "source_system":  "local_repo",
                "path":           rel_path,
                "title":          filepath.name,
                "url":            None,
                "issue_key":      None,
                "page_id":        None,
                "author":         None,
                "labels":         language,
                "updated_at":     "",
                "content_type":   f"source_{language}",
                "language":       language,
                "classes":        ", ".join(symbols["classes"]),
                "methods":        ", ".join(symbols["methods"][:10]),
                "annotations":    ", ".join(symbols["annotations"][:10]),
                "file_extension": ext,
            }

            documents.append(Document(page_content=content, metadata=metadata))

        except Exception as e:
            logger.error(f"Failed to read: {filepath} — {e}")
            skipped += 1

    print(f"[OK] Indexed {len(documents)} source files ({skipped} skipped)")
    logger.info(f"Codebase documents prepared: {len(documents)}")
    return documents
