"""
logger.py
=========
Centralised logging setup for the entire project.

Usage in any file:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.warning("Something unexpected")
    logger.error("Something failed")

Log format:
    2026-03-31 22:15:03  INFO  app.agents.gherkin_agent  Generating Gherkin...

Log level is controlled by the LOG_LEVEL env var (default: INFO).
Set LOG_LEVEL=DEBUG in .env to see detailed step-by-step logs.

Helper functions also provided:
    format_history()   — format conversation history for prompt injection
    strip_think_block() — remove DeepSeek-R1 <think>...</think> from output
"""

import logging
import os
import re
import sys
from typing import List


# ── Log level from environment ────────────────────────────────────────────────
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LEVEL_MAP  = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_NUMERIC_LEVEL = _LEVEL_MAP.get(_LOG_LEVEL, logging.INFO)


# ── Formatter ─────────────────────────────────────────────────────────────────
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ── Root handler (stdout) ─────────────────────────────────────────────────────
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_FORMATTER)
_handler.setLevel(_NUMERIC_LEVEL)

# Configure root logger once
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    _root_logger.addHandler(_handler)
_root_logger.setLevel(_NUMERIC_LEVEL)

# Silence noisy third-party loggers
for _noisy in ["httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Done")
    """
    logger = logging.getLogger(name)
    logger.setLevel(_NUMERIC_LEVEL)
    return logger


def format_history(history: list, max_turns: int = 4) -> str:
    """
    Format the last N conversation turns for injection into a prompt.

    Each turn is formatted as:
        USER: <text>
        ASSISTANT: <text>

    Args:
        history   : list of ConversationTurn dicts [{role, content}, ...]
        max_turns : max number of turns to include (default 4)

    Returns:
        Formatted string, or "No prior conversation." if history is empty.
    """
    if not history:
        return "No prior conversation."

    recent = history[-max_turns:]
    lines  = []
    for turn in recent:
        role    = turn.get("role", "user").upper()
        content = turn.get("content", "")[:500]   # truncate long turns
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def strip_think_block(text: str) -> str:
    """
    Remove DeepSeek-R1 <think>...</think> reasoning blocks from output.

    DeepSeek-R1 streams its internal reasoning inside <think> tags
    before producing the final answer. This helper strips those blocks
    so only the clean answer is stored or displayed.

    Usage:
        answer = strip_think_block(raw_response)

    Note: Strip AFTER accumulating the full streamed response —
          you cannot strip mid-stream.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def add_file_handler(log_file: str = "ingestion_log.txt") -> None:
    """
    Attach a file handler to the root logger.
    Useful for the scheduler to persist ingestion history to a file.

    Usage (in scheduler.py):
        from app.utils.logger import add_file_handler
        add_file_handler("ingestion_log.txt")
    """
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(_FORMATTER)
    file_handler.setLevel(_NUMERIC_LEVEL)
    logging.getLogger().addHandler(file_handler)
