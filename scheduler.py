"""
scheduler.py
============
APScheduler-based background job that runs ingestion automatically
on a configurable interval (default: every 6 hours).

This keeps the ChromaDB knowledge base fresh without manual steps:
  - New Jira issues are indexed within hours of creation
  - Updated Confluence pages are picked up automatically
  - New KB articles and code changes are reflected on next run

Run:
    python scheduler.py               # run scheduler in foreground
    python scheduler.py --run-now     # run one ingestion immediately, then schedule
    python scheduler.py --interval 2  # override interval to every 2 hours

Config (from .env):
  SCHEDULER_INTERVAL_HOURS — hours between ingestion runs (default: 6)
  LOG_LEVEL=DEBUG           — show detailed scheduler logs

The scheduler logs to both stdout and ingestion_log.txt.
Press Ctrl+C to stop.
"""

import argparse
import signal
import sys
import time
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
from app.rag.loader import load_all_documents
from app.rag.vectorstore import load_vectorstore, build_vectorstore, add_documents
from app.utils.logger import get_logger, add_file_handler

logger = get_logger(__name__)

# Also write scheduler log to file
add_file_handler("ingestion_log.txt")


# ── Ingestion job ─────────────────────────────────────────────────────────────

def run_ingestion_job():
    """
    Core ingestion job — called by APScheduler on each interval.

    Strategy:
      - Try incremental update: load vectorstore + add only new documents
      - Fall back to full rebuild if vectorstore does not exist yet
    """
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Scheduled ingestion started at {run_ts}")
    print(f"\n{'='*60}")
    print(f"  SCHEDULED INGESTION  —  {run_ts}")
    print(f"{'='*60}")

    try:
        documents = load_all_documents()

        if not documents:
            logger.warning("No documents found. Skipping vectorstore update.")
            print("[WARN] No documents returned. Skipping update.")
            return

        try:
            # Try incremental add to existing vectorstore
            vectorstore   = load_vectorstore()
            chunks_added  = add_documents(vectorstore, documents)
            logger.info(f"Incremental update: {chunks_added} new chunks added")
            print(f"[OK] Incremental update: {chunks_added} new chunks added")

        except RuntimeError:
            # Vectorstore does not exist — do full build
            logger.info("Vectorstore not found — running full build")
            print("[INFO] Vectorstore not found — running full build")
            build_vectorstore(documents)

        logger.info(f"Scheduled ingestion complete — {len(documents)} docs processed")
        print(f"[OK] Ingestion complete — {len(documents)} documents processed")

    except Exception as e:
        logger.error(f"Scheduled ingestion failed: {e}")
        print(f"[ERROR] Ingestion failed: {e}")


# ── Signal handler (graceful shutdown) ────────────────────────────────────────

def _handle_signal(sig, frame):
    print("\n[INFO] Scheduler shutting down (Ctrl+C)...")
    logger.info("Scheduler interrupted by user")
    sys.exit(0)

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG Agent — Background Scheduler")
    parser.add_argument("--run-now",  action="store_true",
                        help="Run one ingestion immediately, then start the schedule")
    parser.add_argument("--interval", type=float, default=None,
                        help="Override SCHEDULER_INTERVAL_HOURS from .env")
    args = parser.parse_args()

    # Resolve interval
    interval_hours = args.interval or getattr(config, "SCHEDULER_INTERVAL_HOURS", 6.0)
    interval_hours = float(interval_hours)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  LangGraph RAG Agent — Background Scheduler          ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Interval  : every {interval_hours:.1f} hour(s)")
    print(f"  Log file  : ingestion_log.txt")
    print(f"  Config    : CHROMA_PERSIST_DIR = {config.CHROMA_PERSIST_DIR}")
    print("  Press Ctrl+C to stop.\n")

    logger.info(f"Scheduler starting — interval={interval_hours}h")

    # Optional immediate run
    if args.run_now:
        print("[INFO] Running immediate ingestion before scheduling...")
        run_ingestion_job()

    # Set up APScheduler
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        func=run_ingestion_job,
        trigger=IntervalTrigger(hours=interval_hours),
        id="ingestion_job",
        name="RAG Ingestion",
        replace_existing=True,
        misfire_grace_time=300,   # allow up to 5-minute delay
    )

    next_run = datetime.now()
    print(f"[INFO] Next run scheduled in {interval_hours:.1f} hour(s)")
    logger.info("Scheduler started")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
        print("\n[INFO] Scheduler stopped.")


if __name__ == "__main__":
    main()
