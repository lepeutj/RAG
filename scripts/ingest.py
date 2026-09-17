"""Bulk-ingestion CLI script.

Usage:
    python scripts/ingest.py --path data/documents
    python scripts/ingest.py --path data/documents --reset   # clears the index first
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector index.")
    parser.add_argument("--path", type=Path, required=True, help="File or directory to ingest.")
    parser.add_argument("--reset", action="store_true", help="Clear the index before ingestion.")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = RAGPipeline(settings)

    if args.reset:
        logger.info("Resetting vector index...")
        pipeline._vector_store.reset()  # noqa: SLF001 (intentional admin-script usage)

    if args.path.is_dir():
        n_chunks = pipeline.ingest_directory(args.path)
    else:
        n_chunks = pipeline.ingest_file(args.path)

    logger.info("Complete: %d chunks indexed. Total in index: %d", n_chunks, pipeline.stats()["chunks_indexed"])


if __name__ == "__main__":
    main()
