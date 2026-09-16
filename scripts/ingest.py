"""
Script CLI d'ingestion en masse.

Usage:
    python scripts/ingest.py --path data/documents
    python scripts/ingest.py --path data/documents --reset   # vide l'index avant
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
    parser = argparse.ArgumentParser(description="Ingère des documents dans l'index vectoriel.")
    parser.add_argument("--path", type=Path, required=True, help="Fichier ou dossier à ingérer.")
    parser.add_argument("--reset", action="store_true", help="Vide l'index avant d'ingérer.")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = RAGPipeline(settings)

    if args.reset:
        logger.info("Réinitialisation de l'index vectoriel...")
        pipeline._vector_store.reset()  # noqa: SLF001 (usage volontaire depuis un script d'admin)

    if args.path.is_dir():
        n_chunks = pipeline.ingest_directory(args.path)
    else:
        n_chunks = pipeline.ingest_file(args.path)

    logger.info("Terminé: %d chunks indexés. Total dans l'index: %d", n_chunks, pipeline.stats()["chunks_indexed"])


if __name__ == "__main__":
    main()
