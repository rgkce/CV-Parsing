"""
run_pipeline.py — End-to-end embedding + indexing pipeline
==========================================================

Usage::

    python -m semantic_search.run_pipeline
    python -m semantic_search.run_pipeline --dataset path/to/dataset.json

Steps:
    1. Load CV dataset
    2. Load sentence-transformer model
    3. Generate per-section embeddings
    4. Save embeddings to disk
    5. Build FAISS indexes
    6. Save FAISS indexes to disk
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import DATASET_PATH, EMBEDDINGS_DIR, FAISS_DIR, SECTIONS
from .embeddings import (
    generate_all_embeddings,
    load_model,
    save_embeddings,
)
from .indexer import build_all_indexes, save_indexes
from .utils import get_resume_ids, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 3 — Embedding & FAISS indexing pipeline",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Path to the CV dataset JSON file",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=str(EMBEDDINGS_DIR),
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--faiss-dir",
        type=str,
        default=str(FAISS_DIR),
        help="Directory to save FAISS indexes",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # ── 1. Load dataset ──────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 / 6  —  Loading dataset")
    logger.info("=" * 60)
    dataset = load_dataset(args.dataset)
    resume_ids = get_resume_ids(dataset)
    logger.info("  CVs loaded: %d", len(dataset))

    # ── 2. Load model ────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 / 6  —  Loading embedding model")
    logger.info("=" * 60)
    model = load_model()

    # ── 3. Generate embeddings ───────────────
    logger.info("=" * 60)
    logger.info("STEP 3 / 6  —  Generating per-section embeddings")
    logger.info("=" * 60)
    embeddings_dict = generate_all_embeddings(model, dataset)

    # ── 4. Save embeddings ───────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 / 6  —  Saving embeddings to disk")
    logger.info("=" * 60)
    save_embeddings(embeddings_dict, resume_ids, args.embeddings_dir)

    # ── 5. Build FAISS indexes ───────────────
    logger.info("=" * 60)
    logger.info("STEP 5 / 6  —  Building FAISS indexes")
    logger.info("=" * 60)
    indexes = build_all_indexes(embeddings_dict)

    # ── 6. Save FAISS indexes ────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 / 6  —  Saving FAISS indexes to disk")
    logger.info("=" * 60)
    save_indexes(indexes, args.faiss_dir)

    elapsed = time.perf_counter() - t_start

    # ── Summary ──────────────────────────────
    logger.info("=" * 60)
    logger.info("✅  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("  CVs processed     : %d", len(dataset))
    logger.info("  Sections embedded  : %s", SECTIONS)
    logger.info("  Embedding dim      : %d", embeddings_dict[SECTIONS[0]].shape[1])
    logger.info("  Embeddings saved   : %s", args.embeddings_dir)
    logger.info("  FAISS indexes saved: %s", args.faiss_dir)
    logger.info("  Total time         : %.1f seconds", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
