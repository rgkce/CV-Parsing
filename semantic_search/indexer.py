"""
indexer.py — FAISS index construction and persistence
=====================================================

Builds one FAISS ``IndexFlatIP`` (inner-product / cosine) index per
CV section.  Because embeddings are L2-normalised, inner product
equals cosine similarity.

For the current dataset size (~66 CVs), exact brute-force search is
more than fast enough.  When the dataset grows past ~100 K vectors,
switch to ``IndexIVFFlat`` for approximate search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from .config import EMBEDDING_DIM, FAISS_DIR, SECTIONS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  INDEX BUILDING
# ─────────────────────────────────────────────


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS Inner-Product index from L2-normalised embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(N, dim)``, float32, **must be L2-normalised**.

    Returns
    -------
    faiss.IndexFlatIP
        Populated index ready for search.
    """
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    logger.info("Built FAISS IndexFlatIP: %d vectors × %d dimensions", n, dim)
    return index


def build_all_indexes(
    embeddings_dict: Dict[str, np.ndarray],
    sections: List[str] | None = None,
) -> Dict[str, faiss.IndexFlatIP]:
    """
    Build one FAISS index per section.

    Parameters
    ----------
    embeddings_dict : dict[str, np.ndarray]
        Mapping from section name to ``(N, dim)`` embedding array.
    sections : list[str], optional
        Which sections to index.  Defaults to ``config.SECTIONS``.

    Returns
    -------
    dict[str, faiss.IndexFlatIP]
        One index per section.
    """
    sections = sections or SECTIONS
    indexes: Dict[str, faiss.IndexFlatIP] = {}

    for section in sections:
        if section not in embeddings_dict:
            logger.warning("No embeddings for section '%s' — skipping", section)
            continue
        logger.info("Building FAISS index for section: %s", section.upper())
        indexes[section] = build_faiss_index(embeddings_dict[section])

    return indexes


# ─────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────


def save_indexes(
    indexes: Dict[str, faiss.IndexFlatIP],
    output_dir: str | Path | None = None,
) -> None:
    """
    Save all FAISS indexes to disk.

    File layout::

        skills_index.faiss
        experience_index.faiss
        ...
    """
    output_dir = Path(output_dir) if output_dir else FAISS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for section, index in indexes.items():
        fpath = output_dir / f"{section}_index.faiss"
        faiss.write_index(index, str(fpath))
        logger.info("Saved FAISS index  →  %s  (%d vectors)", fpath, index.ntotal)


def load_indexes(
    input_dir: str | Path | None = None,
    sections: List[str] | None = None,
) -> Dict[str, faiss.IndexFlatIP]:
    """
    Load previously saved FAISS indexes from disk.

    Returns
    -------
    dict[str, faiss.IndexFlatIP]
        One index per section.
    """
    input_dir = Path(input_dir) if input_dir else FAISS_DIR
    sections = sections or SECTIONS

    if not input_dir.exists():
        raise FileNotFoundError(f"FAISS directory not found: {input_dir}")

    indexes: Dict[str, faiss.IndexFlatIP] = {}

    for section in sections:
        fpath = input_dir / f"{section}_index.faiss"
        if not fpath.exists():
            raise FileNotFoundError(f"Index file not found: {fpath}")
        indexes[section] = faiss.read_index(str(fpath))
        logger.info(
            "Loaded FAISS index: %s  (%d vectors)",
            section, indexes[section].ntotal,
        )

    return indexes
