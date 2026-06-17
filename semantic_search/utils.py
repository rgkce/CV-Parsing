"""
utils.py — Dataset loading and helper utilities
================================================

Provides functions to load the parsed CV dataset from JSON and
extract section texts / resume IDs in a consistent order.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .config import DATASET_PATH, SECTIONS

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """
    Load the CV dataset from a JSON file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the JSON dataset file.
        Defaults to ``config.DATASET_PATH``.

    Returns
    -------
    list[dict]
        List of CV records.  Each record has at least
        ``resume_id`` and ``sections`` keys.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If any record is missing required fields.
    """
    path = Path(path) if path else DATASET_PATH

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # ── validate ──────────────────────────────
    valid_records: List[Dict[str, Any]] = []
    skipped = 0

    for i, record in enumerate(dataset):
        if "resume_id" not in record:
            logger.warning("Record %d missing 'resume_id' — skipping", i)
            skipped += 1
            continue
        if "sections" not in record:
            logger.warning(
                "Record %d (id=%s) missing 'sections' — skipping",
                i, record.get("resume_id", "?"),
            )
            skipped += 1
            continue
        valid_records.append(record)

    logger.info(
        "Loaded %d valid CVs (skipped %d invalid records)",
        len(valid_records), skipped,
    )
    return valid_records


def get_resume_ids(dataset: List[Dict[str, Any]]) -> List[str]:
    """
    Return an ordered list of resume_id values from the dataset.

    The order matches the embedding / FAISS index order.
    """
    return [record["resume_id"] for record in dataset]


def get_section_texts(
    dataset: List[Dict[str, Any]],
    section: str,
) -> List[str]:
    """
    Extract the text for a specific section from every CV in the dataset.

    Parameters
    ----------
    dataset : list[dict]
        The loaded CV dataset.
    section : str
        One of the section names defined in ``config.SECTIONS``.

    Returns
    -------
    list[str]
        One string per CV.  Empty string if the section is missing
        or blank for a given CV.
    """
    if section not in SECTIONS:
        raise ValueError(
            f"Unknown section '{section}'. Valid sections: {SECTIONS}"
        )

    texts: List[str] = []
    for record in dataset:
        text = record.get("sections", {}).get(section, "")
        # Normalise: strip and collapse None → ""
        texts.append((text or "").strip())
    return texts
