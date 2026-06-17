"""
embeddings.py — Embedding generation and persistence
=====================================================

Generates per-section embeddings using intfloat/multilingual-e5-base
(sentence-transformers).  Each CV section is embedded independently.

Empty sections are stored as zero vectors and flagged so they can be
excluded from downstream scoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .config import (
    EMBEDDING_DIM,
    EMBEDDINGS_DIR,
    ENCODE_BATCH_SIZE,
    MODEL_NAME,
    PASSAGE_PREFIX,
    SECTIONS,
)
from .utils import get_resume_ids, get_section_texts

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────


def load_model(model_name: str | None = None):
    """
    Load and return a SentenceTransformer model.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.
        Defaults to ``config.MODEL_NAME``.

    Returns
    -------
    SentenceTransformer
        The loaded model, ready for ``.encode()``.
    """
    from sentence_transformers import SentenceTransformer

    model_name = model_name or MODEL_NAME
    logger.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Model loaded — embedding dimension: %d", model.get_embedding_dimension())
    return model


# ─────────────────────────────────────────────
#  EMBEDDING GENERATION
# ─────────────────────────────────────────────


def generate_section_embeddings(
    model,
    texts: List[str],
    prefix: str = PASSAGE_PREFIX,
    batch_size: int = ENCODE_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into L2-normalised embeddings.

    Parameters
    ----------
    model : SentenceTransformer
        A loaded sentence-transformer model.
    texts : list[str]
        Raw section texts.  Empty strings are mapped to zero vectors.
    prefix : str
        Prefix prepended to each text (E5 models need ``"passage: "``
        for documents, ``"query: "`` for queries).
    batch_size : int
        Encoding batch size.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), EMBEDDING_DIM)``, float32, L2-normalised.
    """
    n = len(texts)
    dim = model.get_embedding_dimension()
    embeddings = np.zeros((n, dim), dtype=np.float32)

    # Separate non-empty texts for batch encoding
    non_empty_indices: List[int] = []
    non_empty_texts: List[str] = []

    for i, text in enumerate(texts):
        if text.strip():
            non_empty_indices.append(i)
            non_empty_texts.append(prefix + text)

    if not non_empty_texts:
        logger.warning("All %d texts are empty — returning zero matrix", n)
        return embeddings

    logger.info(
        "Encoding %d non-empty texts (%d empty / zero-vector)",
        len(non_empty_texts), n - len(non_empty_texts),
    )

    # Encode in batches with progress bar
    encoded = model.encode(
        non_empty_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalise for cosine similarity via IP
        convert_to_numpy=True,
    )

    # Place encoded vectors into the correct positions
    for idx, emb in zip(non_empty_indices, encoded):
        embeddings[idx] = emb

    return embeddings


def generate_all_embeddings(
    model,
    dataset: List[Dict[str, Any]],
    sections: List[str] | None = None,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for every section across the entire dataset.

    Parameters
    ----------
    model : SentenceTransformer
        A loaded sentence-transformer model.
    dataset : list[dict]
        The parsed CV dataset.
    sections : list[str], optional
        Which sections to embed.  Defaults to ``config.SECTIONS``.
    show_progress : bool
        Whether to show progress bars.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from section name to ``(N, dim)`` embedding array.
    """
    sections = sections or SECTIONS
    all_embeddings: Dict[str, np.ndarray] = {}

    for section in sections:
        logger.info("━" * 50)
        logger.info("Generating embeddings for section: %s", section.upper())
        texts = get_section_texts(dataset, section)
        emb = generate_section_embeddings(
            model, texts, show_progress=show_progress,
        )
        all_embeddings[section] = emb
        logger.info(
            "  → %s embeddings shape: %s", section, emb.shape,
        )

    return all_embeddings


# ─────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────


def save_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    resume_ids: List[str],
    output_dir: str | Path | None = None,
) -> None:
    """
    Save all section embeddings and the resume-ID list to disk.

    File layout inside *output_dir*::

        skills_embeddings.npy
        experience_embeddings.npy
        ...
        resume_ids.json
    """
    output_dir = Path(output_dir) if output_dir else EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for section, emb in embeddings_dict.items():
        fpath = output_dir / f"{section}_embeddings.npy"
        np.save(str(fpath), emb)
        logger.info("Saved %s  →  %s", section, fpath)

    ids_path = output_dir / "resume_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(resume_ids, f, ensure_ascii=False, indent=2)
    logger.info("Saved resume IDs  →  %s", ids_path)


def load_embeddings(
    input_dir: str | Path | None = None,
    sections: List[str] | None = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load previously saved embeddings from disk.

    Returns
    -------
    tuple[dict[str, np.ndarray], list[str]]
        (embeddings_dict, resume_ids)
    """
    input_dir = Path(input_dir) if input_dir else EMBEDDINGS_DIR
    sections = sections or SECTIONS

    if not input_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {input_dir}")

    embeddings_dict: Dict[str, np.ndarray] = {}
    for section in sections:
        fpath = input_dir / f"{section}_embeddings.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"Embedding file not found: {fpath}")
        embeddings_dict[section] = np.load(str(fpath))
        logger.info("Loaded %s embeddings: shape %s", section, embeddings_dict[section].shape)

    ids_path = input_dir / "resume_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(f"Resume IDs file not found: {ids_path}")
    with open(ids_path, "r", encoding="utf-8") as f:
        resume_ids = json.load(f)

    logger.info("Loaded %d resume IDs", len(resume_ids))
    return embeddings_dict, resume_ids
