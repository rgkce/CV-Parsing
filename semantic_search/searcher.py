"""
searcher.py — Query encoding, per-section search, and weighted scoring
======================================================================

Implements the full query flow:
  1. Encode a job-description query with the ``"query: "`` prefix.
  2. Search each FAISS section index independently.
  3. Combine per-section similarities with configurable weights.
  4. Return a ranked list of candidates.

Empty sections are handled gracefully: their weight is redistributed
among the sections that *are* present for each candidate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .config import (
    DEFAULT_WEIGHTS,
    MATCH_THRESHOLD,
    QUERY_PREFIX,
    SECTIONS,
    TOP_K,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  QUERY ENCODING
# ─────────────────────────────────────────────


def encode_query(model, query_text: str) -> np.ndarray:
    """
    Encode a single query string into an L2-normalised embedding.

    The E5 ``"query: "`` prefix is prepended automatically.

    Parameters
    ----------
    model : SentenceTransformer
        The loaded embedding model.
    query_text : str
        Free-text job description or search query.

    Returns
    -------
    np.ndarray
        Shape ``(1, dim)``, float32, L2-normalised.
    """
    prefixed = QUERY_PREFIX + query_text.strip()
    emb = model.encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


# ─────────────────────────────────────────────
#  PER-SECTION SEARCH
# ─────────────────────────────────────────────


def search_section(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    k: int = TOP_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search a single FAISS index.

    Parameters
    ----------
    index : faiss.IndexFlatIP
        The section-level FAISS index.
    query_embedding : np.ndarray
        Shape ``(1, dim)``.
    k : int
        Number of results to retrieve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(scores, indices)`` — each of shape ``(1, k)``.
        Scores are cosine similarities (since vectors are L2-normalised
        and the index uses inner product).
    """
    # Clamp k to the number of vectors in the index
    k = min(k, index.ntotal)
    scores, indices = index.search(query_embedding, k)
    return scores, indices


def search_all_sections(
    indexes: Dict[str, faiss.IndexFlatIP],
    query_embedding: np.ndarray,
    k: int | None = None,
    sections: List[str] | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Search every section index and collect per-section results.

    Parameters
    ----------
    indexes : dict[str, faiss.IndexFlatIP]
        One index per section.
    query_embedding : np.ndarray
        Shape ``(1, dim)``.
    k : int, optional
        Results per section.  ``None`` → retrieve *all* vectors so the
        weighted combination has full coverage.
    sections : list[str], optional
        Sections to search.  Defaults to ``config.SECTIONS``.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        ``{section: (scores, indices)}``
    """
    sections = sections or SECTIONS
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for section in sections:
        if section not in indexes:
            continue
        index = indexes[section]
        # Retrieve ALL vectors so combine_scores sees every candidate
        effective_k = k if k is not None else index.ntotal
        scores, indices = search_section(index, query_embedding, effective_k)
        results[section] = (scores, indices)
        logger.debug(
            "Section %-12s  top-1 score=%.4f  idx=%d",
            section, scores[0][0], indices[0][0],
        )

    return results


# ─────────────────────────────────────────────
#  WEIGHTED SCORE COMBINATION
# ─────────────────────────────────────────────


def combine_scores(
    section_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    resume_ids: List[str],
    weights: Dict[str, float] | None = None,
    top_k: int = TOP_K,
    match_threshold: float = MATCH_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Combine per-section similarities into a single weighted score per
    candidate and return a ranked list.

    Algorithm
    ---------
    For each candidate *i*:

    1.  Collect cosine similarities from every section index.
    2.  Identify which sections are "present" (non-zero embedding).
        A zero-vector candidate in a section will have similarity ≈ 0.
    3.  Re-normalise weights to sum to 1.0 over the *present* sections
        so that CVs missing a section are not unfairly penalised.
    4.  Compute ``final_score = Σ (normalised_weight_s × sim_s)``.
    5.  Record which sections exceeded ``match_threshold``.

    Parameters
    ----------
    section_results : dict[str, (scores, indices)]
        Output of ``search_all_sections``.
    resume_ids : list[str]
        Ordered resume IDs matching index positions.
    weights : dict[str, float], optional
        Section weights.  Defaults to ``config.DEFAULT_WEIGHTS``.
    top_k : int
        How many candidates to return.
    match_threshold : float
        Minimum similarity for a section to appear in ``matched_sections``.

    Returns
    -------
    list[dict]
        Sorted descending by ``score``.  Each dict contains::

            {
                "resume_id": str,
                "score": float,
                "matched_sections": list[str],
                "section_scores": dict[str, float],
            }
    """
    weights = weights or DEFAULT_WEIGHTS
    n_candidates = len(resume_ids)

    # Build a per-candidate similarity matrix: {candidate_idx: {section: sim}}
    candidate_sims: Dict[int, Dict[str, float]] = {
        i: {} for i in range(n_candidates)
    }

    for section, (scores_arr, indices_arr) in section_results.items():
        scores = scores_arr[0]   # shape (k,)
        indices = indices_arr[0] # shape (k,)
        for score, idx in zip(scores, indices):
            idx = int(idx)
            if 0 <= idx < n_candidates:
                candidate_sims[idx][section] = float(score)

    # Score each candidate
    scored: List[Dict[str, Any]] = []

    for idx in range(n_candidates):
        sims = candidate_sims[idx]

        # Determine which sections this candidate actually has
        # (non-zero embedding → similarity will be meaningfully != 0)
        present_sections = {
            s for s in sims
            if s in weights and abs(sims[s]) > 1e-6
        }

        if not present_sections:
            # No meaningful content in any section
            scored.append({
                "resume_id": resume_ids[idx],
                "score": 0.0,
                "matched_sections": [],
                "section_scores": {},
            })
            continue

        # Re-normalise weights over present sections
        raw_weight_sum = sum(weights[s] for s in present_sections)
        if raw_weight_sum < 1e-9:
            raw_weight_sum = 1.0  # safety

        final_score = 0.0
        section_scores: Dict[str, float] = {}
        matched: List[str] = []

        for s in present_sections:
            sim = sims[s]
            normalised_w = weights[s] / raw_weight_sum
            final_score += normalised_w * sim
            section_scores[s] = round(sim, 4)
            if sim >= match_threshold:
                matched.append(s)

        scored.append({
            "resume_id": resume_ids[idx],
            "score": round(final_score, 4),
            "matched_sections": sorted(matched),
            "section_scores": section_scores,
        })

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_k]
