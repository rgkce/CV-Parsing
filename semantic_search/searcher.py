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
    MAX_SCORE_DROP,
    MIN_SCORE_THRESHOLD,
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
    min_score: float = MIN_SCORE_THRESHOLD,
    max_drop: float = MAX_SCORE_DROP,
    dataset: List[Dict[str, Any]] | None = None,
    query_text: str = "",
    bm25: Any = None,
    resume_ids_bm25: List[str] | None = None,
) -> List[Dict[str, Any]]:
    weights = weights or DEFAULT_WEIGHTS
    n_candidates = len(resume_ids)

    # Pre-compute lexical core keywords if dataset is provided
    core_keywords = set()
    if query_text and dataset:
        stopwords = {"mühendisi", "mühendisliği", "öğrencisi", "uzmanı", "geliştirici", "developer", "engineer", "student", "manager"}
        words = query_text.lower().split()
        core_keywords = {w for w in words if w not in stopwords and len(w) > 2}

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

    # Sort descending by dense score to establish dense rank
    scored.sort(key=lambda x: x["score"], reverse=True)

    # ── Filter: absolute minimum score ────────
    if min_score > 0:
        scored = [r for r in scored if r["score"] >= min_score]

    # ── Filter: relative drop from #1 ─────────
    if scored and max_drop > 0:
        best_score = scored[0]["score"]
        scored = [r for r in scored if (best_score - r["score"]) <= max_drop]

    # ── HYBRID SCORING (RRF) ─────────
    # If BM25 is provided, we re-rank the candidates that passed the dense filters.
    if bm25 and resume_ids_bm25 and query_text and scored:
        from semantic_search.bm25_indexer import tokenize_text
        from semantic_search.config import RRF_K
        
        tokenized_query = tokenize_text(query_text)
        
        # Get BM25 scores for ALL documents in the index
        all_bm25_scores = bm25.get_scores(tokenized_query)
        
        # Map resume_id to BM25 score
        bm25_score_map = {rid: score for rid, score in zip(resume_ids_bm25, all_bm25_scores)}
        
        # Assign dense rank
        for i, r in enumerate(scored):
            r["dense_rank"] = i + 1
            r["bm25_score"] = bm25_score_map.get(r["resume_id"], 0.0)
            
        # Sort by BM25 score descending to get sparse rank
        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        
        # Assign sparse rank and compute RRF
        for i, r in enumerate(scored):
            # If BM25 score is 0, assign a severely low rank to penalize it in RRF
            if r["bm25_score"] <= 0.0:
                sparse_rank = 1000
            else:
                sparse_rank = i + 1
                
            r["sparse_rank"] = sparse_rank
            
            # Calculate RRF score
            rrf_score = (1.0 / (RRF_K + r["dense_rank"])) + (1.0 / (RRF_K + sparse_rank))
            r["rrf_score"] = round(rrf_score, 5)
            
        # Re-sort finally by RRF score
        scored.sort(key=lambda x: x["rrf_score"], reverse=True)

    return scored[:top_k]

