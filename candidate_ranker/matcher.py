"""
matcher.py — Section-Level Candidate Matching
===============================================

Computes per-section similarity scores between a parsed job description
and each retrieved candidate CV using embedding-based cosine similarity.

Output per candidate::

    {
        "skills_score":     92.3,
        "experience_score": 85.1,
        "education_score":  78.0,
        "soft_skill_score": 80.5
    }
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import numpy as np

from .config import EDUCATION_LEVELS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  EMBEDDING UTILITIES
# ─────────────────────────────────────────────


def _encode_texts(model, texts: List[str], prefix: str = "query: ") -> np.ndarray:
    """
    Encode a list of texts into L2-normalised embeddings.

    Uses the E5 query prefix for JD terms (they act as queries
    searching against CV passages).
    """
    if not texts:
        return np.array([])

    prefixed = [prefix + t.strip() for t in texts if t.strip()]
    if not prefixed:
        return np.array([])

    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def _encode_single(model, text: str, prefix: str = "passage: ") -> np.ndarray:
    """Encode a single text into an L2-normalised embedding."""
    if not text or not text.strip():
        return np.zeros((1, model.get_embedding_dimension()), dtype=np.float32)

    prefixed = prefix + text.strip()
    emb = model.encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalised vectors.

    Since vectors are normalised, this is just the dot product.
    """
    if a.size == 0 or b.size == 0:
        return 0.0

    # Flatten to 1D if needed
    a = a.flatten()
    b = b.flatten()

    if a.shape != b.shape:
        return 0.0

    return float(np.dot(a, b))


def _average_best_similarity(
    query_embeddings: np.ndarray,
    passage_embedding: np.ndarray,
) -> float:
    """
    For each query embedding, compute similarity against the passage,
    then average.  This measures "how well does the passage cover
    all the query terms on average".

    Parameters
    ----------
    query_embeddings : np.ndarray
        Shape ``(N, dim)`` — one embedding per JD requirement term.
    passage_embedding : np.ndarray
        Shape ``(1, dim)`` — the CV section embedding.

    Returns
    -------
    float
        Average cosine similarity, scaled to 0-100.
    """
    if query_embeddings.size == 0 or passage_embedding.size == 0:
        return 0.0

    passage_flat = passage_embedding.flatten()

    similarities = []
    for i in range(query_embeddings.shape[0]):
        sim = float(np.dot(query_embeddings[i], passage_flat))
        similarities.append(sim)

    if not similarities:
        return 0.0

    # Average similarity, scaled to 0-100
    # Cosine similarity with E5 models typically ranges 0.3-0.95
    # We map [0.3, 1.0] → [0, 100] for more useful scores
    avg_sim = sum(similarities) / len(similarities)
    score = max(0.0, (avg_sim - 0.3) / 0.7) * 100.0
    return round(min(score, 100.0), 1)


# ─────────────────────────────────────────────
#  SECTION SCORING FUNCTIONS
# ─────────────────────────────────────────────


def _score_skills(
    model,
    parsed_jd: Dict[str, Any],
    candidate: Dict[str, Any],
) -> float:
    """
    Score how well the candidate's skills match the JD requirements.

    Embeds each JD skill term → computes similarity against
    the candidate's skills section embedding.
    """
    jd_skills = parsed_jd.get("required_skills", []) + parsed_jd.get("preferred_skills", [])

    if not jd_skills:
        # No specific skills in JD → use raw JD text as query
        jd_skills = [parsed_jd.get("raw_text", "")]

    cv_skills_text = candidate.get("sections", {}).get("skills", "")
    if not cv_skills_text or not cv_skills_text.strip():
        return 0.0

    query_embs = _encode_texts(model, jd_skills, prefix="query: ")
    passage_emb = _encode_single(model, cv_skills_text, prefix="passage: ")

    return _average_best_similarity(query_embs, passage_emb)


def _score_experience(
    model,
    parsed_jd: Dict[str, Any],
    candidate: Dict[str, Any],
) -> float:
    """
    Score how well the candidate's experience matches the JD.

    Combines experience + projects sections for a richer signal.
    """
    jd_experience = parsed_jd.get("required_experience", [])

    # Build JD experience query text
    jd_texts = []
    if jd_experience:
        jd_texts.extend([str(e) for e in jd_experience])

    # Also use skills as experience context (e.g. "5 years Python" → Python experience)
    jd_skills = parsed_jd.get("required_skills", [])
    if jd_skills:
        jd_texts.append(" ".join(jd_skills))

    if not jd_texts:
        jd_texts = [parsed_jd.get("raw_text", "")]

    # Combine candidate experience and projects
    sections = candidate.get("sections", {})
    exp_text = sections.get("experience", "")
    proj_text = sections.get("projects", "")
    cv_text = f"{exp_text} {proj_text}".strip()

    if not cv_text:
        return 0.0

    query_embs = _encode_texts(model, jd_texts, prefix="query: ")
    passage_emb = _encode_single(model, cv_text, prefix="passage: ")

    return _average_best_similarity(query_embs, passage_emb)


def _score_education(
    model,
    parsed_jd: Dict[str, Any],
    candidate: Dict[str, Any],
) -> float:
    """
    Score education match using keyword overlap for degree level
    plus embedding similarity for field of study.

    Degree-level matching (keyword):
    - Exact level match → 100%
    - Higher level than required → 100%
    - Lower level → 40%

    Field matching (embedding): averaged with degree score.
    """
    edu_reqs = parsed_jd.get("education_requirements", [])
    cv_edu_text = candidate.get("sections", {}).get("education", "")

    if not cv_edu_text or not cv_edu_text.strip():
        return 0.0

    if not edu_reqs:
        # No specific education requirement → moderate default
        return 60.0

    # ── Degree-level matching ──
    level_hierarchy = {"associate": 1, "bachelor": 2, "master": 3, "phd": 4}
    cv_edu_lower = cv_edu_text.lower()

    # Detect candidate's education level
    candidate_level = 0
    for term, level_name in EDUCATION_LEVELS.items():
        if term in cv_edu_lower:
            level_val = level_hierarchy.get(level_name, 0)
            candidate_level = max(candidate_level, level_val)

    # If we couldn't detect level, assume bachelor (university mention)
    if candidate_level == 0 and ("üniversite" in cv_edu_lower or "university" in cv_edu_lower):
        candidate_level = 2  # bachelor

    # Get required level
    required_level = 0
    for req in edu_reqs:
        level_name = req.get("level", "")
        level_val = level_hierarchy.get(level_name, 0)
        required_level = max(required_level, level_val)

    # Score the level match
    if required_level == 0:
        level_score = 70.0  # no specific level → moderate
    elif candidate_level >= required_level:
        level_score = 100.0
    elif candidate_level > 0:
        level_score = 40.0  # has education but lower level
    else:
        level_score = 20.0  # no detected education level

    # ── Field-of-study matching (embedding) ──
    field_texts = [req.get("field", "") for req in edu_reqs if req.get("field")]
    if field_texts:
        query_embs = _encode_texts(model, field_texts, prefix="query: ")
        passage_emb = _encode_single(model, cv_edu_text, prefix="passage: ")
        field_score = _average_best_similarity(query_embs, passage_emb)
    else:
        field_score = level_score  # no field requirement → use level score

    # Combine: 50% level + 50% field
    final_score = (level_score * 0.5) + (field_score * 0.5)
    return round(final_score, 1)


def _score_soft_skills(
    model,
    parsed_jd: Dict[str, Any],
    candidate: Dict[str, Any],
) -> float:
    """
    Score soft skills by embedding JD soft-skill terms and comparing
    against candidate summary + experience sections.
    """
    jd_soft = parsed_jd.get("soft_skills", [])
    if not jd_soft:
        return 60.0  # no soft skills in JD → neutral

    sections = candidate.get("sections", {})
    summary = sections.get("summary", "")
    experience = sections.get("experience", "")
    cv_text = f"{summary} {experience}".strip()

    if not cv_text:
        return 0.0

    query_embs = _encode_texts(model, jd_soft, prefix="query: ")
    passage_emb = _encode_single(model, cv_text, prefix="passage: ")

    return _average_best_similarity(query_embs, passage_emb)


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────


def calculate_section_similarity(
    parsed_jd: Dict[str, Any],
    candidate: Dict[str, Any],
    model,
) -> Dict[str, float]:
    """
    Calculate section-level matching scores between a parsed JD
    and a single candidate CV.

    Parameters
    ----------
    parsed_jd : dict
        Output from ``parse_job_description()``.
    candidate : dict
        A single CV record from the dataset.
    model : SentenceTransformer
        The loaded embedding model.

    Returns
    -------
    dict
        Section scores (0-100 scale):
        - ``skills_score``
        - ``experience_score``
        - ``education_score``
        - ``soft_skill_score``
    """
    candidate_id = candidate.get("resume_id", "unknown")
    logger.debug("Scoring candidate: %s", candidate_id)

    scores = {
        "skills_score": _score_skills(model, parsed_jd, candidate),
        "experience_score": _score_experience(model, parsed_jd, candidate),
        "education_score": _score_education(model, parsed_jd, candidate),
        "soft_skill_score": _score_soft_skills(model, parsed_jd, candidate),
    }

    logger.debug(
        "  Candidate %s scores: skills=%.1f exp=%.1f edu=%.1f soft=%.1f",
        candidate_id, scores["skills_score"], scores["experience_score"],
        scores["education_score"], scores["soft_skill_score"],
    )

    return scores
