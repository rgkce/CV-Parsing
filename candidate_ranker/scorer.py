"""
scorer.py — Weighted Final Score Calculation
=============================================

Applies configurable weights to section-level scores to produce
a single final candidate score on a 0-100 scale.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .config import DEFAULT_SCORING_WEIGHTS

logger = logging.getLogger(__name__)


def calculate_final_score(
    section_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a weighted final score from section-level scores.

    Parameters
    ----------
    section_scores : dict
        Must contain keys: ``skills_score``, ``experience_score``,
        ``education_score``, ``soft_skill_score``.
        Each value is on a 0-100 scale.
    weights : dict, optional
        Custom weights with keys: ``skills``, ``experience``,
        ``education``, ``soft_skill``.
        Must sum to 1.0.  Defaults to ``DEFAULT_SCORING_WEIGHTS``.

    Returns
    -------
    float
        Final score on 0-100 scale, rounded to 1 decimal place.

    Raises
    ------
    ValueError
        If required score keys are missing.
    """
    weights = weights or DEFAULT_SCORING_WEIGHTS

    # Validate required keys
    required_keys = {"skills_score", "experience_score", "education_score", "soft_skill_score"}
    missing = required_keys - set(section_scores.keys())
    if missing:
        raise ValueError(f"Missing section scores: {missing}")

    # Validate weights sum
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(
            "Weights sum to %.3f (expected 1.0) — normalising", weight_sum,
        )
        weights = {k: v / weight_sum for k, v in weights.items()}

    # Calculate weighted score
    final_score = (
        section_scores["skills_score"]     * weights["skills"]
        + section_scores["experience_score"] * weights["experience"]
        + section_scores["education_score"]  * weights["education"]
        + section_scores["soft_skill_score"] * weights["soft_skill"]
    )

    final_score = round(max(0.0, min(100.0, final_score)), 1)

    logger.debug(
        "Final score: %.1f (skills=%.1f×%.2f + exp=%.1f×%.2f + "
        "edu=%.1f×%.2f + soft=%.1f×%.2f)",
        final_score,
        section_scores["skills_score"], weights["skills"],
        section_scores["experience_score"], weights["experience"],
        section_scores["education_score"], weights["education"],
        section_scores["soft_skill_score"], weights["soft_skill"],
    )

    return final_score
