"""
run_ranking.py — Milestone 4 Candidate Ranking Pipeline
=========================================================

End-to-end CLI for the Candidate Ranking and Decision Support System.

Usage::

    # English job description
    python -m candidate_ranker.run_ranking \\
        --jd "Python developer with machine learning experience"

    # Turkish job description
    python -m candidate_ranker.run_ranking \\
        --jd "AutoCAD ve SAP2000 bilen inşaat mühendisi"

    # Custom top-k and weights
    python -m candidate_ranker.run_ranking \\
        --jd "Data scientist with Python" \\
        --top-k 10 \\
        --weight-skills 0.50

    # JSON-only output (no text report to console)
    python -m candidate_ranker.run_ranking \\
        --jd "Frontend developer" \\
        --json

Pipeline:
    1. Load embedding model + FAISS indexes (from Milestone 3)
    2. Retrieve Top-K candidates via hybrid search
    3. Parse job description → structured fields
    4. Calculate section-level similarity scores
    5. Compute weighted final scores
    6. Generate LLM explanations (or template fallback)
    7. Produce recruiter report + JSON output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import hashlib
from typing import Dict, List

from .config import DEFAULT_SCORING_WEIGHTS, DEFAULT_TOP_K
from .jd_parser import parse_job_description
from .matcher import calculate_section_similarity
from .scorer import calculate_final_score
from .llm_explainer import generate_llm_explanation
from .report_generator import generate_candidate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _generate_job_id(jd_text: str) -> str:
    """Generate a short deterministic job ID from the JD text."""
    h = hashlib.md5(jd_text.encode("utf-8")).hexdigest()[:8]
    return f"JD-{h}"


def rank_candidates(
    jd_text: str,
    top_k: int = DEFAULT_TOP_K,
    weights: Dict[str, float] | None = None,
    json_only: bool = False,
    skip_llm: bool = False,
) -> Dict:
    """
    Full Milestone 4 pipeline: parse JD → retrieve → score → explain → report.

    Parameters
    ----------
    jd_text : str
        Raw job description text.
    top_k : int
        Number of candidates to rank.
    weights : dict, optional
        Custom scoring weights.
    json_only : bool
        If True, suppress text report to console.
    skip_llm : bool
        If True, skip LLM explanation (use template only).

    Returns
    -------
    dict
        Full structured output (same as saved JSON).
    """
    t_start = time.perf_counter()
    weights = weights or DEFAULT_SCORING_WEIGHTS
    job_id = _generate_job_id(jd_text)

    # ── Step 1: Load Milestone 3 resources ──
    logger.info("=" * 60)
    logger.info("STEP 1 / 7  —  Loading Milestone 3 resources")
    logger.info("=" * 60)

    from semantic_search.embeddings import load_model
    from semantic_search.indexer import load_indexes
    from semantic_search.embeddings import load_embeddings
    from semantic_search.utils import load_dataset
    from semantic_search.bm25_indexer import load_bm25_index

    model = load_model()
    indexes = load_indexes()
    _, resume_ids = load_embeddings()
    dataset = load_dataset()
    bm25, resume_ids_bm25 = load_bm25_index()

    # Build resume_id → candidate lookup
    candidate_lookup = {c["resume_id"]: c for c in dataset}

    # ── Step 2: Retrieve Top-K candidates ──
    logger.info("=" * 60)
    logger.info("STEP 2 / 7  —  Retrieving Top-K candidates")
    logger.info("=" * 60)

    from semantic_search.run_query import run_single_query
    from semantic_search.config import DEFAULT_WEIGHTS as M3_WEIGHTS

    retrieval_results = run_single_query(
        model, indexes, resume_ids, jd_text,
        weights=M3_WEIGHTS, top_k=top_k,
        dataset=dataset,
        bm25=bm25,
        resume_ids_bm25=resume_ids_bm25,
    )

    logger.info("  Retrieved %d candidates", len(retrieval_results))
    for r in retrieval_results:
        logger.info("    → %s (retrieval score: %.4f)", r["resume_id"], r["score"])

    if not retrieval_results:
        logger.warning("No candidates retrieved — check query or dataset")
        return {"job_id": job_id, "top_candidates": [], "error": "No candidates retrieved"}

    # ── Step 3: Parse job description ──
    logger.info("=" * 60)
    logger.info("STEP 3 / 7  —  Parsing job description")
    logger.info("=" * 60)

    parsed_jd = parse_job_description(jd_text)

    logger.info("  Required skills   : %s", parsed_jd["required_skills"])
    logger.info("  Preferred skills  : %s", parsed_jd["preferred_skills"])
    logger.info("  Experience reqs   : %s", parsed_jd["required_experience"])
    logger.info("  Education reqs    : %s", parsed_jd["education_requirements"])
    logger.info("  Soft skills       : %s", parsed_jd["soft_skills"])

    # ── Step 4: Calculate section-level scores ──
    logger.info("=" * 60)
    logger.info("STEP 4 / 7  —  Calculating section-level scores")
    logger.info("=" * 60)

    scored_candidates: List[Dict] = []

    for result in retrieval_results:
        rid = result["resume_id"]
        candidate = candidate_lookup.get(rid)

        if not candidate:
            logger.warning("Candidate %s not found in dataset — skipping", rid)
            continue

        section_scores = calculate_section_similarity(parsed_jd, candidate, model)

        scored_candidates.append({
            "candidate_id": rid,
            "candidate_data": candidate,
            "section_scores": section_scores,
            "retrieval_score": result["score"],
        })

    # ── Step 5: Calculate weighted final scores ──
    logger.info("=" * 60)
    logger.info("STEP 5 / 7  —  Calculating final scores")
    logger.info("=" * 60)

    for cand in scored_candidates:
        cand["final_score"] = calculate_final_score(
            cand["section_scores"], weights,
        )
        logger.info(
            "  %s → final_score: %.1f",
            cand["candidate_id"], cand["final_score"],
        )

    # Sort by final score descending
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # ── Step 6: Generate LLM explanations ──
    logger.info("=" * 60)
    logger.info("STEP 6 / 7  —  Generating explanations")
    logger.info("=" * 60)

    import os
    has_api_key = bool(os.environ.get("GOOGLE_API_KEY", ""))
    if skip_llm or not has_api_key:
        if not has_api_key:
            logger.info("  GOOGLE_API_KEY not set — using template-based explanations")
        else:
            logger.info("  --skip-llm flag set — using template-based explanations")

    for cand in scored_candidates:
        if skip_llm:
            # Force template fallback
            from .llm_explainer import _generate_template_explanation
            cand["llm_explanation"] = _generate_template_explanation(
                cand["candidate_data"], cand["section_scores"],
                cand["final_score"], parsed_jd,
            )
        else:
            cand["llm_explanation"] = generate_llm_explanation(
                jd_text, cand["candidate_data"],
                cand["section_scores"], cand["final_score"],
                parsed_jd=parsed_jd,
            )

    # ── Step 7: Generate report ──
    logger.info("=" * 60)
    logger.info("STEP 7 / 7  —  Generating report")
    logger.info("=" * 60)

    # Clean up candidate data before report (remove large raw data)
    report_candidates = []
    for cand in scored_candidates:
        report_candidates.append({
            "candidate_id": cand["candidate_id"],
            "final_score": cand["final_score"],
            "section_scores": cand["section_scores"],
            "llm_explanation": cand["llm_explanation"],
        })

    json_output = generate_candidate_report(
        job_id, jd_text, report_candidates, parsed_jd,
    )

    elapsed = time.perf_counter() - t_start

    logger.info("=" * 60)
    logger.info("✅  RANKING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Job ID           : %s", job_id)
    logger.info("  Candidates ranked: %d", len(report_candidates))
    logger.info("  Total time       : %.1f seconds", elapsed)
    logger.info("=" * 60)

    return json_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 4 — Candidate Ranking and Decision Support",
    )
    parser.add_argument(
        "--jd", "-j",
        type=str,
        required=True,
        help="Job description text (English or Turkish)",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--weight-skills", type=float, default=None)
    parser.add_argument("--weight-experience", type=float, default=None)
    parser.add_argument("--weight-education", type=float, default=None)
    parser.add_argument("--weight-soft-skill", type=float, default=None)
    parser.add_argument(
        "--json", dest="json_only", action="store_true",
        help="Output only JSON (suppress text report to console)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM calls — use template-based explanations only",
    )
    args = parser.parse_args()

    # Build weights
    weights = dict(DEFAULT_SCORING_WEIGHTS)
    overrides = {
        "skills": args.weight_skills,
        "experience": args.weight_experience,
        "education": args.weight_education,
        "soft_skill": args.weight_soft_skill,
    }
    any_override = False
    for key, val in overrides.items():
        if val is not None:
            weights[key] = val
            any_override = True

    if any_override:
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        logger.info("Custom weights (normalised): %s", weights)

    result = rank_candidates(
        jd_text=args.jd,
        top_k=args.top_k,
        weights=weights,
        json_only=args.json_only,
        skip_llm=args.skip_llm,
    )

    if args.json_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
