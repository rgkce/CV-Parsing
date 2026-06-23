"""
report_generator.py — Recruiter Report and JSON Output
========================================================

Produces:
1. Structured JSON output (full system response)
2. Human-readable text report with ranking tables

All outputs are saved to ``ranking_outputs/`` directory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  TEXT REPORT GENERATION
# ─────────────────────────────────────────────


def _generate_text_report(
    job_id: str,
    jd_text: str,
    ranked_candidates: List[Dict[str, Any]],
) -> str:
    """
    Generate a recruiter-friendly text report with ranking table
    and per-candidate explanations.
    """
    lines: List[str] = []

    # Header
    lines.append("=" * 80)
    lines.append("  CANDIDATE RANKING REPORT")
    lines.append("=" * 80)
    lines.append(f"  Job ID     : {job_id}")
    lines.append(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Candidates : {len(ranked_candidates)}")
    lines.append("")

    # Truncate JD for display
    jd_display = jd_text[:200] + "..." if len(jd_text) > 200 else jd_text
    lines.append(f"  Job Description:")
    lines.append(f"  {jd_display}")
    lines.append("")

    # Ranking table
    lines.append("-" * 105)
    header = (
        f"  {'Rank':<6}{'Candidate Name':<25}{'Candidate ID':<38}"
        f"{'Skills':<9}{'Exp':<9}{'Edu':<9}{'Soft':<9}{'Total':<8}"
    )
    lines.append(header)
    lines.append("  " + "-" * 101)

    for i, cand in enumerate(ranked_candidates, 1):
        scores = cand.get("section_scores", {})
        rid = cand.get("candidate_id", "?")
        name = cand.get("candidate_name", "Unknown")
        # Truncate long IDs for table display
        rid_display = rid[:34] + ".." if len(rid) > 36 else rid
        name_display = name[:22] + ".." if len(name) > 24 else name

        row = (
            f"  {i:<6}{name_display:<25}{rid_display:<38}"
            f"{scores.get('skills_score', 0):>6.1f}  "
            f"{scores.get('experience_score', 0):>6.1f}  "
            f"{scores.get('education_score', 0):>6.1f}  "
            f"{scores.get('soft_skill_score', 0):>6.1f}  "
            f"{cand.get('final_score', 0):>6.1f}"
        )
        lines.append(row)

    lines.append("-" * 105)
    lines.append("")

    # Per-candidate details
    for i, cand in enumerate(ranked_candidates, 1):
        name = cand.get("candidate_name", "Unknown")
        lines.append(f"  CANDIDATE #{i} — {name} ({cand.get('candidate_id', '?')})")
        lines.append(f"  Final Score: {cand.get('final_score', 0):.1f}/100")

        explanation = cand.get("llm_explanation", {})
        recommendation = explanation.get("recommendation", "N/A")
        lines.append(f"  Recommendation: {recommendation}")
        lines.append("")

        # Strengths
        strengths = explanation.get("strengths", [])
        if strengths:
            lines.append("  [+] Strengths:")
            for s in strengths:
                lines.append(f"    - {s}")
            lines.append("")

        # Weaknesses
        weaknesses = explanation.get("weaknesses", [])
        if weaknesses:
            lines.append("  [-] Weaknesses:")
            for w in weaknesses:
                lines.append(f"    - {w}")
            lines.append("")

        # Missing requirements
        missing = explanation.get("missing_requirements", [])
        if missing:
            lines.append("  [!] Missing Requirements:")
            for m in missing:
                lines.append(f"    - {m}")
            lines.append("")

        lines.append("  " + "-" * 40)
        lines.append("")

    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  JSON OUTPUT GENERATION
# ─────────────────────────────────────────────


def _generate_json_output(
    job_id: str,
    jd_text: str,
    ranked_candidates: List[Dict[str, Any]],
    parsed_jd: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the structured JSON output."""
    return {
        "job_id": job_id,
        "job_description": jd_text,
        "parsed_jd": {
            "required_skills": parsed_jd.get("required_skills", []),
            "preferred_skills": parsed_jd.get("preferred_skills", []),
            "required_experience": parsed_jd.get("required_experience", []),
            "education_requirements": parsed_jd.get("education_requirements", []),
            "soft_skills": parsed_jd.get("soft_skills", []),
        },
        "timestamp": datetime.now().isoformat(),
        "total_candidates_evaluated": len(ranked_candidates),
        "scoring_weights": {
            "skills": 0.40,
            "experience": 0.35,
            "education": 0.15,
            "soft_skill": 0.10,
        },
        "top_candidates": [
            {
                "rank": i + 1,
                "candidate_name": cand.get("candidate_name", "Unknown"),
                "candidate_id": cand.get("candidate_id", "?"),
                "final_score": cand.get("final_score", 0),
                "section_scores": cand.get("section_scores", {}),
                "llm_explanation": cand.get("llm_explanation", {}),
            }
            for i, cand in enumerate(ranked_candidates)
        ],
    }


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────


def generate_candidate_report(
    job_id: str,
    jd_text: str,
    ranked_candidates: List[Dict[str, Any]],
    parsed_jd: Dict[str, Any],
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Generate and save the complete candidate ranking report.

    Produces two files:
    - ``{job_id}_report.txt``  — human-readable text report
    - ``{job_id}_results.json`` — structured JSON output

    Parameters
    ----------
    job_id : str
        Unique identifier for this ranking job.
    jd_text : str
        Raw job description text.
    ranked_candidates : list[dict]
        Candidates sorted by final_score descending.
        Each must have: ``candidate_id``, ``final_score``,
        ``section_scores``, ``llm_explanation``.
    parsed_jd : dict
        Parsed JD from ``parse_job_description()``.
    output_dir : str or Path, optional
        Where to save outputs. Defaults to ``ranking_outputs/``.

    Returns
    -------
    dict
        The structured JSON output (also saved to disk).
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate JSON
    json_output = _generate_json_output(job_id, jd_text, ranked_candidates, parsed_jd)

    # Save JSON
    json_path = out_dir / f"{job_id}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON results → %s", json_path)

    # Generate and save text report
    text_report = _generate_text_report(job_id, jd_text, ranked_candidates)

    report_path = out_dir / f"{job_id}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    logger.info("Saved text report → %s", report_path)

    # Print report to console
    print(text_report)

    return json_output
