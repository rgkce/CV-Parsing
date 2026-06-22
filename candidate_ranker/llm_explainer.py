"""
llm_explainer.py — LLM-Powered Candidate Explanation
======================================================

Uses Google Gemini (gemini-2.5-flash) to generate structured
explanations for candidate ranking decisions.

Falls back to template-based explanations when:
- No API key is configured (GOOGLE_API_KEY env var)
- LLM call fails or returns invalid JSON
- Rate limiting or network errors

Output schema::

    {
        "candidate_id": "23",
        "llm_score": 89,
        "strengths": [],
        "weaknesses": [],
        "missing_requirements": [],
        "recommendation": "Strong Match"
    }
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from .config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  RECOMMENDATION LABELS
# ─────────────────────────────────────────────

def _get_recommendation(score: float) -> str:
    """Map a score to a recommendation label."""
    if score >= 85:
        return "Strong Match"
    elif score >= 70:
        return "Good Match"
    elif score >= 55:
        return "Moderate Match"
    elif score >= 40:
        return "Weak Match"
    else:
        return "Not Recommended"


# ─────────────────────────────────────────────
#  TEMPLATE-BASED FALLBACK
# ─────────────────────────────────────────────


def _generate_template_explanation(
    candidate: Dict[str, Any],
    section_scores: Dict[str, float],
    final_score: float,
    parsed_jd: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a structured explanation without an LLM, using
    score thresholds and keyword matching.
    """
    candidate_id = candidate.get("resume_id", "unknown")

    strengths: List[str] = []
    weaknesses: List[str] = []
    missing: List[str] = []

    # ── Analyse skills ──
    skills_score = section_scores.get("skills_score", 0)
    cv_skills = candidate.get("sections", {}).get("skills", "").lower()

    if skills_score >= 75:
        strengths.append(f"Strong technical skill alignment (score: {skills_score:.0f})")
    elif skills_score >= 50:
        strengths.append(f"Moderate technical skill match (score: {skills_score:.0f})")
    else:
        weaknesses.append(f"Low technical skill alignment (score: {skills_score:.0f})")

    # Check which required skills are missing
    for skill in parsed_jd.get("required_skills", []):
        if skill.lower() not in cv_skills:
            missing.append(skill)

    # ── Analyse experience ──
    exp_score = section_scores.get("experience_score", 0)
    if exp_score >= 75:
        strengths.append(f"Relevant professional experience (score: {exp_score:.0f})")
    elif exp_score >= 50:
        strengths.append(f"Some relevant experience found (score: {exp_score:.0f})")
    else:
        weaknesses.append(f"Limited relevant experience (score: {exp_score:.0f})")

    # ── Analyse education ──
    edu_score = section_scores.get("education_score", 0)
    if edu_score >= 75:
        strengths.append(f"Education requirements satisfied (score: {edu_score:.0f})")
    elif edu_score >= 50:
        pass  # Neutral — don't mention
    else:
        weaknesses.append(f"Education requirements may not be fully met (score: {edu_score:.0f})")

    # ── Analyse soft skills ──
    soft_score = section_scores.get("soft_skill_score", 0)
    if soft_score >= 75:
        strengths.append(f"Good soft skill indicators (score: {soft_score:.0f})")
    elif soft_score < 40:
        weaknesses.append(f"Limited soft skill evidence in CV (score: {soft_score:.0f})")

    return {
        "candidate_id": candidate_id,
        "llm_score": round(final_score),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "missing_requirements": missing,
        "recommendation": _get_recommendation(final_score),
    }


# ─────────────────────────────────────────────
#  LLM CALL (GEMINI)
# ─────────────────────────────────────────────


def _build_prompt(
    jd_text: str,
    candidate: Dict[str, Any],
    section_scores: Dict[str, float],
    final_score: float,
) -> str:
    """Build the structured prompt for the LLM."""
    sections = candidate.get("sections", {})
    candidate_id = candidate.get("resume_id", "unknown")

    prompt = f"""You are an expert HR analyst evaluating a candidate CV against a job description.

## Job Description
{jd_text}

## Candidate Information (ID: {candidate_id})

**Title:** {sections.get('title', 'N/A')}

**Skills:** {sections.get('skills', 'N/A')}

**Experience:** {sections.get('experience', 'N/A')}

**Education:** {sections.get('education', 'N/A')}

**Projects:** {sections.get('projects', 'N/A')}

**Summary:** {sections.get('summary', 'N/A')}

## Computed Section Scores
- Skills Score: {section_scores.get('skills_score', 0):.1f}/100
- Experience Score: {section_scores.get('experience_score', 0):.1f}/100
- Education Score: {section_scores.get('education_score', 0):.1f}/100
- Soft Skill Score: {section_scores.get('soft_skill_score', 0):.1f}/100
- Weighted Final Score: {final_score:.1f}/100

## Task
Analyse this candidate against the job description. Consider the computed scores as a starting point but use your own judgement.

Return ONLY a JSON object with this exact structure (no markdown, no code fences):
{{
    "candidate_id": "{candidate_id}",
    "llm_score": <your score 0-100>,
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "missing_requirements": ["<missing 1>", "<missing 2>", ...],
    "recommendation": "<one of: Strong Match, Good Match, Moderate Match, Weak Match, Not Recommended>"
}}

Be specific and concise. Reference actual skills, experience, and qualifications from the CV.
Respond in the same language as the job description."""

    return prompt


def _call_gemini(prompt: str) -> Dict[str, Any] | None:
    """
    Call Gemini API and parse the JSON response.
    Returns None on any failure.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.info("No GOOGLE_API_KEY set — skipping LLM call")
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(LLM_MODEL)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )

        response_text = response.text.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            response_text = "\n".join(lines).strip()

        result = json.loads(response_text)

        # Validate required keys
        required_keys = {"candidate_id", "llm_score", "strengths", "weaknesses",
                         "missing_requirements", "recommendation"}
        if not required_keys.issubset(result.keys()):
            logger.warning("LLM response missing keys: %s", required_keys - set(result.keys()))
            return None

        return result

    except ImportError:
        logger.warning("google-generativeai not installed — using template fallback")
        return None
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON response: %s", e)
        return None
    except Exception as e:
        logger.warning("LLM call failed: %s — using template fallback", e)
        return None


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────


def generate_llm_explanation(
    jd_text: str,
    candidate: Dict[str, Any],
    section_scores: Dict[str, float],
    final_score: float,
    parsed_jd: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Generate a structured explanation for a candidate's ranking.

    Attempts to use Gemini LLM first; falls back to template-based
    explanation if the LLM is unavailable or fails.

    Parameters
    ----------
    jd_text : str
        The raw job description text.
    candidate : dict
        The candidate CV record.
    section_scores : dict
        Per-section scores from ``calculate_section_similarity()``.
    final_score : float
        Weighted final score from ``calculate_final_score()``.
    parsed_jd : dict, optional
        Parsed JD (used for template fallback to identify missing skills).

    Returns
    -------
    dict
        Structured explanation with keys:
        ``candidate_id``, ``llm_score``, ``strengths``, ``weaknesses``,
        ``missing_requirements``, ``recommendation``.
    """
    candidate_id = candidate.get("resume_id", "unknown")
    logger.info("Generating explanation for candidate: %s", candidate_id)

    # Try LLM first
    prompt = _build_prompt(jd_text, candidate, section_scores, final_score)
    llm_result = _call_gemini(prompt)

    if llm_result is not None:
        logger.info("  → LLM explanation generated successfully")
        return llm_result

    # Fallback to template
    logger.info("  → Using template-based explanation (fallback)")
    return _generate_template_explanation(
        candidate, section_scores, final_score,
        parsed_jd or {"required_skills": [], "preferred_skills": []},
    )
