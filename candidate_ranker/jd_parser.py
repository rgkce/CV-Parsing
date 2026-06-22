"""
jd_parser.py — Job Description Parsing (Bilingual EN/TR)
=========================================================

Extracts structured fields from raw job description text using
hybrid NLP + rule-based extraction.  Supports both English and
Turkish job descriptions.

Output schema::

    {
        "required_skills": [],
        "preferred_skills": [],
        "required_experience": [],
        "education_requirements": [],
        "soft_skills": []
    }
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Set

from .config import (
    EDUCATION_FIELDS,
    EDUCATION_LEVELS,
    EXPERIENCE_KEYWORDS,
    PREFERRED_SIGNALS,
    REQUIRED_SIGNALS,
    SOFT_SKILLS,
    TECHNICAL_SKILLS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  TEXT NORMALISATION
# ─────────────────────────────────────────────


def _normalise(text: str) -> str:
    """
    Lowercase, strip excess whitespace, and normalise Unicode.

    Turkish-aware: preserves ç, ğ, ı, ö, ş, ü but normalises
    composed vs decomposed forms.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _split_into_segments(text: str) -> List[str]:
    """
    Split JD text into meaningful segments (sentences / bullet points).

    Handles:
    - Line breaks and bullet characters (•, -, *, ►, ●)
    - Numbered lists (1., 2., etc.)
    - Semicolons and sentence-ending punctuation
    """
    # Replace bullet characters with newlines
    text = re.sub(r"[•►●◆▪■]", "\n", text)
    # Split on line breaks, semicolons, and sentence endings
    segments = re.split(r"[\n;]+|(?<=[.!?])\s+", text)
    # Clean up and filter empty
    segments = [s.strip() for s in segments if s.strip()]
    return segments


def _check_signals(segment: str, signal_words: Set[str]) -> bool:
    """Check if a text segment contains any of the signal words."""
    segment_lower = _normalise(segment)
    for signal in signal_words:
        if signal in segment_lower:
            return True
    return False


# ─────────────────────────────────────────────
#  EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────


def _extract_technical_skills(text: str) -> List[str]:
    """
    Extract technical skills by matching against the bilingual
    skills dictionary.  Handles multi-word skills (e.g. "machine learning").
    """
    text_lower = _normalise(text)
    found: List[str] = []

    # Sort by length descending so "machine learning" matches before "machine"
    sorted_skills = sorted(TECHNICAL_SKILLS, key=len, reverse=True)

    for skill in sorted_skills:
        # Use word-boundary matching for single words,
        # substring matching for multi-word phrases
        if " " in skill:
            if skill in text_lower:
                found.append(skill)
                # Remove matched phrase to avoid double-counting
                text_lower = text_lower.replace(skill, " ")
        else:
            # Word boundary: match "python" but not "pythonic"
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, text_lower):
                found.append(skill)

    return list(dict.fromkeys(found))  # dedupe preserving order


def _extract_soft_skills(text: str) -> List[str]:
    """Extract soft skills from the JD text."""
    text_lower = _normalise(text)
    found: List[str] = []

    sorted_skills = sorted(SOFT_SKILLS, key=len, reverse=True)

    for skill in sorted_skills:
        if " " in skill:
            if skill in text_lower:
                found.append(skill)
        else:
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, text_lower):
                found.append(skill)

    return list(dict.fromkeys(found))


def _extract_education(text: str) -> List[Dict[str, str]]:
    """
    Extract education requirements: degree level + field of study.

    Returns list of dicts like:
        {"level": "bachelor", "field": "computer engineering"}
    """
    text_lower = _normalise(text)
    results: List[Dict[str, str]] = []

    # Find degree levels
    found_levels: List[str] = []
    for term, level in sorted(EDUCATION_LEVELS.items(), key=lambda x: len(x[0]), reverse=True):
        if " " in term:
            if term in text_lower:
                found_levels.append(level)
        else:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, text_lower):
                found_levels.append(level)

    # Deduplicate levels
    found_levels = list(dict.fromkeys(found_levels))

    # Find fields of study
    found_fields: List[str] = []
    for field in sorted(EDUCATION_FIELDS, key=len, reverse=True):
        if field in text_lower:
            found_fields.append(field)

    found_fields = list(dict.fromkeys(found_fields))

    # Combine: pair levels with fields if possible
    if found_levels and found_fields:
        for level in found_levels:
            for field in found_fields:
                results.append({"level": level, "field": field})
    elif found_levels:
        for level in found_levels:
            results.append({"level": level, "field": ""})
    elif found_fields:
        for field in found_fields:
            results.append({"level": "", "field": field})

    return results


def _extract_experience(text: str) -> List[str]:
    """
    Extract experience requirements.

    Looks for:
    - Year patterns: "3+ years", "en az 5 yıl deneyim"
    - Role/seniority: "senior", "kıdemli", "lead"
    - Domain experience phrases
    """
    text_lower = _normalise(text)
    found: List[str] = []

    # Year-based patterns (English)
    en_patterns = [
        r"(\d+\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?)",
        r"(at\s+least\s+\d+\s*(?:years?|yrs?))",
        r"(minimum\s+\d+\s*(?:years?|yrs?))",
    ]

    # Year-based patterns (Turkish)
    tr_patterns = [
        r"(\d+\+?\s*(?:yıl|yil)\s*(?:deneyim|tecrübe|tecrube)?)",
        r"(en\s+az\s+\d+\s*(?:yıl|yil))",
        r"(minimum\s+\d+\s*(?:yıl|yil))",
    ]

    for pattern in en_patterns + tr_patterns:
        matches = re.findall(pattern, text_lower)
        found.extend(matches)

    # Seniority keywords
    seniority_terms = {
        "senior", "junior", "mid-level", "entry-level",
        "lead", "principal", "staff", "intern", "internship",
        "kıdemli", "kidemli", "stajyer", "staj",
    }
    for term in seniority_terms:
        if " " in term:
            if term in text_lower:
                found.append(term)
        else:
            if re.search(rf"\b{re.escape(term)}\b", text_lower):
                found.append(term)

    return list(dict.fromkeys(found))


# ─────────────────────────────────────────────
#  REQUIRED vs PREFERRED CLASSIFICATION
# ─────────────────────────────────────────────


def _classify_skills(
    text: str,
    skills: List[str],
) -> tuple[List[str], List[str]]:
    """
    Classify extracted skills into required vs preferred based on
    surrounding context signals.

    Strategy:
    - Split text into segments
    - For each segment, check if it contains required/preferred signals
    - Skills found in preferred-signal segments → preferred
    - Everything else → required (conservative default)
    """
    segments = _split_into_segments(text)

    preferred_context_skills: Set[str] = set()

    for segment in segments:
        if _check_signals(segment, PREFERRED_SIGNALS):
            # Find which skills appear in this preferred-context segment
            segment_lower = _normalise(segment)
            for skill in skills:
                if skill in segment_lower:
                    preferred_context_skills.add(skill)

    required = [s for s in skills if s not in preferred_context_skills]
    preferred = [s for s in skills if s in preferred_context_skills]

    return required, preferred


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────


def parse_job_description(jd_text: str) -> Dict[str, Any]:
    """
    Parse a raw job description into structured fields.

    Parameters
    ----------
    jd_text : str
        Raw job description text (English or Turkish).

    Returns
    -------
    dict
        Structured JD with keys:
        - ``required_skills``      : list[str]
        - ``preferred_skills``     : list[str]
        - ``required_experience``  : list[str]
        - ``education_requirements``: list[dict]
        - ``soft_skills``          : list[str]
        - ``raw_text``             : str  (original input)
    """
    logger.info("Parsing job description (%d chars)", len(jd_text))

    # Extract all categories
    all_skills = _extract_technical_skills(jd_text)
    soft_skills = _extract_soft_skills(jd_text)
    education = _extract_education(jd_text)
    experience = _extract_experience(jd_text)

    # Classify technical skills into required vs preferred
    required_skills, preferred_skills = _classify_skills(jd_text, all_skills)

    result = {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "required_experience": experience,
        "education_requirements": education,
        "soft_skills": soft_skills,
        "raw_text": jd_text,
    }

    logger.info(
        "Parsed JD: %d required skills, %d preferred skills, "
        "%d experience reqs, %d education reqs, %d soft skills",
        len(required_skills), len(preferred_skills),
        len(experience), len(education), len(soft_skills),
    )

    return result
