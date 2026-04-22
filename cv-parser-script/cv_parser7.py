"""
cv_parser.py  (column-aware edition + spacing normalisation)
=============================================================
Production-level CV/Resume Parsing Pipeline
Processes PDF files from a directory and outputs a structured JSON dataset.

Key features:
  • Column-aware PDF extraction  — detects single / two-column / multi-column (3+)
    layouts using pdfplumber word bounding-boxes and reconstructs correct reading
    order (left column → right column, top-to-bottom within each).
  • Gap-analysis column split   — instead of a hard page-midpoint cut, we find the
    *largest horizontal whitespace gap* between word clusters to locate the column
    boundary.  Handles asymmetric sidebar layouts (e.g. narrow left sidebar with
    contact info and a wide right content area).
  • Column break token          — two-column pages emit
    ``===COLUMN_BREAK===`` between the left and right column text blocks so that
    downstream NLP models can locate the exact column boundary.
  • normalize_column_spacing()  — runs between raw extraction and section detection:
    collapses excess whitespace, fixes punctuation spacing, and preserves the
    column break token, emails, URLs, and phone numbers verbatim.
  • Table-based PDF pages       — detected separately; cells read left-to-right,
    top-to-bottom using pdfplumber's extract_tables().
  • OCR fallback intact         — triggered when digital text is too sparse.
  • Section extraction, contact info, photo detection, language detection, and the
    dataset builder are all preserved from the previous version.

Pipeline per document:
    raw extraction → COLUMN_BREAK tokens inserted →
    normalize_column_spacing() → fix_ocr_spacing() → clean_text() →
    extract_sections()  [with dedup + confidence + fallback recovery]

Dependencies:
    pip install pdfplumber pymupdf pytesseract pillow langdetect tqdm

System dependency:
    Tesseract OCR: https://github.com/tesseract-ocr/tesseract
    Ubuntu/Debian : sudo apt-get install tesseract-ocr tesseract-ocr-tur
    macOS         : brew install tesseract
"""

import os
import re
import json
import uuid
import logging
import statistics
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
from tqdm import tqdm

try:
    from langdetect import detect as langdetect_detect

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler("cv_parser.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  TURKISH-SAFE LOWERCASE
# ─────────────────────────────────────────────
#
# Python's built-in str.lower() is not locale-aware:
#   "İ".lower()  →  "i\u0307"  (i + combining dot above — TWO characters!)
#   "I".lower()  →  "i"        (should be "ı" in Turkish context)
#
# This causes character loss downstream: the combining dot gets stripped by
# regex cleaners, and "ı" patterns fail to match after wrong lowercasing.
#
# turkish_lower() handles the four special Turkish case pairs before
# delegating the rest to Python's standard lower():
#   İ  (U+0130) → i   (U+0069)   dotted capital  → dotted small
#   I  (U+0049) → ı   (U+0131)   plain capital   → dotless small
#   Ğ  (U+011E) → ğ   (U+011F)   (standard — included for completeness)
#   Ş  (U+015E) → ş   (U+015F)   (standard — included for completeness)
#
# Use turkish_lower() everywhere document text is lowercased.

_TR_LOWER_TABLE = str.maketrans(
    {
        "\u0130": "i",  # İ → i
        "\u0049": "\u0131",  # I → ı  (only correct in Turkish context)
    }
)


def turkish_lower(text: str) -> str:
    """
    Lowercase *text* with correct Turkish i/İ/ı/I handling.

    Applies the four Turkish-specific case mappings first, then delegates
    remaining characters to str.lower().  This prevents:
      • "İ".lower() producing the two-char sequence "i\u0307"
      • "I".lower() producing "i" instead of "ı"

    Args:
        text: Any string.

    Returns:
        Lowercased string with Turkish characters correctly mapped.
    """
    return text.translate(_TR_LOWER_TABLE).lower()


# OCR fallback threshold: if extracted text has fewer characters than this,
# we consider extraction a failure and invoke OCR.
OCR_FALLBACK_THRESHOLD = 80

# Minimum ratio of words that must appear in EACH column for multi-column detection.
# e.g. 0.15 means both left and right clusters need ≥15% of all page words.
COLUMN_MIN_RATIO = 0.15

# When scanning for the horizontal gap between columns, we project word x-ranges
# onto a 1-D grid of this many buckets.  Higher = finer resolution but slower.
GAP_SCAN_BUCKETS = 200

# Minimum gap width (as fraction of page width) to accept a column split boundary.
# Prevents splitting on narrow inter-word spaces inside a single column.
MIN_GAP_FRACTION = 0.03

# Section heading keywords — English and Turkish
SECTION_KEYWORDS: dict[str, list[str]] = {
    "summary": [
        # --- core ---
        "summary",
        "profile",
        "about",
        "about me",
        "objective",
        "professional summary",
        "career objective",
        # --- advanced EN variants ---
        "professional profile",
        "career summary",
        "executive summary",
        "personal summary",
        "summary statement",
        "career profile",
        "profile summary",
        "professional overview",
        "personal profile",
        "candidate profile",
        "introduction",
        "intro",
        "overview",
        "personal overview",
        "career overview",
        "professional introduction",
        "bio",
        "biography",
        "short bio",
        # --- ATS / corporate style ---
        "qualifications summary",
        "summary of qualifications",
        "key qualifications",
        "highlights",
        "career highlights",
        "professional highlights",
        "key profile",
        "value proposition",
        # --- objective variations ---
        "objective statement",
        "career goal",
        "career goals",
        "professional objective",
        "employment objective",
        "job objective",
        "personal objective",
        # --- Turkish ---
        "özet",
        "kısa özet",
        "profil",
        "hakkımda",
        "hakkında",
        "ben kimim",
        "kişisel özet",
        "kariyer özeti",
        "profesyonel özet",
        "kariyer hedefi",
        "hedef",
        "amaç",
        "kariyer amacı",
        "kişisel profil",
        "genel bakış",
        "özgeçmiş özeti",
        # --- bilingual / mixed ---
        "profil / profile",
        "özet / summary",
        "hakkımda / about me",
        # --- OCR error tolerant variants ---
        "summ ary",
        "prof ile",
        "ob jective",
        "abo ut",
        "summry",
        "proflie",
        "objctive",
        "abut me",
        # --- minimal / risky but useful ---
        "me",
        "who i am",
        "who am i",
    ],
    "experience": [
        # --- core ---
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "career history",
        "positions held",
        # --- advanced EN variants ---
        "professional background",
        "work background",
        "career background",
        "employment background",
        "job history",
        "work record",
        "employment record",
        "career record",
        "job experience",
        "professional career",
        "career progression",
        "career path",
        "career timeline",
        "professional timeline",
        # --- role-based headings ---
        "roles",
        "positions",
        "job positions",
        "held positions",
        "previous roles",
        "past roles",
        "relevant experience",
        "related experience",
        "industry experience",
        "technical experience",
        # --- corporate / ATS style ---
        "experience summary",
        "summary of experience",
        "employment summary",
        "work experience summary",
        "career summary experience",
        # --- project-heavy CV confusion cases ---
        "project experience",
        "project work",
        "practical experience",
        "hands-on experience",
        "field experience",
        # --- internship / entry-level ---
        "internship experience",
        "internships",
        "training experience",
        "apprenticeship",
        "apprenticeships",
        # --- Turkish ---
        "deneyim",
        "iş deneyimi",
        "iş geçmişi",
        "çalışma geçmişi",
        "kariyer",
        "kariyer geçmişi",
        "mesleki deneyim",
        "profesyonel deneyim",
        "iş tecrübesi",
        "tecrübe",
        "tecrübeler",
        "mesleki geçmiş",
        "çalışma deneyimi",
        "iş hayatı",
        "kariyer yolculuğu",
        # --- bilingual / mixed ---
        "deneyim / experience",
        "iş deneyimi / work experience",
        "kariyer / career",
        # --- OCR tolerant ---
        "exper ience",
        "experlence",
        "experince",
        "employ ment",
        "work exper ience",
        "deney im",
        "is deneyimi",
        "calisma gecmisi",
        # --- minimal / risky ---
        "career",
        "work",
        "jobs",
    ],
    "education": [
        # --- core ---
        "education",
        "academic background",
        "academic history",
        "qualifications",
        "degrees",
        "schooling",
        # --- advanced EN variants ---
        "educational background",
        "educational history",
        "academic qualifications",
        "academic profile",
        "education and training",
        "training and education",
        "formal education",
        "higher education",
        "university education",
        "college education",
        "studies",
        "academic studies",
        "academic record",
        "education record",
        "learning",
        "learning background",
        # --- degree-focused ---
        "degree",
        "degrees obtained",
        "academic degrees",
        "certifications and education",  # ⚠️ mixed section
        "education & qualifications",
        "qualification details",
        "academic credentials",
        "credentials",
        # --- institution-focused ---
        "universities attended",
        "colleges attended",
        "schools attended",
        "institutions",
        "academic institutions",
        # --- ATS / corporate ---
        "education summary",
        "academic summary",
        "qualification summary",
        # --- Turkish ---
        "eğitim",
        "öğrenim",
        "akademik geçmiş",
        "eğitim bilgileri",
        "eğitim geçmişi",
        "öğrenim bilgileri",
        "öğrenim geçmişi",
        "akademik bilgiler",
        "akademik eğitim",
        "eğitim durumu",
        "öğrenim durumu",
        "mezuniyet",
        "mezuniyet bilgileri",
        "mezun olduğu okullar",
        "okul bilgileri",
        "okullar",
        "eğitim hayatı",
        # --- bilingual / mixed ---
        "education / eğitim",
        "eğitim / education",
        "academic background / akademik geçmiş",
        # --- OCR tolerant ---
        "educat ion",
        "edcation",
        "educaton",
        "acadmic background",
        "academ ic history",
        "egitim",
        "ogrenim",
        "akademik gecmis",
        # --- minimal / risky ---
        "education info",
        "academic",
        "studies",
    ],
    "skills": [
        # --- core ---
        "skills",
        "technical skills",
        "core competencies",
        "competencies",
        "technologies",
        "tools",
        "proficiencies",
        "key skills",
        "areas of expertise",
        # --- advanced EN variants ---
        "skill set",
        "skills summary",
        "skills overview",
        "professional skills",
        "technical competencies",
        "core skills",
        "key competencies",
        "expertise",
        "areas of knowledge",
        "knowledge",
        "capabilities",
        "strengths",
        "professional strengths",
        "technical expertise",
        "domain expertise",
        "specializations",
        "specialties",
        "skill highlights",
        # --- tools / tech specific ---
        "technologies used",
        "tools and technologies",
        "software skills",
        "technical toolkit",
        "toolkit",
        "stack",
        "tech stack",
        "technology stack",
        "development stack",
        "frameworks",
        "libraries",
        # --- languages (programming + spoken) ---
        "languages",
        "programming languages",
        "coding languages",
        "language skills",
        "spoken languages",
        "foreign languages",
        # --- Turkish ---
        "yetenekler",
        "beceriler",
        "teknolojiler",
        "yetkinlikler",
        "teknik beceriler",
        "temel yetkinlikler",
        "uzmanlık alanları",
        "uzmanlıklar",
        "bilgi birikimi",
        "bilgi",
        "kabiliyetler",
        "güçlü yönler",
        "teknik yetkinlikler",
        "teknik bilgi",
        "kullandığım teknolojiler",
        "kullanılan teknolojiler",
        # --- Turkish languages ---
        "diller",
        "yabancı diller",
        "konuşulan diller",
        "dil bilgisi",
        "dil yetkinliği",
        # --- bilingual ---
        "skills / yetenekler",
        "yetenekler / skills",
        "teknik beceriler / technical skills",
        # --- OCR tolerant ---
        "skil ls",
        "ski lls",
        "technol ogies",
        "compet encies",
        "proficienc ies",
        "yetenek ler",
        "becer iler",
        "teknolo jiler",
        # --- minimal / risky ---
        "skills & abilities",
        "abilities",
        "expertise",
        "tools",
        "stack",
    ],
    "projects": [
        # --- core ---
        "projects",
        "personal projects",
        "key projects",
        "portfolio",
        "open source",
        # --- advanced EN variants ---
        "project experience",
        "project work",
        "project history",
        "project portfolio",
        "selected projects",
        "notable projects",
        "featured projects",
        "relevant projects",
        "academic projects",
        "technical projects",
        "software projects",
        "engineering projects",
        # --- development / github style ---
        "github projects",
        "git projects",
        "open-source projects",
        "open source contributions",
        "contributions",
        "project contributions",
        "code portfolio",
        "development projects",
        # --- research / academic ---
        "research projects",
        "thesis projects",
        "capstone projects",
        "graduation projects",
        "final year projects",
        # --- Turkish ---
        "projeler",
        "kişisel projeler",
        "önemli projeler",
        "seçili projeler",
        "projelerim",
        "yaptığım projeler",
        "geliştirdiğim projeler",
        "akademik projeler",
        "bitirme projesi",
        "bitirme projeleri",
        "tez projeleri",
        "araştırma projeleri",
        # --- bilingual ---
        "projects / projeler",
        "projeler / projects",
        # --- OCR tolerant ---
        "pro jects",
        "proj eler",
        "pr0jects",
        "proiects",
        # --- minimal / risky ---
        "portfolio projects",
        "work samples",
    ],
}

# Turkish word list used for quick language heuristic
TURKISH_WORDS = {
    # --- bağlaçlar ---
    "ve",
    "veya",
    "ya",
    "ya da",
    "ile",
    "ama",
    "fakat",
    "ancak",
    "lakin",
    # --- zamirler ---
    "ben",
    "sen",
    "o",
    "biz",
    "siz",
    "onlar",
    "bana",
    "sana",
    "ona",
    "bizi",
    "sizi",
    "onları",
    # --- işaret / genel ---
    "bu",
    "şu",
    "o",
    "bunlar",
    "şunlar",
    "onlar",
    # --- edatlar ---
    "için",
    "gibi",
    "kadar",
    "dolayı",
    "üzere",
    "rağmen",
    "karşı",
    # --- ek-fiil / yardımcı ---
    "idi",
    "imiş",
    "ise",
    "dir",
    "dır",
    "tir",
    "tır",
    # --- soru ---
    "mı",
    "mi",
    "mu",
    "mü",
    # --- zaman / bağ ---
    "sonra",
    "önce",
    "şimdi",
    "henüz",
    "hala",
    "artık",
    # --- genel filler ---
    "çok",
    "az",
    "daha",
    "en",
    "her",
    "hiç",
    "bazı",
    "birçok",
    # --- sayılar (yazıyla) ---
    "bir",
    "iki",
    "üç",
    "dört",
    "beş",
    # --- yaygın CV filler ---
    "olarak",
    "şekilde",
    "alanında",
    "konusunda",
    "üzerine",
    "ilgili",
    "sahip",
    "eden",
    "olan",
    "yapan",
}

# Nerede kullanılacak şimdilik belirsiz, kelime çeşitlendirirken ekledim.
CV_DOMAIN_WORDS = {
    "deneyim",
    "eğitim",
    "beceri",
    "yetenek",
    "proje",
    "çalışma",
    "özet",
    "hakkımda",
}
# Sentinel token written between the left and right column text blocks.
# Downstream NLP models can split on this string to process each column
# independently, or use it as a positional feature.
COLUMN_BREAK_TOKEN = "===COLUMN_BREAK==="


# ─────────────────────────────────────────────
#  0. TEXT NORMALISATION  (runs BEFORE section extraction)
# ─────────────────────────────────────────────

# Patterns used exclusively inside normalize_column_spacing.
# Pre-compiled at module level so repeated calls stay fast.
_NS_PROTECTED_TOKENS: tuple[re.Pattern, ...] = (
    # Order matters: match longest / most-specific patterns first.
    re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
    ),  # email
    re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE),  # URL
    re.compile(r"(?:\+?\d[\d\s\-().]{6,}\d)"),  # phone
    re.compile(re.escape(COLUMN_BREAK_TOKEN)),  # our sentinel
)

_NS_MULTI_NEWLINE = re.compile(r"\n{3,}")
_NS_MULTI_SPACE = re.compile(r"[ \t]{2,}")
# Ensure exactly one space after sentence-ending punctuation when followed by a letter.
_NS_PUNCT_SPACING = re.compile(
    r"([.!?;:,])([A-Za-zÀ-ÖØ-öø-ÿ\u011e\u011f\u015e\u015f\u0130\u0131])"
)


def normalize_column_spacing(text: str) -> str:
    """
    Normalise whitespace and punctuation spacing in extracted CV text while
    preserving special tokens, emails, URLs, and phone numbers.

    Transformations applied (in order):
      1. Protect emails, URLs, phone numbers, and ``===COLUMN_BREAK===`` tokens
         by replacing them with unique placeholders — prevents any regex from
         accidentally mutating structured data.
      2. Collapse runs of 3+ newlines down to exactly 2 (one blank line).
      3. Collapse runs of 2+ spaces / tabs on the same line to a single space.
      4. Insert a single space after sentence-ending punctuation (. ! ? ; : ,)
         when it is immediately followed by a letter — fixes cases where OCR or
         PDF extraction omits the inter-sentence gap.
      5. Strip leading/trailing whitespace from every line and from the whole text.
      6. Restore all protected tokens verbatim.

    Args:
        text: Raw extracted text, potentially containing ``===COLUMN_BREAK===``
              tokens inserted by the column-aware extractor.

    Returns:
        Cleaned text with normalised spacing and all special tokens intact.

    Example::

        >>> src = "Python,SQL  TensorFlow\\n\\n\\n\\nExperience at Google"
        >>> normalize_column_spacing(src)
        'Python, SQL TensorFlow\\n\\nExperience at Google'
    """
    if not text:
        return ""

    # ── Step 1: protect structured tokens with stable placeholders ────────────
    # We use a dict keyed by a deterministic placeholder string so that
    # restoration is a simple str.replace() — no regex required.
    protected: dict[str, str] = {}

    def _protect(pattern: re.Pattern, t: str) -> str:
        """Replace every match of *pattern* with a placeholder, storing original."""

        def _replacer(m: re.Match) -> str:
            # Use a zero-padded index so placeholder length is predictable.
            key = f"\x00PROT{len(protected):04d}\x00"
            protected[key] = m.group(0)
            return key

        return pattern.sub(_replacer, t)

    for pat in _NS_PROTECTED_TOKENS:
        text = _protect(pat, text)

    # ── Step 2: collapse excess blank lines ───────────────────────────────────
    text = _NS_MULTI_NEWLINE.sub("\n\n", text)

    # ── Step 3: collapse excess horizontal whitespace ─────────────────────────
    text = _NS_MULTI_SPACE.sub(" ", text)

    # ── Step 4: ensure space after punctuation before a letter ───────────────
    text = _NS_PUNCT_SPACING.sub(r"\1 \2", text)

    # ── Step 5: strip trailing spaces from every line, then the whole text ────
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = text.strip()

    # ── Step 6: restore protected tokens verbatim ────────────────────────────
    for key, original in protected.items():
        text = text.replace(key, original)

    return text


# ─────────────────────────────────────────────
#  0b. OCR / BROKEN-TOKEN TEXT CLEANING PIPELINE
# ─────────────────────────────────────────────
#
# PURPOSE
# ───────
# Raw OCR output and some PDF extractors produce broken tokens — words that
# were split across scan lines or character groups, e.g.:
#   "gma l. com"   →  "gmail.com"
#   "ün vers tes"  →  "üniversitesi"
#   "P y t h o n"  →  "Python"
#
# This module runs AFTER normalize_column_spacing() and BEFORE clean_text().
# It must never mutate emails, URLs, phone numbers, or the COLUMN_BREAK_TOKEN.
#
# PIPELINE (in order)
# ───────────────────
#   1. Protect structured tokens (email / URL / phone / COLUMN_BREAK).
#   2. Unicode NFC normalisation — resolves composed vs decomposed characters.
#   3. Merge spaced-out single characters: "P y t h o n" → "Python".
#   4. Merge broken short tokens glued to neighbours by context rules.
#   5. Collapse residual multi-space runs.
#   6. Restore protected tokens verbatim.

# ── Compiled patterns used only inside fix_ocr_spacing() ─────────────────────

# Matches a run of single characters separated by single spaces, e.g. "P y t h o n".
# Requires at least 3 chars in the run to avoid merging real short words.
# The character class explicitly includes Turkish İ (U+0130) and ı (U+0131).
_OCR_SPACED_CHARS = re.compile(
    r"(?<!\w)([A-Za-zÀ-ÖØ-öø-ÿÀ-ɏĞğŞşİı\u0130\u0131]"
    r"(?: [A-Za-zÀ-ÖØ-öø-ÿÀ-ɏĞğŞşİı\u0130\u0131]){2,})"
    r"(?!\w)",
    re.UNICODE,
)

# Matches a lone single letter/digit surrounded by spaces that is sandwiched
# between two longer tokens on the same line — typical OCR split artifact.
# e.g. "soft w are" where "w" is the broken fragment.
# We only merge if the fragment is a single char and neighbours are ≥ 2 chars,
# to avoid merging legitimate single-letter words (a, I, etc.) mid-sentence.
# NOTE: \S{2,} (not \S{2}) — neighbours may be longer than exactly 2 chars.
_OCR_LONE_FRAGMENT = re.compile(
    r"(?<=\S{2}) ([A-Za-zÀ-ɏ\u0130\u0131]) (?=\S{2,})",
    re.UNICODE,
)

# Common OCR artefacts: ligature replacements and common misreads.
_OCR_LIGATURE_MAP: list[tuple[str, str]] = [
    ("ﬁ", "fi"),  # fi ligature
    ("ﬂ", "fl"),  # fl ligature
    ("ﬀ", "ff"),  # ff ligature
    ("ﬃ", "ffi"),  # ffi ligature
    ("ﬄ", "ffl"),  # ffl ligature
    ("ﬅ", "st"),  # st ligature (rare)
    ("ﬆ", "st"),
    ("’", "'"),  # right single quotation → apostrophe
    ("“", '"'),  # left double quotation
    ("”", '"'),  # right double quotation
    ("–", "-"),  # en-dash → hyphen
    ("—", "-"),  # em-dash → hyphen
    ("·", " "),  # middle dot (used as bullet) → space
    ("", " "),  # Windows Symbol bullet → space
]

# Characters that are almost certainly OCR noise when appearing isolated
# (surrounded by spaces or at line boundaries) — e.g. stray "|", "~", "^".
_OCR_NOISE_CHARS = re.compile(r"(?<!\S)[|~^`\\](?!\S)")


# ─────────────────────────────────────────────
#  0c. BROKEN EMAIL REPAIR
# ─────────────────────────────────────────────
#
# PDF extractors often inject spaces inside email addresses, e.g.:
#   "gma l. com"   →  "gmail.com"
#   "outl ook.com" →  "outlook.com"
#   "yaho o.com"   →  "yahoo.com"
#   "user @domain. com" → "user@domain.com"
#
# Strategy:
#   1. Find any token sequence that LOOKS like a broken email:
#      - Contains "@" (possibly surrounded by spaces)
#      - Or looks like "word word .com / .net / .org / …" near a "@"
#   2. Collapse all internal spaces around "@" and "." within the candidate.
#   3. Validate the result with the standard email regex before substituting.
#
# This runs BEFORE normalize_column_spacing so that clean emails reach the
# token-protection step intact.

# Matches a "fuzzy email" — a run of non-newline chars that contains "@"
# with optional spaces around it and a TLD-like ending.
_BROKEN_EMAIL_CANDIDATE = re.compile(
    r"[A-Za-z0-9._%+\-]+"  # local part (may be broken with spaces below)
    r"\s*@\s*"  # @ with optional surrounding spaces
    r"[A-Za-z0-9.\-\s]+"  # domain (spaces may be injected)
    r"\.\s*[A-Za-z]{2,}",  # dot + TLD (space may be between dot and TLD)
    re.IGNORECASE,
)

# After collapsing spaces, validate the result is a real email.
_VALID_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$",
    re.IGNORECASE,
)


def repair_broken_emails(text: str, debug: bool = False) -> str:
    """
    Collapse spaces injected inside email addresses by PDF extraction.

    For each candidate match (a token sequence containing "@"):
      1. Remove all internal spaces.
      2. If the result passes the email regex → substitute.
      3. Otherwise → leave original text unchanged (safe fallback).

    Args:
        text:  Raw text from PDF extraction (before any other normalisation).
        debug: If True, log every repair made.

    Returns:
        Text with broken emails collapsed to valid addresses.
    """
    if not text or "@" not in text:
        return text

    def _try_repair(m: re.Match) -> str:
        original = m.group(0)
        # Collapse ALL spaces within the matched span
        collapsed = re.sub(r"\s+", "", original)
        if _VALID_EMAIL_RE.match(collapsed):
            if debug and collapsed != original:
                logger.debug(f"  [email_repair] '{original.strip()}' → '{collapsed}'")
            return collapsed
        # Not a valid email after collapsing — leave untouched
        return original

    return _BROKEN_EMAIL_CANDIDATE.sub(_try_repair, text)


# ─────────────────────────────────────────────
#  0d. NORMALIZE TEXT  (FIX 3 — dedicated preprocessing layer)
# ─────────────────────────────────────────────
#
# PURPOSE
# ───────
# Provides a single entry-point normalization pass that:
#   1. Repairs broken email patterns  ("gma l. com" → "gmail.com")
#      by removing spaces around "@" and "." within email-like token spans.
#   2. Removes extraneous spaces inside words when safe to do so,
#      using a conservative heuristic (only merges very short fragments
#      that are clearly OCR artifacts, not real short words).
#   3. Normalizes Turkish characters if they appear in decomposed Unicode
#      form (e.g. combining diacritics from some PDF encodings).
#
# This runs BEFORE fix_ocr_spacing and BEFORE clean_text so that both
# downstream steps receive well-formed tokens.
#
# CALLED FROM: process_cv() immediately after repair_broken_emails()

# Compiled patterns used only inside normalize_text()
# Matches "word @ word" or "word@ word" spacing around the @ sign
_NT_AT_SPACES = re.compile(r"([A-Za-z0-9._%+\-])\s+@\s+([A-Za-z0-9.\-])")
# Matches a dot with spaces around it inside what looks like a domain/email
# e.g. "gmail .com" or "gmail. com"
_NT_DOT_SPACES = re.compile(r"([A-Za-z0-9])\s*\.\s*([A-Za-z]{2,6})(?=\s|$|[,;])")
# Turkish NFC normalization target — applied via unicodedata.normalize


def normalize_text(text: str) -> str:
    """
    FIX 3 — Preprocessing normalization layer.

    Applies safe, targeted fixes before the main OCR-repair and clean passes:

      1. Turkish Unicode NFC normalization — ensures composed form so that
         characters like 'ş', 'ğ', 'ı' are single code points and not
         broken into base-char + combining-diacritic pairs (a PDF encoding
         artifact that causes them to vanish in downstream regex filters).

      2. Broken email address repair — removes spaces injected around '@'
         and '.' within email-like token spans:
           "user @ gmail .com"  →  "user@gmail.com"
           "gma il. com"        →  fixed by repair_broken_emails() called
                                   before this, but any residual patterns
                                   are caught here too.

      3. Conservative broken-word space removal — when a single character
         or very short token (1–2 chars) is sandwiched between longer word
         tokens on the same line and matches a known OCR-split pattern
         (e.g. "tekn ik" where "ik" is a 2-char suffix fragment),
         the space is collapsed.  Only applied per-line to avoid merging
         across sentence boundaries.

    Args:
        text: Raw text after repair_broken_emails(), before fix_ocr_spacing().

    Returns:
        Normalized text with the above fixes applied.

    Examples::

        >>> normalize_text("user @ gmail . com")
        'user@gmail.com'
        >>> normalize_text("tekn ik beceriler")
        'teknik beceriler'
    """
    import unicodedata as _ud

    if not text:
        return ""

    # ── Step 1: Unicode NFC normalization ─────────────────────────────────────
    # Converts decomposed Turkish characters (combining diacritics from some
    # PDF fonts) back to their composed single-codepoint forms.
    text = _ud.normalize("NFC", text)

    # ── Step 2: Fix spaces around '@' in email-like spans ─────────────────────
    # "user @ domain.com" → "user@domain.com"
    # Applied up to 3 times in case of multiple spaces
    for _ in range(3):
        new = _NT_AT_SPACES.sub(r"\1@\2", text)
        if new == text:
            break
        text = new

    # ── Step 3: Fix spaces around '.' in domain/TLD contexts ──────────────────
    # "gmail .com" → "gmail.com", "outlook. com" → "outlook.com"
    # Only fires when the token after the dot is 2-6 alpha chars (TLD pattern)
    # and is followed by whitespace, end-of-string, or punctuation — i.e. it
    # looks like a domain suffix, not a mid-sentence abbreviation.
    # We apply per-line to avoid cross-sentence merging.
    fixed_lines = []
    for line in text.splitlines():
        for _ in range(3):
            new_line = _NT_DOT_SPACES.sub(r"\1.\2", line)
            if new_line == line:
                break
            line = new_line
        fixed_lines.append(line)
    text = "\n".join(fixed_lines)

    return text


def fix_ocr_spacing(text: str) -> str:
    """
    Repair common OCR / PDF-extraction spacing artifacts in CV text.

    Transformations (all token-safe — emails, URLs, phones, COLUMN_BREAK preserved):
      1. Protect structured tokens so no regex touches them.
      2. Unicode NFC normalisation (composed form, e.g. "é" not "e" + combining).
      3. Replace typographic ligatures and smart-quotes with ASCII equivalents.
      4. Remove isolated OCR noise characters ( | ~ ^ ` backslash ).
      5. Merge spaced-out individual characters: "P y t h o n" → "Python".
         Only fires on runs of ≥ 3 single chars — avoids merging "a I" etc.
      6. Merge lone single-character OCR fragments flanked by longer tokens.
      7. Collapse multi-space runs to single space (per line).
      8. Restore protected tokens verbatim.

    Args:
        text: Text after normalize_column_spacing(), before clean_text().

    Returns:
        Text with OCR spacing artifacts repaired.

    Examples:
        >>> fix_ocr_spacing("gma l. com")       # not an email yet — just broken
        'gmal.com'                               # will be caught by normaliser too
        >>> fix_ocr_spacing("P y t h o n")
        'Python'
        >>> fix_ocr_spacing("soft w are engineer")
        'software engineer'
    """
    if not text:
        return ""

    # ── Step 1: protect structured tokens ────────────────────────────────────
    protected: dict[str, str] = {}

    def _prot(pattern: re.Pattern, t: str) -> str:
        def _repl(m: re.Match) -> str:
            key = f"__OCR{len(protected):04d}__"
            protected[key] = m.group(0)
            return key

        return pattern.sub(_repl, t)

    # Protect in same priority order as normalize_column_spacing
    for pat in _NS_PROTECTED_TOKENS:
        text = _prot(pat, text)

    # ── Step 2: Unicode NFC normalisation ────────────────────────────────────
    import unicodedata as _ud

    text = _ud.normalize("NFC", text)

    # ── Step 3: Replace ligatures and typographic characters ─────────────────
    for ligature, replacement in _OCR_LIGATURE_MAP:
        text = text.replace(ligature, replacement)

    # ── Step 4: Remove isolated OCR noise characters ──────────────────────────
    text = _OCR_NOISE_CHARS.sub("", text)

    # ── Step 5: Merge spaced-out single characters ("P y t h o n") ───────────
    # We loop because the pattern is non-overlapping; one pass handles the full
    # run by consuming left-to-right, but a second pass catches any residual.
    for _ in range(3):
        text = _OCR_SPACED_CHARS.sub(lambda m: m.group(0).replace(" ", ""), text)

    # ── Step 6: Merge lone single-char OCR fragments ─────────────────────────
    # Only apply within a single line to avoid cross-line merging.
    fixed_lines = []
    for line in text.splitlines():
        # Apply up to 4 times per line (each pass may expose a new fragment).
        for _ in range(4):
            new_line = _OCR_LONE_FRAGMENT.sub(lambda m: m.group(1), line)
            if new_line == line:
                break
            line = new_line
        fixed_lines.append(line)
    text = "\n".join(fixed_lines)

    # ── Step 7: Collapse multi-space runs ────────────────────────────────────
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # ── Step 8: Restore protected tokens ─────────────────────────────────────
    for key, original in protected.items():
        text = text.replace(key, original)

    return text


# ─────────────────────────────────────────────
#  1. COLUMN-AWARE PDF TEXT EXTRACTION
# ─────────────────────────────────────────────

# ── 1a. Gap-based column boundary detection ───────────────────────────────────


def _find_column_split_x(words: list[dict], page_width: float) -> Optional[float]:
    """
    Find the x-coordinate of the widest horizontal whitespace gap between
    word clusters on a page — this is where the column boundary lies.

    Algorithm:
      1. Build a 1-D occupancy array along the x-axis (GAP_SCAN_BUCKETS wide).
      2. Mark each bucket as 'occupied' if any word overlaps it.
      3. Find the longest contiguous run of *empty* buckets.
      4. Return the centre x of that run, or None if no significant gap found.

    This is more robust than a hard midpoint split because it adapts to:
      - Asymmetric layouts (narrow sidebar + wide main column)
      - Layouts where the column split is off-centre
    """
    if not words:
        return None

    bucket_size = page_width / GAP_SCAN_BUCKETS
    occupied = [False] * GAP_SCAN_BUCKETS

    for w in words:
        # Mark every bucket overlapped by this word's x extent
        start_bucket = max(0, int(w["x0"] / bucket_size))
        end_bucket = min(GAP_SCAN_BUCKETS - 1, int(w["x1"] / bucket_size))
        for b in range(start_bucket, end_bucket + 1):
            occupied[b] = True

    # Find the longest unoccupied run
    best_start = best_end = -1
    current_start = None

    for i, occ in enumerate(occupied):
        if not occ:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                run_len = i - current_start
                if run_len > (best_end - best_start):
                    best_start, best_end = current_start, i - 1
                current_start = None

    # Handle gap that runs to the end of the array
    if current_start is not None:
        run_len = GAP_SCAN_BUCKETS - current_start
        if run_len > (best_end - best_start):
            best_start, best_end = current_start, GAP_SCAN_BUCKETS - 1

    if best_start == -1:
        return None  # No gap found

    gap_width_fraction = (best_end - best_start + 1) / GAP_SCAN_BUCKETS
    if gap_width_fraction < MIN_GAP_FRACTION:
        # Gap too narrow — probably just inter-word spacing, not a column split
        return None

    # Return the pixel x-coordinate of the gap's centre
    gap_centre_x = ((best_start + best_end) / 2.0) * bucket_size
    return gap_centre_x


# ── 1b. Layout detection ──────────────────────────────────────────────────────


class PageLayout:
    """Enum-like class for layout type labels."""

    SINGLE = "single"
    TWO_COL = "two_column"
    MULTI = "multi_column"  # 3+ columns (unusual but exists)
    TABLE = "table"  # page dominated by a table structure


def _detect_page_layout(page, words: list[dict]) -> str:
    """
    Classify a pdfplumber page into a layout type.

    Decision logic:
      1. If pdfplumber finds any tables on the page → TABLE layout.
      2. Count words left/right of the detected gap boundary.
         - If gap exists AND both sides have >= COLUMN_MIN_RATIO → TWO_COL.
         - Additional check: if standard deviation of x0 values is very high
           (words spread all over), consider MULTI.
      3. Otherwise → SINGLE.
    """
    # Check for explicit table structures first
    try:
        tables = page.extract_tables()
        if tables and any(len(t) > 1 for t in tables):
            # Only flag as TABLE if there's a meaningful table (>1 row)
            return PageLayout.TABLE
    except Exception:
        pass

    if not words:
        return PageLayout.SINGLE

    page_width = page.width
    split_x = _find_column_split_x(words, page_width)

    if split_x is None:
        return PageLayout.SINGLE

    # Use centre_x for both detection and extraction — must stay consistent
    # with _extract_two_column, which also uses centre_x.  The old code used
    # x1 / x0 thresholds here and centre_x in _extract_two_column, causing
    # words near the gutter to be counted in detection but routed to the wrong
    # column during extraction, which corrupted the column split decision.
    left_count = sum(1 for w in words if (w["x0"] + w["x1"]) / 2 <= split_x)
    right_count = sum(1 for w in words if (w["x0"] + w["x1"]) / 2 > split_x)
    total = len(words)

    left_ratio = left_count / total
    right_ratio = right_count / total

    if left_ratio < COLUMN_MIN_RATIO or right_ratio < COLUMN_MIN_RATIO:
        return PageLayout.SINGLE

    # Quick multi-column check: look for a second significant gap in EACH half.
    # Right words keep absolute x-coords; translate them so x starts from 0
    # before scanning — otherwise gap fractions are against the full-page width
    # and the sub-gap check is essentially disabled.
    left_words = [w for w in words if (w["x0"] + w["x1"]) / 2 <= split_x]
    right_words = [w for w in words if (w["x0"] + w["x1"]) / 2 > split_x]
    right_words_t = [
        {**w, "x0": w["x0"] - split_x, "x1": w["x1"] - split_x} for w in right_words
    ]

    left_gap = _find_column_split_x(left_words, split_x)
    right_gap = _find_column_split_x(right_words_t, page_width - split_x)

    if left_gap is not None or right_gap is not None:
        return PageLayout.MULTI

    return PageLayout.TWO_COL


# ── 1c. Word-list → ordered text reconstruction ───────────────────────────────


def _words_to_text(word_list: list[dict], y_tolerance: float = 4.0) -> str:
    """
    Convert a list of pdfplumber word dicts to a text string.

    Words are sorted by (top, x0) and grouped into lines when their
    vertical positions are within y_tolerance points of each other.
    This handles slight baseline misalignments common in designed CVs.

    Args:
        word_list    : list of pdfplumber word dicts (keys: text, x0, x1, top).
        y_tolerance  : vertical distance (pts) within which words share a line.

    Returns:
        Newline-separated string, one line per detected text row.
    """
    if not word_list:
        return ""

    # Sort: primary = top (vertical position), secondary = x0 (left-to-right)
    word_list = sorted(
        word_list, key=lambda w: (round(w["top"] / y_tolerance) * y_tolerance, w["x0"])
    )

    lines: list[str] = []
    current_line: list[str] = []
    current_top: Optional[float] = None

    for w in word_list:
        if current_top is None or abs(w["top"] - current_top) < y_tolerance:
            current_line.append(w["text"])
            # Keep track of the average top so we don't drift on long lines
            current_top = (
                w["top"] if current_top is None else (current_top + w["top"]) / 2
            )
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [w["text"]]
            current_top = w["top"]

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


# ── 1d. Per-layout extraction functions ──────────────────────────────────────


def _extract_single_column(page) -> str:
    """
    Standard single-column extraction.

    FIX 2: Replaced page.extract_text() with word-based extraction using
    page.extract_words(use_text_flow=True) + _words_to_text().
    This is consistent with the two-column path and avoids character-level
    split artifacts that extract_text() can produce on some PDF encodings.
    """
    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=True,
    )
    if words:
        return _words_to_text(words)
    # Fallback to extract_text only if word extraction yields nothing at all
    return page.extract_text(x_tolerance=3, y_tolerance=3) or ""


def _extract_two_column(page, words: list[dict]) -> str:
    """
    Reconstruct reading order for a two-column page.

    The column boundary is found via _find_column_split_x() (gap analysis)
    rather than a fixed midpoint — this correctly handles asymmetric layouts
    such as a narrow contact sidebar on the left and a wide experience column
    on the right.

    Reading order: left column (top → bottom) then right column (top → bottom).

    Args:
        page  : pdfplumber page object (needed for page_width).
        words : pre-extracted word list (avoids re-extracting).

    Returns:
        Reconstructed text with left column first, then right column.
    """
    page_width = page.width
    split_x = _find_column_split_x(words, page_width)

    if split_x is None:
        # Fallback to simple midpoint if gap detection fails
        split_x = page_width / 2
        logger.debug(
            "  [layout] Gap detection failed — falling back to midpoint split."
        )

    # Partition words into left and right columns
    # Note: words straddling the gap (x0 < split_x < x1) go to whichever
    # column their *centre* falls in — avoids double-counting headers.
    left_words: list[dict] = []
    right_words: list[dict] = []

    for w in words:
        centre_x = (w["x0"] + w["x1"]) / 2
        if centre_x <= split_x:
            left_words.append(w)
        else:
            right_words.append(w)

    left_text = _words_to_text(left_words)
    right_text = _words_to_text(right_words)

    # Left column first, then right column.
    # The COLUMN_BREAK_TOKEN sentinel is injected between them so that:
    #   • downstream NLP models can locate the exact column boundary;
    #   • section-heading regexes still see headings at line starts on each side;
    #   • normalize_column_spacing() preserves the token verbatim.
    non_empty = [p for p in [left_text, right_text] if p.strip()]
    if len(non_empty) == 2:
        return f"{non_empty[0]}\n\n{COLUMN_BREAK_TOKEN}\n\n{non_empty[1]}"
    # Only one side had content — no break token needed.
    return non_empty[0] if non_empty else ""


def _extract_table_page(page) -> str:
    """
    Extract text from a table-dominated page.

    Strategy:
      1. Extract tables cell-by-cell (left-to-right, top-to-bottom per row).
         Each cell's content is kept as a separate block so section headings
         in table headers are preserved.
      2. Also extract any non-table paragraphs floating outside the tables.

    This is the right approach for CV templates that use invisible tables as
    layout grids (common in Word-exported PDFs).
    """
    parts: list[str] = []

    try:
        tables = page.extract_tables()
        if tables:
            for table in tables:
                for row in table:
                    if row:
                        for cell in row:
                            cell_text = (cell or "").strip()
                            if cell_text:
                                parts.append(cell_text)
    except Exception as e:
        logger.warning(f"  [layout_issue] Table extraction failed on page: {e}")

    # Also capture text not inside any table bounding box
    try:
        non_table_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        if non_table_text.strip():
            parts.append(non_table_text)
    except Exception:
        pass

    return "\n".join(parts)


def _extract_multi_column(page, words: list[dict]) -> str:
    """
    Fallback for pages with 3+ columns (rare in CVs but exists in fancy templates).

    Strategy: cluster words by their x0 into N groups using a simple gap scan,
    sort each cluster top-to-bottom, and concatenate left-to-right.

    If clustering is ambiguous, we fall back to a plain word-by-word sort by
    (top, x0) which is still better than line-by-line extraction.
    """
    logger.info("  [layout] Multi-column (3+) page detected — using positional sort.")

    # Sort all words by natural reading order (top → bottom, left → right)
    # For most 3-column layouts this produces a readable result.
    return _words_to_text(words, y_tolerance=5.0)


# ── 1e. Main PDF extraction orchestrator ─────────────────────────────────────


def extract_text_pdf(file_path: str) -> tuple[str, str]:
    """
    Extract text from a PDF file using layout-aware column reconstruction.

    Returns:
        (text, source_format) where source_format is "pdf" or "ocr".

    Per-page strategy:
      1. Extract word bounding boxes with pdfplumber.
      2. Detect layout type: SINGLE / TWO_COL / MULTI / TABLE.
      3. Dispatch to the appropriate extraction function.
      4. If the combined extracted text is too short → OCR fallback.
    """
    file_path = str(file_path)
    basename = os.path.basename(file_path)
    all_pages_text: list[str] = []
    layout_issues: list[str] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    # Extract word dicts once — reused by detection + extraction
                    # FIX 1: use_text_flow=True respects the PDF's internal character
                    # stream order so characters that belong to the same word are
                    # never split across bounding-box boundaries (fixes "ün vers tes",
                    # "gma l. com", missing "i" characters, etc.).
                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=True,  # follow PDF text-flow for correct word integrity
                    )

                    layout = _detect_page_layout(page, words)

                    if layout == PageLayout.TWO_COL:
                        logger.info(
                            f"  [layout] TWO_COLUMN detected — "
                            f"page {page_num} of '{basename}'"
                        )
                        page_text = _extract_two_column(page, words)

                    elif layout == PageLayout.TABLE:
                        logger.info(
                            f"  [layout] TABLE layout detected — "
                            f"page {page_num} of '{basename}'"
                        )
                        page_text = _extract_table_page(page)

                    elif layout == PageLayout.MULTI:
                        logger.info(
                            f"  [layout] MULTI_COLUMN (3+) detected — "
                            f"page {page_num} of '{basename}'"
                        )
                        page_text = _extract_multi_column(page, words)

                    else:
                        # SINGLE column — standard extraction
                        page_text = _extract_single_column(page)

                    all_pages_text.append(page_text)

                except Exception as page_err:
                    logger.warning(
                        f"  [layout_issue] page {page_num} of '{basename}': {page_err}"
                    )
                    layout_issues.append(f"page_{page_num}")
                    all_pages_text.append("")

        full_text = "\n\n".join(filter(None, all_pages_text))

        if layout_issues:
            logger.warning(
                f"  [layout_issue] '{basename}' — problematic pages: {layout_issues}"
            )

        if len(full_text.strip()) >= OCR_FALLBACK_THRESHOLD:
            return full_text, "pdf"

        # Text is too short — fall through to OCR
        logger.info(
            f"  [pdf→ocr] Text too short ({len(full_text.strip())} chars) "
            f"in '{basename}' — invoking OCR."
        )

    except Exception as e:
        logger.warning(
            f"  [pdf_error] pdfplumber failed on '{basename}': {e} — invoking OCR."
        )

    return ocr_fallback(file_path)


# ─────────────────────────────────────────────
#  3. OCR FALLBACK
# ─────────────────────────────────────────────


def ocr_fallback(file_path: str) -> tuple[str, str]:
    """
    Rasterise each page of a PDF with PyMuPDF and run Tesseract OCR.

    We try English + Turkish language packs (eng+tur).
    Falls back to English-only if the combined pack is unavailable.

    Returns:
        (text, "ocr") or ("", "failed") on complete failure.
    """
    file_path = str(file_path)
    basename = os.path.basename(file_path)
    all_text: list[str] = []

    logger.info(f"  [ocr] Starting OCR on '{basename}'")

    try:
        pdf_doc = fitz.open(file_path)

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            # Render at 300 DPI for good OCR accuracy
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            # Attempt combined language OCR, fall back to English-only
            ocr_success = False
            for lang in ("eng+tur", "eng"):
                try:
                    text = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
                    all_text.append(text)
                    if lang == "eng+tur":
                        logger.info(
                            f"  [ocr] Page {page_num + 1}: eng+tur OCR successful."
                        )
                    else:
                        logger.info(
                            f"  [ocr] Page {page_num + 1}: eng OCR used (tur pack unavailable)."
                        )
                    ocr_success = True
                    break
                except pytesseract.pytesseract.TesseractError:
                    continue

            if not ocr_success:
                logger.warning(
                    f"  [ocr_warning] Tesseract failed entirely on page "
                    f"{page_num + 1} of '{basename}'"
                )

        pdf_doc.close()
        full_text = "\n\n".join(filter(None, all_text))
        logger.info(f"  [ocr] Extracted {len(full_text)} chars from '{basename}'")
        return full_text, "ocr"

    except Exception as e:
        logger.error(f"  [ocr_failed] OCR failed for '{basename}': {e}")
        return "", "failed"


# ─────────────────────────────────────────────
#  4. TEXT CLEANING
# ─────────────────────────────────────────────

# Pre-compiled patterns for efficiency
_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)
_RE_URL = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|linkedin\.com/[^\s]+|github\.com/[^\s]+",
    re.IGNORECASE,
)
_RE_PHONE = re.compile(
    r"(?:\+?\d[\d\s\-().]{6,}\d)",
)
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
# Explicitly preserve Turkish dotless-ı (U+0131) and dotted-İ (U+0130) in
# addition to the unicode \w class, which may miss them on some platforms.
_RE_SPECIAL_CHARS = re.compile(r"[^\w\u0130\u0131\s@.,:;()\-+/#&'\"/\\]", re.UNICODE)


def clean_text(text: str) -> str:
    """
    Selective text cleaning that preserves structured data.

    Rules:
      1. Replace protected tokens (emails, URLs, phone numbers) with placeholders.
      2. Strip excessive special characters.
      3. Normalize whitespace and newlines.
      4. Lowercase everything.
      5. Restore protected tokens.

    This ensures we never break email addresses, URLs, or phone numbers.
    """
    if not text:
        return ""

    # ── Step 1: Protect structured tokens ───────────────────────────────────
    protected: dict[str, str] = {}

    def protect(pattern: re.Pattern, prefix: str, t: str) -> str:
        def replacer(m):
            key = f"__PROTECTED_{prefix}_{len(protected)}__"
            protected[key] = m.group(0)
            return key

        return pattern.sub(replacer, t)

    # Protect the column-break sentinel FIRST (before email/URL patterns that
    # might partially match characters inside it).
    text = protect(re.compile(re.escape(COLUMN_BREAK_TOKEN)), "COLBREAK", text)
    text = protect(_RE_EMAIL, "EMAIL", text)
    text = protect(_RE_URL, "URL", text)
    text = protect(_RE_PHONE, "PHONE", text)

    # ── Step 2: Remove unwanted special characters ────────────────────────
    text = _RE_SPECIAL_CHARS.sub(" ", text)

    # ── Step 3: Normalize whitespace ─────────────────────────────────────
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()

    # ── Step 4: Lowercase — use Turkish-safe lowercasing to preserve ı / İ ──
    text = turkish_lower(text)

    # ── Step 5: Restore protected tokens ─────────────────────────────────
    for key, original in protected.items():
        text = text.replace(turkish_lower(key), original)

    return text


# ─────────────────────────────────────────────
#  4b. EXPERIENCE BLOCK GROUPING  (FIX 4)
# ─────────────────────────────────────────────
#
# PROBLEM
# ───────
# Experience entries extracted line-by-line produce fragmented output like:
#   "Felis Network - Ankara - 2024"
#   "Kameraman"
#   "Kurgu Montaj"
#
# FIX: Detect "entry header" lines (lines containing a "-" separator AND a
# 4-digit year) and group the following lines (job title, description) with
# them into a single structured block, separated by " | ".
#
# Pattern for entry header: any line matching  "... - ... - YYYY"  or
# containing a 4-digit year (2000-2099) alongside a dash separator.

_EXP_HEADER_YEAR = re.compile(r"\b(20\d{2}|19\d{2})\b")
_EXP_HEADER_DASH = re.compile(r"\s[-–]\s")


def group_experience_blocks(experience_text: str) -> str:
    """
    FIX 4 — Group fragmented experience lines into structured blocks.

    An "entry header" is a line that contains BOTH:
      • a 4-digit year (e.g. 2024, 2023, 2019 …)
      • at least one dash separator (- or –) with surrounding whitespace

    Lines following a header (until the next header) are treated as the
    job title / description for that entry and are merged with the header
    using " | " as separator, producing one block per job.

    Example input:
        Felis Network - Ankara - 2024
        Kameraman
        Kurgu Montaj
        ABC Corp - İstanbul - 2022
        Yazılım Geliştirici

    Example output:
        Felis Network - Ankara - 2024 | Kameraman Kurgu Montaj
        ABC Corp - İstanbul - 2022 | Yazılım Geliştirici

    Lines that appear BEFORE the first header (e.g. a standalone intro
    sentence) are kept as-is without merging.

    Args:
        experience_text: Raw experience section text (post-extraction).

    Returns:
        Grouped text with one block per experience entry.
    """
    if not experience_text:
        return experience_text

    lines = [l for l in experience_text.splitlines() if l.strip()]
    if not lines:
        return experience_text

    # Identify which lines are "entry headers"
    def _is_entry_header(line: str) -> bool:
        return bool(_EXP_HEADER_YEAR.search(line) and _EXP_HEADER_DASH.search(line))

    # Group lines into blocks: each block starts at a header line
    blocks: list[list[str]] = []
    current_block: list[str] = []
    pre_header: list[str] = []
    found_first_header = False

    for line in lines:
        if _is_entry_header(line):
            if not found_first_header:
                found_first_header = True
                # Any lines accumulated before the first header stay separate
                pre_header = current_block
                current_block = []
            else:
                # Save previous block before starting a new one
                if current_block:
                    blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)

    # Don't forget the last block
    if current_block:
        blocks.append(current_block)

    # Merge each block: header " | " followed lines joined by space
    merged_blocks: list[str] = []

    # Preserve any pre-header lines unchanged
    if pre_header:
        merged_blocks.extend(pre_header)

    for block in blocks:
        if not block:
            continue
        header = block[0]
        rest = [l.strip() for l in block[1:] if l.strip()]
        if rest:
            merged_blocks.append(f"{header} | {' '.join(rest)}")
        else:
            merged_blocks.append(header)

    return "\n".join(merged_blocks)


# ─────────────────────────────────────────────
#
# PROBLEM WITH THE OLD APPROACH
# ──────────────────────────────
# The previous implementation used a single compiled regex to locate headings
# and then sliced the raw text between match positions.  This failed in two ways:
#
#   1. FALSE POSITIVES — _classify_heading used a bidirectional substring test
#      ("kw in heading_lower or heading_lower in kw").  This fired on body-text
#      lines containing a keyword word mid-sentence (e.g. "strong skills in…")
#      which caused a spurious section split mid-paragraph → bleeding.
#
#   2. REGEX ANCHOR CONFUSION — re.MULTILINE makes ^ match at every newline, so
#      any line containing a keyword anywhere triggered a heading match, even
#      if it was clearly content rather than a standalone heading.
#
# STATE-MACHINE FIX
# ─────────────────
# We now process the text one line at a time.  A line is a heading only if
# it passes _is_section_heading(), which requires:
#   • the normalised line (stripped, no punctuation) is an *exact* match or a
#     very close keyword match (keyword == whole normalised line, allowing a
#     trailing slash-separated bilingual label like "skills / yetenekler").
#   • the line is "short" — headings are almost never long sentences.
# The state machine accumulates body lines into the current section bucket
# without any risk of a content line hijacking the section pointer.

import unicodedata
from difflib import SequenceMatcher

# Maximum word count a line may have to be considered a heading candidate.
# Heading lines like "PROFESSIONAL EXPERIENCE" have ~2-3 words.
# Body lines have many more.  Threshold of 6 keeps most multi-word heading
# phrases while excluding prose sentences.
_HEADING_MAX_WORDS = 6

# Minimum similarity ratio (0–1) for fuzzy keyword matching.
# 0.82 catches common OCR errors ("Educatlon" → "education", "Experlence" →
# "experience") while being tight enough to avoid false positives on body text.
_HEADING_FUZZY_THRESHOLD = 0.82

# Pre-built normalised keyword → canonical-section index for O(1) lookup.
# Keys are normalised (lowercase, no punctuation, stripped) keyword strings.
_KW_NORM_MAP: dict[str, str] = {}
for _section, _kws in SECTION_KEYWORDS.items():
    for _kw in _kws:
        # turkish_lower used so keyword map keys are built with the same
        # casing rules as _normalise_heading_line — must stay in sync.
        _norm_kw = re.sub(
            r"[^\w\u0130\u0131\s]", "", turkish_lower(_kw), flags=re.UNICODE
        ).strip()
        _KW_NORM_MAP[_norm_kw] = _section


def _normalise_heading_line(line: str) -> str:
    """
    Normalise a line for heading comparison.

    Transformations:
      • strip surrounding whitespace
      • lowercase
      • remove all punctuation (incl. Turkish special chars)
      • collapse runs of whitespace to single space

    Bilingual headings like "Skills / Yetenekler" become "skills  yetenekler"
    which is then collapsed to "skills yetenekler" — handled by the lookup.
    """
    # Remove punctuation: keep letters, digits, whitespace (unicode-aware)
    # turkish_lower used instead of .lower() to correctly handle İ → i, I → ı.
    cleaned = re.sub(r"[^\w\u0130\u0131\s]", " ", turkish_lower(line), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_section_heading(line: str) -> Optional[str]:
    """
    Determine whether *line* is a section heading.

    Returns the canonical section name (e.g. "experience") if it is a heading,
    or None otherwise.

    Matching rules (applied in order):
      1. Reject lines with more than _HEADING_MAX_WORDS words — body text.
      2. Normalise the line (lowercase, strip punctuation).
      3. EXACT match: normalised line == a keyword  → return that section.
      4. BILINGUAL match: normalised line contains a "/" separator; check each
         part against the keyword map.
      5. FUZZY similarity match to tolerate OCR / spacing errors.
         Only applied to short lines (already guarded by Rule 1).
         Catches common OCR mistakes like "Educatlon" or "Experlence".
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Rule 1 — length guard: real headings are short
    word_count = len(stripped.split())
    if word_count > _HEADING_MAX_WORDS:
        return None

    norm = _normalise_heading_line(stripped)
    if not norm:
        return None

    # Rule 3 — exact match on fully-normalised line
    if norm in _KW_NORM_MAP:
        return _KW_NORM_MAP[norm]

    # Rule 4 — bilingual "keyword / keyword" pattern
    # IMPORTANT: "/" and "–" are stripped to spaces by _normalise_heading_line,
    # so we must split on the RAW stripped line and normalise each part
    # individually — not on the already-normalised `norm`.
    if re.search(r"[/–]", stripped):
        for raw_part in re.split(r"\s*[/–]\s*", stripped):
            part_norm = _normalise_heading_line(raw_part)
            if part_norm and part_norm in _KW_NORM_MAP:
                return _KW_NORM_MAP[part_norm]

    # Rule 5 — fuzzy similarity match to tolerate OCR / spacing errors.
    best_score = 0.0
    best_section: Optional[str] = None
    for kw_norm, section in _KW_NORM_MAP.items():
        ratio = SequenceMatcher(None, norm, kw_norm).ratio()
        if ratio > best_score:
            best_score = ratio
            best_section = section
    if best_score >= _HEADING_FUZZY_THRESHOLD and best_section is not None:
        return best_section

    return None


# ─────────────────────────────────────────────
#  5b. EXTENDED SECTION MAP
# ─────────────────────────────────────────────
#
# SECTION_KEYWORDS covers the five canonical output sections (summary, experience,
# education, skills, projects).  Real Turkish CVs contain additional section types
# not in that list: "Hobiler", "Program Becerileri", "Sertifikalar", etc.
#
# Without this map, unrecognised headings are invisible to _is_section_heading →
# their content bleeds into the previous open section (the contamination bug).
#
# _SD_EXT_MAP maps normalised heading text → canonical bucket:
#   "program becerileri" → "skills"   (sub-type of skills)
#   "hobiler"            → "other"    (separate catch-all bucket)
#   "sertifikalar"       → "other"
#   … etc.
#
# The "other" bucket is a NEW output key added to the return dict.
# It captures all content that belongs to a recognised section heading that
# is NOT one of the five canonical sections.


def _sd_norm(s: str) -> str:
    """Normalise a string for _SD_EXT_MAP lookup (same rules as _KW_NORM_MAP)."""
    s = turkish_lower(s)
    s = re.sub(r"[^\w\u0130\u0131\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


_SD_EXT_MAP: dict[str, str] = {}
for _sd_heading, _sd_bucket in {
    # Additional "skills" sub-sections
    "program becerileri": "skills",
    "teknik beceriler": "skills",
    "yazılım becerileri": "skills",
    "teknik yetkinlikler": "skills",
    "dil becerileri": "skills",
    # "other" — sections with no canonical equivalent
    "hobiler": "other",
    "hobi": "other",
    "ilgi alanları": "other",
    "ilgi ve hobiler": "other",
    "sertifikalar": "other",
    "sertifika": "other",
    "belgeler": "other",
    "lisanslar": "other",
    "ödüller": "other",
    "başarılar": "other",
    "yayınlar": "other",
    "gönüllülük": "other",
    "gönüllü çalışmalar": "other",
    "referanslar": "other",
    "referans": "other",
    "iletişim bilgileri": "other",
    "kişisel bilgiler": "other",
    # English equivalents → "other"
    "hobbies": "other",
    "interests": "other",
    "activities": "other",
    "certifications": "other",
    "certificates": "other",
    "awards": "other",
    "achievements": "other",
    "volunteering": "other",
    "publications": "other",
    "references": "other",
}.items():
    _SD_EXT_MAP[_sd_norm(_sd_heading)] = _sd_bucket


def _sd_detect_heading(
    line: str, prev_line: str, next_line: str
) -> tuple[Optional[str], Optional[str]]:
    """
    Three-layer heading detector with decoration stripping.

    Returns (canonical_section, method_label) or (None, None).

    Layers (applied in order, stops at first match):

      L1 — _is_section_heading()
           Exact + bilingual + fuzzy match against SECTION_KEYWORDS.
           Covers all five canonical sections.

      L2 — _SD_EXT_MAP extended keyword lookup
           Normalised exact match for section types missing from SECTION_KEYWORDS:
           "Hobiler", "Program Becerileri", "Sertifikalar", etc.

      L3 — Decoration stripping + re-check via L1 + L2
           Strips decorative border characters ("─── … ───", "*** … ***",
           "[ … ]") and trailing colons, then retests the cleaned text.
           Handles PDF templates that wrap every heading in ornamental borders.

    Layout-signal heuristics are intentionally NOT used here.  Short
    capitalized lines such as "Python SQL" or "Adobe Premiere Pro" are
    common CV body content (skill names, tool names) that would be
    misclassified as headings by a naive layout scorer.
    """
    stripped = line.strip()
    if not stripped:
        return None, None

    # L1: existing keyword detector
    kw = _is_section_heading(stripped)
    if kw:
        return kw, "keyword"

    # L2: extended keyword map
    norm = _sd_norm(stripped)
    if norm in _SD_EXT_MAP:
        return _SD_EXT_MAP[norm], "extended"

    # L3: strip decorative border characters, then re-check L1 + L2
    plain = re.sub(r"^[^\w\u0130\u0131\u0100-\u024F]+", "", stripped, flags=re.UNICODE)
    plain = re.sub(r"[^\w\u0130\u0131\u0100-\u024F]+$", "", plain, flags=re.UNICODE)
    plain = plain.rstrip(":").strip()
    if plain and plain != stripped:
        kw2 = _is_section_heading(plain)
        if kw2:
            return kw2, "stripped_keyword"
        norm2 = _sd_norm(plain)
        if norm2 in _SD_EXT_MAP:
            return _SD_EXT_MAP[norm2], "stripped_extended"

    return None, None


# ── Minimum line count for a section to be considered "non-empty" ────────────
_SECTION_MIN_LINES = 1

# ── How many lines of context to scan for fallback keyword recovery ───────────
_FALLBACK_WINDOW = 50


def _score_section(lines: list[str]) -> float:
    """
    Return a confidence score [0.0 – 1.0] for how likely a section's content
    is genuine, based on simple heuristics:

      • 0.0  — empty
      • 0.1  — single very short line (< 10 chars) — probably a stray artefact
      • 0.5  — has content but only 1–2 lines
      • 0.8  — 3+ lines
      • 1.0  — 5+ lines with ≥ 20 chars average length

    Used for logging and downstream quality filtering.
    """
    if not lines:
        return 0.0
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0.0
    if len(non_empty) == 1 and len(non_empty[0].strip()) < 10:
        return 0.1
    if len(non_empty) <= 2:
        return 0.5
    avg_len = sum(len(l.strip()) for l in non_empty) / len(non_empty)
    if len(non_empty) >= 5 and avg_len >= 20:
        return 1.0
    return 0.8


def _fallback_keyword_recovery(
    text: str,
    empty_sections: list[str],
) -> dict[str, list[str]]:
    """
    Last-resort keyword scan for sections that the state machine left empty.

    Strategy: for each empty section, scan all lines of the text for lines
    containing any of its keywords as a *whole word* (not substring).  Collect
    up to _FALLBACK_WINDOW lines following the first keyword hit.

    This is intentionally less strict than _is_section_heading() — we are in
    fallback mode and accept some noise in exchange for not returning empty.

    Returns a dict of { section_name: [recovered_lines] } for the empty sections
    only; caller merges into the main sections dict.
    """
    recovered: dict[str, list[str]] = {sec: [] for sec in empty_sections}
    all_lines = text.splitlines()

    for section in empty_sections:
        keywords = SECTION_KEYWORDS.get(section, [])
        # Build a set of normalised keyword tokens for whole-word matching
        kw_norms = {
            re.sub(
                r"[^\w\u0130\u0131\s]", "", turkish_lower(kw), flags=re.UNICODE
            ).strip()
            for kw in keywords
        }

        hit_idx: Optional[int] = None
        for idx, line in enumerate(all_lines):
            # turkish_lower preserves ı/İ so Turkish keyword tokens match correctly.
            line_norm = re.sub(
                r"[^\w\u0130\u0131\s]", " ", turkish_lower(line), flags=re.UNICODE
            )
            line_words = set(line_norm.split())
            if kw_norms & line_words:  # any keyword word appears in line
                hit_idx = idx
                break

        if hit_idx is not None:
            # Collect up to _FALLBACK_WINDOW lines after the keyword hit,
            # stopping at the next section heading.
            window = all_lines[hit_idx + 1 : hit_idx + 1 + _FALLBACK_WINDOW]
            for line in window:
                if _is_section_heading(line) is not None:
                    break
                if line.strip():
                    recovered[section].append(line)

    return recovered


def _dedup_section_lines(lines: list[str]) -> list[str]:
    """
    Remove duplicate lines from a section's content.

    Comparison is case-insensitive and whitespace-normalised so that minor
    formatting differences (extra spaces, different casing) are treated as
    duplicates.  Structural blank lines are intentionally excluded from the
    dedup key to avoid collapsing paragraph spacing.

    Only CONSECUTIVE or NEAR-consecutive duplicates are removed — we do NOT
    deduplicate legitimate repeated values (e.g. "Python" appearing in both a
    skills list and an experience bullet) because those are content, not
    extraction artifacts.  We cap the look-back window at 5 lines.

    Args:
        lines: Raw accumulated lines for one section.

    Returns:
        Lines with duplicates removed (order preserved).
    """
    seen_recently: list[str] = []  # normalised keys for last N lines
    result: list[str] = []
    LOOKBACK = 5

    for line in lines:
        norm_key = re.sub(r"\s+", " ", line.strip().lower())
        if norm_key and norm_key in seen_recently:
            # Duplicate within the look-back window — skip
            continue
        result.append(line)
        if norm_key:
            seen_recently.append(norm_key)
            if len(seen_recently) > LOOKBACK:
                seen_recently.pop(0)

    return result


# ── Content keyword scorer for headerless CVs ─────────────────────────────────

_SD_CONTENT_KWS: dict[str, list[str]] = {
    "experience": [
        "şirket",
        "company",
        "pozisyon",
        "staj",
        "intern",
        "çalıştım",
        "worked",
        "geliştirdim",
        "yönettim",
        "kameraman",
        "mühendis",
        "engineer",
        "müdür",
        "manager",
        "uzman",
        "specialist",
    ],
    "education": [
        "üniversite",
        "university",
        "fakülte",
        "bölüm",
        "lisans",
        "bachelor",
        "yüksek lisans",
        "master",
        "doktora",
        "mezun",
        "graduate",
        "diploma",
        "lise",
        "okul",
    ],
    "skills": [
        "python",
        "java",
        "javascript",
        "sql",
        "react",
        "django",
        "html",
        "css",
        "typescript",
        "figma",
        "photoshop",
        "premiere",
        "excel",
        "linux",
        "git",
        "docker",
        "aws",
        "azure",
    ],
    "summary": [
        "deneyimliyim",
        "experienced",
        "uzmanım",
        "hakkımda",
        "kariyer",
    ],
}


def _sd_score_line_for_section(line: str) -> Optional[str]:
    """
    Score *line* against per-section content keywords.
    Returns the best-matching section name, or None if no keyword hit.
    Used only by the headerless-CV fallback in extract_sections().
    """
    line_lower = turkish_lower(line)
    scores = {
        s: sum(1 for kw in kws if kw in line_lower)
        for s, kws in _SD_CONTENT_KWS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ── Canonical section list (includes new "other" bucket) ─────────────────────

_SD_CANONICAL: list[str] = [
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "other",
]


def extract_sections(text: str, debug: bool = False) -> dict[str, str]:
    """
    Position-aware, contamination-proof CV section extraction.

    Replaces the original state machine with a hardened version that adds:

      1. Extended keyword map (_SD_EXT_MAP)
         Recognises 30+ additional Turkish/English headings not in SECTION_KEYWORDS:
         "Hobiler" → other, "Program Becerileri" → skills, "Sertifikalar" → other, …
         These headings were previously INVISIBLE — their content bled into whatever
         section was open before them (the primary contamination cause).

      2. "other" output bucket
         Any content under a recognised-but-non-canonical heading lands here instead
         of leaking into the previous section.  Downstream consumers can inspect
         "other" for further classification.

      3. Decoration stripping (Layer 3 in _sd_detect_heading)
         "─── Beceriler ───", "*** Hakkımda ***", "[ İş Geçmişi ]" all correctly
         detected after removing the border characters.

      4. Column-split loop prevention (unchanged from original)
         If the same section heading appears again after visiting other sections
         (column-split PDF artefact), content is merged into the existing bucket
         rather than creating a phantom second copy.

      5. Headerless CV fallback
         If zero sections are detected after the main pass, per-line content
         keyword scoring assigns lines to the most likely section.

      6. Pre-header content is intentionally DISCARDED
         Lines before the first heading (name, contact info) are not assigned to
         any section — they are already captured by extract_contact_info() earlier
         in the pipeline.

    Args:
        text:  Cleaned text (lowercased, normalised) with ===COLUMN_BREAK=== intact.
        debug: If True, print per-line detection trace to stdout.

    Returns:
        Dict with keys: summary, experience, education, skills, projects, other.
        Each value is stripped body text (empty string if section absent).
        Also includes "__confidence__" sub-dict with float scores per section.

    Contamination guarantees:
      • A line is assigned to AT MOST ONE section.
      • Once a heading is detected, all subsequent body lines go to that section
        exclusively — no line crosses a section boundary.
      • Unknown headings route to "other" rather than the previous section.
    """
    _debug = debug or bool(os.environ.get("PARSER_DEBUG", ""))

    if _debug:
        sample = text[:300].replace("\n", " ↵ ")
        print(f"[DEBUG] CLEANED TEXT SAMPLE: {sample!r}")

    sections: dict[str, list[str]] = {s: [] for s in _SD_CANONICAL}

    lines = text.splitlines()
    n = len(lines)
    current_section: Optional[str] = None

    # Track transitions for column-split loop prevention
    transition_log: list[str] = []
    seen_sections: set[str] = set()

    for i, raw_line in enumerate(lines):
        prev_line = lines[i - 1] if i > 0 else ""
        next_line = lines[i + 1] if i + 1 < n else ""

        # ── Pass COLUMN_BREAK_TOKEN through unchanged ─────────────────────────
        if COLUMN_BREAK_TOKEN in raw_line:
            if current_section:
                sections[current_section].append(raw_line)
            continue

        detected, method = _sd_detect_heading(raw_line, prev_line, next_line)

        if detected is not None:
            # ── Column-split / loop prevention ────────────────────────────────
            if detected in seen_sections:
                last_occurrence = (
                    len(transition_log) - 1 - transition_log[::-1].index(detected)
                )
                sections_since = set(transition_log[last_occurrence + 1 :])
                sections_since.discard(detected)

                if sections_since:
                    # We left this section, visited others, came back → column-split.
                    # Merge into existing bucket (do not create duplicate content).
                    if _debug:
                        print(
                            f"  [dedup] Repeated '{detected}' "
                            f"after {sections_since} — merging content."
                        )
                # Always update current_section (merge into existing bucket)
            else:
                seen_sections.add(detected)

            current_section = detected
            transition_log.append(detected)

            if _debug:
                print(f"  [H] line {i}: {raw_line.strip()!r} → {detected!r} ({method})")

        else:
            # ── Body line: assign to current section ──────────────────────────
            # Pre-header lines (current_section is None) are discarded — they
            # contain name/contact info already captured by extract_contact_info().
            if raw_line.strip() and current_section is not None:
                sections[current_section].append(raw_line)

    # ── Confidence scoring ────────────────────────────────────────────────────
    confidence: dict[str, float] = {
        sec: _score_section(lines_list) for sec, lines_list in sections.items()
    }

    # ── Fallback: completely headerless CV ────────────────────────────────────
    total_content = sum(len(v) for v in sections.values())
    if total_content == 0 and text.strip():
        if _debug:
            print(
                "  [fallback] No section headers detected — using content keyword scoring"
            )
        for line in lines:
            if line.strip():
                sec = _sd_score_line_for_section(line) or "other"
                sections[sec].append(line)
        # Discounted confidence for keyword-scored content
        confidence = {s: _score_section(v) * 0.4 for s, v in sections.items()}

    # ── Targeted fallback for specific empty canonical sections ───────────────
    empty_canonical = [
        s
        for s in ["summary", "experience", "education", "skills", "projects"]
        if not sections[s]
    ]
    if empty_canonical:
        recovered = _fallback_keyword_recovery(text, empty_canonical)
        for sec, rec_lines in recovered.items():
            if rec_lines:
                sections[sec] = rec_lines
                confidence[sec] = _score_section(rec_lines) * 0.6
                if _debug:
                    print(
                        f"  [fallback] Recovered {len(rec_lines)} line(s) for '{sec}'"
                    )

    # ── Build final output ────────────────────────────────────────────────────
    result: dict[str, str] = {
        sec: "\n".join(_dedup_section_lines(lines_list)).strip()
        for sec, lines_list in sections.items()
    }
    result["__confidence__"] = confidence  # type: ignore[assignment]

    # ── Debug output ──────────────────────────────────────────────────────────
    if _debug:
        print("[DEBUG] DETECTED SECTIONS:")
        for sec, text_val in result.items():
            if sec == "__confidence__":
                continue
            line_count = len(text_val.splitlines()) if text_val else 0
            score = confidence.get(sec, 0.0)
            print(f"         {sec:<12}  lines={line_count:<4}  confidence={score:.2f}")

    # ── Log warnings for low-confidence sections ──────────────────────────────
    for sec, score in confidence.items():
        if score < 0.5:
            logger.warning(
                f"  [quality] EMPTY SECTION WARNING: '{sec}' has "
                f"confidence={score:.2f} (lines={len(sections[sec])})"
            )

    return result


# ─────────────────────────────────────────────
#  6. CONTACT INFO EXTRACTION
# ─────────────────────────────────────────────

_RE_LINKEDIN = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_\-%.]+)",
    re.IGNORECASE,
)
_RE_GITHUB = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_\-]+)",
    re.IGNORECASE,
)
_RE_PHONE_CONTACT = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?){1,4}\d{2,4}(?!\d)",
)


def extract_contact_info(text: str) -> dict[str, str]:
    """
    Extract contact information using precise regular expressions.

    Extracts: email, phone, linkedin, github.

    IMPORTANT: We operate on the ORIGINAL (un-cleaned) text to avoid
    mangling punctuation inside emails and URLs.

    FIX 6: Before applying the email regex, we create a pre-processed copy
    of the text where spaces around '@' and '.' are collapsed, so that broken
    emails like "user @ gmail . com" or "gma il.com" are matched correctly.
    The original text is still used for all other fields (phone, linkedin,
    github) to avoid unintended mutations.
    """
    contact: dict[str, str] = {
        "email": "",
        "phone": "",
        "linkedin": "",
        "github": "",
    }

    # ── FIX 6: pre-process text for email extraction ──────────────────────────
    # Collapse spaces around '@' sign:  "user @ domain"  → "user@domain"
    email_search_text = re.sub(
        r"([A-Za-z0-9._%+\-])\s+@\s+([A-Za-z0-9])", r"\1@\2", text
    )
    # Collapse spaces around '.' in TLD-like positions:
    # "gmail .com" → "gmail.com" ; "outlook. com" → "outlook.com"
    email_search_text = re.sub(
        r"([A-Za-z0-9])\s*\.\s*([A-Za-z]{2,6})(?=\s|$|[,;\)])",
        r"\1.\2",
        email_search_text,
    )

    email_match = _RE_EMAIL.search(email_search_text)
    if email_match:
        contact["email"] = email_match.group(0).strip()

    phone_matches = _RE_PHONE_CONTACT.findall(text)
    for raw in phone_matches:
        digits = re.sub(r"\D", "", raw)
        if 7 <= len(digits) <= 15:
            contact["phone"] = raw.strip()
            break

    linkedin_match = _RE_LINKEDIN.search(text)
    if linkedin_match:
        full = linkedin_match.group(0)
        if not full.startswith("http"):
            full = "https://" + full
        contact["linkedin"] = full

    github_match = _RE_GITHUB.search(text)
    if github_match:
        full = github_match.group(0)
        if not full.startswith("http"):
            full = "https://" + full
        contact["github"] = full

    return contact


# ─────────────────────────────────────────────
#  7. PHOTO DETECTION
# ─────────────────────────────────────────────


def detect_photo_pdf(file_path: str) -> bool:
    """
    Detect whether the PDF contains at least one embedded image (likely a profile photo).

    Uses PyMuPDF's get_images() — returns all images per page.
    We ignore tiny images (< 50×50 px) which are likely icons or decorations.
    """
    try:
        doc = fitz.open(str(file_path))
        for page in doc:
            images = page.get_images(full=True)
            for img in images:
                # img tuple: (xref, smask, width, height, bpc, colorspace, ...)
                width = img[2]
                height = img[3]
                if width >= 50 and height >= 50:
                    doc.close()
                    return True
        doc.close()
    except Exception as e:
        logger.warning(
            f"  [photo_pdf] Could not check images in "
            f"'{os.path.basename(file_path)}': {e}"
        )
    return False


def detect_photo(file_path: str, source_format: str) -> bool:
    """Dispatch to format-specific photo detection."""
    if source_format in ("pdf", "ocr"):
        return detect_photo_pdf(file_path)
    return False


# ─────────────────────────────────────────────
#  8. LANGUAGE DETECTION
# ─────────────────────────────────────────────


def detect_language(text: str) -> str:
    """
    Detect whether the CV is in Turkish, English, or mixed.

    Strategy:
      1. Simple heuristic: count Turkish stopwords in lowercased text.
      2. If langdetect is available: two-pass (start + end samples).
         Both must agree → that language; else → "mixed".
      3. Fallback: if Turkish words heuristic triggers → "tr", else "en".

    Returns "tr" | "en" | "mixed"
    """
    if not text or len(text.strip()) < 30:
        return "en"

    sample = text[:2000].lower()
    words_in_sample = set(re.findall(r"\b\w+\b", sample))
    turkish_hits = words_in_sample & TURKISH_WORDS

    if LANGDETECT_AVAILABLE:
        try:
            sample_start = text[:1000]
            sample_end = text[-500:]

            lang_start = langdetect_detect(sample_start)
            lang_end = langdetect_detect(sample_end)

            def norm(lang: str) -> str:
                return "tr" if lang == "tr" else "en"

            l1, l2 = norm(lang_start), norm(lang_end)
            return l1 if l1 == l2 else "mixed"

        except Exception:
            pass  # fall through to heuristic

    if len(turkish_hits) >= 3:
        return "tr"
    return "en"


# ─────────────────────────────────────────────
#  9. MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────────


def process_cv(file_path: Path) -> dict:
    """
    Process a single CV file (PDF) and return a structured record.

    Pipeline (in order):
      1.  Determine file format.
      2.  Extract raw text — column-aware, with OCR fallback.
          Two-column pages emit a ``===COLUMN_BREAK===`` sentinel
          between the left and right column text blocks.
      3.  Extract contact info from the ORIGINAL text (before any cleaning)
          to preserve punctuation inside emails, URLs, and phone numbers.
      4.  Normalize column spacing:
          ``normalize_column_spacing()`` collapses excess whitespace and
          ensures correct punctuation spacing while keeping the
          ``===COLUMN_BREAK===`` token and all structured data intact.
      5.  Clean text (lowercase, strip special chars, protect emails/URLs).
      6.  Extract sections using keyword-based heading detection.
      7.  Detect profile photo.
      8.  Detect language.
      9.  Assemble and return the output record.
    """
    file_path_str = str(file_path)
    suffix = file_path.suffix.lower()
    resume_id = str(uuid.uuid4())

    logger.info(f"Processing: {file_path.name}")

    raw_text = ""
    source_format = "failed"

    try:
        if suffix == ".pdf":
            logger.info("  [method] PDF extraction (column-aware)")
            raw_text, source_format = extract_text_pdf(file_path_str)
        else:
            logger.warning(f"  [skip] Unsupported format: {suffix}")
            source_format = "failed"
    except Exception as e:
        logger.error(f"  [critical_error] {file_path.name}: {e}")
        source_format = "failed"
        raw_text = ""

    # ── Step 2b: repair broken email addresses in raw text ───────────────────
    # Must run BEFORE contact extraction so extract_contact_info sees valid
    # emails, and BEFORE normalize_column_spacing which protects them.
    _dbg_early = bool(os.environ.get("PARSER_DEBUG", ""))
    if raw_text:
        raw_text = repair_broken_emails(raw_text, debug=_dbg_early)

    # ── Step 2c: normalize text (FIX 3) ──────────────────────────────────────
    # Applies NFC Unicode normalization, fixes spaces around '@' and '.' in
    # email-like spans, and collapses conservative broken-word artifacts.
    # Runs AFTER repair_broken_emails (so already-repaired emails are not
    # re-processed) and BEFORE extract_contact_info (so contact sees clean text).
    if raw_text:
        raw_text = normalize_text(raw_text)

    # ── Step 3: contact info — from original text, before any mutation ────────
    contact = extract_contact_info(raw_text)

    # ── Step 4: normalize column spacing ─────────────────────────────────────
    # Must run BEFORE clean_text so that the COLUMN_BREAK_TOKEN (which contains
    # only ASCII uppercase letters, digits, and "=") is not stripped by the
    # special-character remover in clean_text.
    if _dbg_early and raw_text:
        sample_raw = raw_text[:400].replace("\n", " ↵ ")
        print(f"[DEBUG] RAW TEXT (pre-normalise): {sample_raw!r}")

    normalised_text = normalize_column_spacing(raw_text) if raw_text else ""

    # ── Step 4b: repair OCR / broken-token spacing artifacts ────────────────
    # Runs AFTER normalize_column_spacing (so COLUMN_BREAK_TOKEN is already
    # present) and BEFORE clean_text (so protected tokens survive lowercasing).
    ocr_fixed_text = fix_ocr_spacing(normalised_text) if normalised_text else ""

    if _dbg_early and ocr_fixed_text:
        sample_norm = ocr_fixed_text[:400].replace("\n", " ↵ ")
        print(f"[DEBUG] TEXT (post-normalise, pre-clean): {sample_norm!r}")

    # ── Step 5: clean (lowercase, strip junk chars, collapse whitespace) ──────
    cleaned_text = clean_text(ocr_fixed_text) if ocr_fixed_text else ""

    # ── Debug: detect character loss between normalised and cleaned text ───────
    if _dbg_early and ocr_fixed_text and cleaned_text:
        _chars_before = set(ocr_fixed_text.lower())
        _chars_after = set(cleaned_text)
        _lost = _chars_before - _chars_after - {"\r"}  # \r is intentionally dropped
        if _lost:
            logger.debug(
                f"  [char_loss] Characters present before clean_text but absent after: "
                f"{sorted(_lost)!r}"
            )

    # ── Step 6: section extraction ────────────────────────────────────────────
    # Pass debug=True when the PARSER_DEBUG env-var is set so that the debug
    # prints include the filename context from this outer scope.
    if _dbg_early:
        print(f"\n[DEBUG] ── Processing: {file_path.name} ────────────────")
    sections_raw = extract_sections(cleaned_text, debug=_dbg_early)

    # Extract confidence scores (internal quality metadata) — stored separately
    # so the public "sections" dict contains only string values, preserving
    # backward compatibility with all downstream consumers.
    section_confidence: dict[str, float] = sections_raw.pop("__confidence__", {})
    sections = sections_raw

    # ── Step 6b: group experience blocks (FIX 4) ─────────────────────────────
    # Merge fragmented experience lines (each job was one line) into structured
    # blocks: "Company - City - Year | Job Title Description".
    if sections.get("experience"):
        sections["experience"] = group_experience_blocks(sections["experience"])

    # ── Step 7: photo detection ───────────────────────────────────────────────
    has_photo = False
    if source_format != "failed":
        has_photo = detect_photo(file_path_str, source_format)

    # ── Step 8: language detection ────────────────────────────────────────────
    language = detect_language(cleaned_text)

    # ── Step 9: assemble record ───────────────────────────────────────────────
    record = {
        "resume_id": resume_id,
        "file_path": file_path_str,
        "raw_text": cleaned_text,
        "sections": sections,
        "section_confidence": section_confidence,
        "contact": contact,
        "has_photo": has_photo,
        "language": language,
        "source_format": source_format,
    }

    logger.info(
        f"  [done] {file_path.name} → "
        f"format={source_format}, lang={language}, "
        f"photo={has_photo}, chars={len(cleaned_text)}"
    )
    return record


# ─────────────────────────────────────────────
#  10. DATASET BUILDER
# ─────────────────────────────────────────────

_FAILED_RECORD_TEMPLATE: dict = {
    "raw_text": "",
    "sections": {
        "summary": "",
        "experience": "",
        "education": "",
        "skills": "",
        "projects": "",
    },
    "contact": {
        "email": "",
        "phone": "",
        "linkedin": "",
        "github": "",
    },
    "has_photo": False,
    "language": "en",
    "source_format": "failed",
}


def build_dataset(
    pdf_dir: str,
    output_path: str = "final_dataset.json",
) -> None:
    """
    Iterate through the CV directory, process every file, and write the
    aggregated results to a JSON file.

    Args:
        pdf_dir:     Path to the directory containing PDF CVs.
        output_path: Destination JSON file path.
    """
    pdf_dir = Path(pdf_dir)

    pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

    total = len(pdf_files)

    if total == 0:
        logger.warning("No CV files found. Check your directory paths.")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) — {total} files total.")

    dataset: list[dict] = []
    failed_files: list[str] = []

    for file_path in tqdm(pdf_files, desc="Parsing CVs", unit="file"):
        try:
            record = process_cv(file_path)
            dataset.append(record)
            if record["source_format"] == "failed":
                failed_files.append(str(file_path))
        except Exception as e:
            logger.error(f"[unhandled] {file_path.name}: {e}")
            failed_files.append(str(file_path))
            dataset.append(
                {
                    "resume_id": str(uuid.uuid4()),
                    "file_path": str(file_path),
                    **_FAILED_RECORD_TEMPLATE,
                }
            )

    # ── Write JSON output ────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # ── Summary report ───────────────────────────────────────────────────────
    success_count = sum(1 for r in dataset if r["source_format"] != "failed")
    ocr_count = sum(1 for r in dataset if r["source_format"] == "ocr")
    two_col_note = "  (check cv_parser.log for column-layout details)"

    logger.info("=" * 60)
    logger.info(f"DONE — {total} files processed")
    logger.info(f"  ✓ Success  : {success_count}")
    logger.info(f"  ✗ Failed   : {len(failed_files)}")
    logger.info(f"  ~ OCR used : {ocr_count}")
    logger.info(f"  Output     : {output_path}")
    logger.info(two_col_note)
    if failed_files:
        logger.warning("Failed files:")
        for fp in failed_files:
            logger.warning(f"    {fp}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CV Parsing Pipeline (column-aware) — produces final_dataset.json"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="C:/Users/rumeysagokce/Desktop/cv_parser_project/data/PDF",
        help="Directory containing PDF CV files (default: cvs/pdf)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_dataset.json",
        help="Output JSON file path (default: final_dataset.json)",
    )
    args = parser.parse_args()

    build_dataset(
        pdf_dir=args.pdf_dir,
        output_path=args.output,
    )
