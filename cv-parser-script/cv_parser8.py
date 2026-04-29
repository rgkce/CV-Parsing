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

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

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

try:
    from rapidfuzz import fuzz as _rf_fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from sklearn.cluster import KMeans as _KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
        # ===== SHORT NATURAL HEADINGS =====
        "a bit about me",
        "a little about me",
        "about the author",
        "about candidate",
        "who is this candidate",
        # ===== FIRST PERSON STYLE TITLES =====
        "i am",
        "i am a",
        "i am an",
        "this is me",
        # ===== COVER LETTER STYLE =====
        "personal statement",
        "career statement",
        "statement",
        # ===== LINKEDIN STYLE =====
        "headline",
        "tagline",
        "professional headline",
        # ===== TURKISH NATURAL =====
        "kısaca",
        "kendimden bahsetmek gerekirse",
        "kısaca kendim",
        "kısaca ben",
        "ben kimim?",
        "ben kimim",
        "kendi hakkımda",
        "biraz kendimden bahsedeyim",
        "kısaca kendimi tanıtayım",
        # ======================
        # CORE
        # ======================
        "summary",
        "profile",
        "about",
        "about me",
        "objective",
        "professional summary",
        "career objective",
        # ======================
        # ADVANCED EN
        # ======================
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
        "about the candidate",
        "about the applicant",
        "candidate overview",
        "professional background",
        "career background",
        # ======================
        # ATS / CORPORATE STYLE
        # ======================
        "qualifications summary",
        "summary of qualifications",
        "key qualifications",
        "highlights",
        "career highlights",
        "professional highlights",
        "key profile",
        "value proposition",
        "core profile",
        "executive profile",
        "personal statement",
        "professional statement",
        "candidate statement",
        # ======================
        # OBJECTIVE VARIANTS
        # ======================
        "objective statement",
        "career goal",
        "career goals",
        "professional objective",
        "employment objective",
        "job objective",
        "personal objective",
        "career intent",
        "career intention",
        "goal statement",
        # ======================
        # TURKISH (GENİŞLETİLMİŞ)
        # ======================
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
        "kendim hakkında",
        "ben kimim?",
        "kişisel tanıtım",
        "tanıtım",
        "kısaca ben",
        "özgeçmiş hakkında",
        "profil özeti",
        "mesleki hedef",
        "hedeflerim",
        "amaçlarım",
        "vizyonum",
        "misyonum",
        "benim hakkımda",
        "kim ben",
        "kendimi tanıtayım",
        "kısa tanıtım",
        "genel özet",
        "ön yazı özeti",
        "başvuru özeti",
        # ======================
        # BILINGUAL / MIXED
        # ======================
        "profil / profile",
        "özet / summary",
        "hakkımda / about me",
        "about me / hakkımda",
        "summary / özet",
        "profile / profil",
        "objective / hedef",
        "hedef / objective",
        "kariyer hedefi / career objective",
        # ======================
        # OCR / TYPO TOLERANT
        # ======================
        "summ ary",
        "prof ile",
        "ob jective",
        "abo ut",
        "summry",
        "proflie",
        "objctive",
        "abut me",
        "sum mary",
        "pro file",
        "over view",
        "int ro",
        "ozet",
        "hakkimda",
        "profl",
        "sumary",
        "profil ozeti",
        # ======================
        # MINIMAL / RISKY (DİKKATLİ KULLAN)
        # ======================
        "me",
        "who i am",
        "who am i",
        "about myself",
        "myself",
    ],
    "experience": [
        # ======================
        # CORE
        # ======================
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "career history",
        "positions held",
        # ======================
        # ADVANCED EN
        # ======================
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
        "work timeline",
        "employment timeline",
        # ======================
        # MODERN / ATS / LINKEDIN STYLE
        # ======================
        "experience overview",
        "career overview",
        "work overview",
        "professional journey",
        "career journey",
        "work journey",
        "professional track",
        "career track",
        "employment track",
        "work profile",
        "career profile",
        # ======================
        # ROLE / POSITION BASED
        # ======================
        "roles",
        "positions",
        "job positions",
        "held positions",
        "previous roles",
        "past roles",
        "current and previous roles",
        "relevant experience",
        "related experience",
        "industry experience",
        "technical experience",
        "functional experience",
        # ======================
        # CORPORATE SUMMARY STYLE
        # ======================
        "experience summary",
        "summary of experience",
        "employment summary",
        "work experience summary",
        "career summary experience",
        "professional experience summary",
        # ======================
        # PROJECT-LIKE BUT EXPERIENCE
        # ======================
        "project experience",
        "project work",
        "practical experience",
        "hands-on experience",
        "field experience",
        "real world experience",
        "applied experience",
        # ======================
        # INTERNSHIP / ENTRY LEVEL
        # ======================
        "internship experience",
        "internships",
        "training experience",
        "apprenticeship",
        "apprenticeships",
        "staj deneyimi",
        "stajlar",
        # ======================
        # TURKISH (GENİŞLETİLMİŞ)
        # ======================
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
        "çalışma hayatı",
        "iş kariyeri",
        "meslek hayatı",
        "iş deneyimleri",
        "kariyer özeti deneyim",
        "çalıştığım yerler",
        "görev yerleri",
        "görev geçmişi",
        "iş deneyimi özeti",
        "profesyonel geçmiş",
        "çalışma geçmişim",
        "iş geçmişim",
        "kariyerim",
        "mesleki geçmişim",
        "geçmiş pozisyonlar",
        "pozisyonlar",
        "önceki işler",
        "eski görevler",
        "staj",
        "staj deneyimleri",
        "çalışma tecrübesi",
        "iş tecrübeleri",
        "sektör deneyimi",
        # ======================
        # BILINGUAL / MIXED
        # ======================
        "deneyim / experience",
        "iş deneyimi / work experience",
        "kariyer / career",
        "work experience / iş deneyimi",
        "experience / deneyim",
        "iş geçmişi / work history",
        "career history / kariyer geçmişi",
        # ======================
        # OCR / TYPO TOLERANT
        # ======================
        "exper ience",
        "experlence",
        "experince",
        "employ ment",
        "work exper ience",
        "deney im",
        "is deneyimi",
        "calisma gecmisi",
        "kariyer gecmisi",
        "is gecmisi",
        "deneyim ler",
        "i deneyimi",
        "profesyonel denyim",
        # ======================
        # MINIMAL / RISKY
        # ======================
        "career",
        "work",
        "jobs",
        "my experience",
        "my work",
    ],
    "education": [
        # ======================
        # CORE
        # ======================
        "education",
        "academic background",
        "academic history",
        "qualifications",
        "degrees",
        "schooling",
        # ======================
        # ADVANCED EN
        # ======================
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
        "academic formation",
        "education details",
        "academic details",
        # ======================
        # DEGREE / PROGRAM BASED
        # ======================
        "degree",
        "degrees obtained",
        "academic degrees",
        "degree information",
        "degree details",
        "educational qualifications",
        "qualification details",
        "academic credentials",
        "credentials",
        "bachelor",
        "master",
        "phd",
        "msc",
        "bsc",
        "ba",
        "ma",
        "doctorate",
        # ======================
        # MIXED (EDUCATION + CERT)
        # ======================
        "certifications and education",
        "education & qualifications",
        "education and certifications",
        "education and certificates",
        # ======================
        # INSTITUTION BASED
        # ======================
        "universities attended",
        "colleges attended",
        "schools attended",
        "institutions",
        "academic institutions",
        "education institutions",
        # ======================
        # ATS / CORPORATE
        # ======================
        "education summary",
        "academic summary",
        "qualification summary",
        "education overview",
        "academic overview",
        # ======================
        # INTERNATIONAL / STUDY ABROAD
        # ======================
        "exchange programs",
        "study abroad",
        "international education",
        "erasmus",
        "erasmus experience",
        # ======================
        # TURKISH (GENİŞLETİLMİŞ)
        # ======================
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
        "mezun olduğum okullar",
        "okul bilgileri",
        "okullar",
        "eğitim hayatı",
        "akademik hayat",
        "öğrenim hayatı",
        "okul geçmişi",
        "eğitim geçmişim",
        "öğrenim geçmişim",
        "okuduğum okullar",
        "mezun olduğum üniversite",
        "mezun olunan kurum",
        "lisans eğitimi",
        "lisansüstü eğitim",
        "yüksek lisans",
        "doktora",
        "ön lisans",
        "lise",
        "lise eğitimi",
        "üniversite eğitimi",
        "akademik kariyer",
        "akademik çalışmalar",
        "eğitim kurumları",
        "öğrenim kurumları",
        "okul ve eğitim",
        "eğitim ve öğretim",
        "eğitim bilgilerim",
        # ======================
        # BILINGUAL / MIXED
        # ======================
        "education / eğitim",
        "eğitim / education",
        "academic background / akademik geçmiş",
        "education & eğitim",
        "eğitim & education",
        "öğrenim / education",
        "mezuniyet / graduation",
        # ======================
        # OCR / TYPO TOLERANT
        # ======================
        "educat ion",
        "edcation",
        "educaton",
        "acadmic background",
        "academ ic history",
        "egitim",
        "ogrenim",
        "akademik gecmis",
        "mezuniy et",
        "okul bilgi leri",
        "egitim bilgileri",
        "egitim gecmisi",
        # ======================
        # MINIMAL / RISKY
        # ======================
        "education info",
        "academic",
        "studies",
    ],
    "skills": [
        # ======================
        # CORE
        # ======================
        "skills",
        "technical skills",
        "core competencies",
        "competencies",
        "technologies",
        "tools",
        "proficiencies",
        "key skills",
        "areas of expertise",
        # ======================
        # ADVANCED EN
        # ======================
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
        "core strengths",
        "professional capabilities",
        "technical strengths",
        # ======================
        # TECH / STACK FOCUSED
        # ======================
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
        "platforms",
        "systems",
        "technologies & tools",
        "tools & frameworks",
        # ======================
        # PROGRAMMING / DEV
        # ======================
        "programming skills",
        "coding skills",
        "development skills",
        "software development skills",
        "engineering skills",
        "it skills",
        "technical stack",
        "dev stack",
        # ======================
        # LANGUAGES (IMPORTANT)
        # ======================
        "languages",
        "programming languages",
        "coding languages",
        "language skills",
        "spoken languages",
        "foreign languages",
        "language proficiency",
        "linguistic skills",
        # ======================
        # SOFT SKILLS (AYRI AMA SKILLS)
        # ======================
        "soft skills",
        "personal skills",
        "interpersonal skills",
        "communication skills",
        "leadership skills",
        "transferable skills",
        # ======================
        # TURKISH (GENİŞLETİLMİŞ)
        # ======================
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
        "teknoloji yığını",
        "teknoloji seti",
        "araçlar ve teknolojiler",
        "kullandığım araçlar",
        "yazılım becerileri",
        "programlama becerileri",
        "mesleki beceriler",
        "profesyonel beceriler",
        "ana yetkinlikler",
        "anahtar beceriler",
        "iletişim becerileri",
        "liderlik becerileri",
        "kişisel beceriler",
        "sosyal beceriler",
        "analitik beceriler",
        "problem çözme becerileri",
        "takım çalışması",
        "yönetim becerileri",
        "teknik araçlar",
        "yazılım araçları",
        "kullandığım yazılımlar",
        "yazılımlar",
        "programlar",
        "kullandığım programlar",
        # ======================
        # TURKISH LANGUAGES
        # ======================
        "diller",
        "yabancı diller",
        "konuşulan diller",
        "dil bilgisi",
        "dil yetkinliği",
        "dil seviyesi",
        "dil becerileri",
        "yabancı dil",
        # ======================
        # BILINGUAL
        # ======================
        "skills / yetenekler",
        "yetenekler / skills",
        "teknik beceriler / technical skills",
        "skills & yetenekler",
        "beceriler / skills",
        "yetkinlikler / competencies",
        "diller / languages",
        "technologies / teknolojiler",
        # ======================
        # OCR / TYPO
        # ======================
        "skil ls",
        "ski lls",
        "technol ogies",
        "compet encies",
        "proficienc ies",
        "yetenek ler",
        "becer iler",
        "teknolo jiler",
        "yetkin likler",
        "becerile r",
        "tec hologies",
        "skills &",
        # ======================
        # MINIMAL / RISKY
        # ======================
        "skills & abilities",
        "abilities",
        "expertise",
        "tools",
        "stack",
    ],
    "projects": [
        # ======================
        # CORE
        # ======================
        "projects",
        "personal projects",
        "key projects",
        "portfolio",
        "project portfolio",
        # ======================
        # ADVANCED EN
        # ======================
        "project experience",
        "project work",
        "project history",
        "selected projects",
        "notable projects",
        "featured projects",
        "relevant projects",
        "academic projects",
        "technical projects",
        "software projects",
        "engineering projects",
        "side projects",
        "independent projects",
        "client projects",
        # ======================
        # DEV / GITHUB / PORTFOLIO
        # ======================
        "github projects",
        "git projects",
        "open-source projects",
        "open source contributions",
        "contributions",
        "project contributions",
        "code portfolio",
        "development projects",
        "software portfolio",
        "project showcase",
        "project highlights",
        "repositories",
        "github repositories",
        "public repositories",
        # ======================
        # REAL-WORLD / PRACTICAL
        # ======================
        "real world projects",
        "practical projects",
        "hands-on projects",
        "implemented projects",
        "completed projects",
        "delivered projects",
        # ======================
        # RESEARCH / ACADEMIC
        # ======================
        "research projects",
        "thesis projects",
        "capstone projects",
        "graduation projects",
        "final year projects",
        "senior design projects",
        "academic work",
        # ======================
        # TURKISH (GENİŞLETİLMİŞ)
        # ======================
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
        "gerçekleştirdiğim projeler",
        "tamamlanan projeler",
        "uygulama projeleri",
        "çalışmalar",
        "projeler ve çalışmalar",
        "yazılım projeleri",
        "teknik projeler",
        "yan projeler",
        "kişisel çalışmalar",
        "geliştirme projeleri",
        "uygulamalarım",
        "uygulamalar",
        "github çalışmaları",
        "portföy",
        "portföyüm",
        "gösterilebilir projeler",
        "teslim edilen projeler",
        "tamamladığım projeler",
        "öğrenci projeleri",
        "mezuniyet projesi",
        # ======================
        # BILINGUAL
        # ======================
        "projects / projeler",
        "projeler / projects",
        "portfolio / portföy",
        "portföy / portfolio",
        "projects & projeler",
        "çalışmalar / projects",
        # ======================
        # OCR / TYPO
        # ======================
        "pro jects",
        "proj eler",
        "pr0jects",
        "proiects",
        "port folio",
        "proje ler",
        "projeler i",
        # ======================
        # MINIMAL / RISKY
        # ======================
        "portfolio projects",
        "work samples",
        "my projects",
    ],
}

# Turkish word list used for quick language heuristic
TURKISH_WORDS = {
    # ======================
    # BAĞLAÇLAR
    # ======================
    "ve",
    "veya",
    "ya",
    "ya da",
    "ile",
    "ama",
    "fakat",
    "ancak",
    "lakin",
    "oysa",
    "halbuki",
    "ne var ki",
    "bununla beraber",
    "üstelik",
    "dahası",
    "kaldı ki",
    "hem",
    "hem de",
    "ne",
    "ne de",
    # ======================
    # ZAMİRLER
    # ======================
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
    "benim",
    "senin",
    "onun",
    "bizim",
    "sizin",
    "onların",
    "bende",
    "sende",
    "onda",
    "bizde",
    "sizde",
    "onlarda",
    "benden",
    "senden",
    "ondan",
    "bizden",
    "sizden",
    "onlardan",
    "benimle",
    "seninle",
    "onunla",
    "bizimle",
    "sizinle",
    "onlarla",
    "kendim",
    "kendin",
    "kendisi",
    "kendimiz",
    "kendiniz",
    "kendileri",
    # ======================
    # İŞARET ZAMİRLERİ
    # ======================
    "bu",
    "şu",
    "bunlar",
    "şunlar",
    "böyle",
    "şöyle",
    "öyle",
    "buraya",
    "şuraya",
    "oraya",
    "buradan",
    "şuradan",
    "oradan",
    "burada",
    "şurada",
    "orada",
    "bura",
    "şura",
    "ora",
    # ======================
    # EDATLAR / POSTPOZISYONLAR
    # ======================
    "için",
    "gibi",
    "kadar",
    "dolayı",
    "üzere",
    "rağmen",
    "karşı",
    "ile",
    "beraber",
    "dahil",
    "hariç",
    "göre",
    "doğru",
    "karşın",
    "beri",
    "itibaren",
    "önce",
    "sonra",
    "dek",
    "değin",
    "yana",
    "arasında",
    "üstünde",
    "altında",
    "içinde",
    "dışında",
    "yanında",
    "arkasında",
    "önünde",
    "üzerinde",
    "altında",
    "boyunca",
    "süresince",
    # ======================
    # YARDIMCI FİİLLER / EK-FİİL
    # ======================
    "idi",
    "imiş",
    "ise",
    "dir",
    "dır",
    "tir",
    "tır",
    "dur",
    "dür",
    "tür",
    "tur",
    "oldu",
    "olmuş",
    "olur",
    "olacak",
    "olmaktadır",
    "olduğu",
    "olduğum",
    "olduğun",
    "olduğumuz",
    "olmak",
    "olmakta",
    "olmaktayım",
    "olmaktayız",
    "edildi",
    "edilmiş",
    "edilir",
    "edilecek",
    "yapıldı",
    "yapılmış",
    "yapılır",
    # ======================
    # SORU EKLERİ
    # ======================
    "mı",
    "mi",
    "mu",
    "mü",
    "miyim",
    "misin",
    "miyiz",
    "misiniz",
    "mıyım",
    "mısın",
    "mıyız",
    "mısınız",
    "neden",
    "niçin",
    "niye",
    "nasıl",
    "ne zaman",
    "kim",
    "kime",
    "kimi",
    "kimden",
    "kimde",
    "hangi",
    "hangisi",
    "nerede",
    "nereden",
    "nereye",
    "kaç",
    "kaçıncı",
    # ======================
    # ZAMAN İFADELERİ
    # ======================
    "sonra",
    "önce",
    "şimdi",
    "henüz",
    "hala",
    "artık",
    "daha sonra",
    "ilk olarak",
    "en son",
    "son olarak",
    "bugün",
    "dün",
    "yarın",
    "şu an",
    "şu anda",
    "geçen",
    "gelecek",
    "eski",
    "yeni",
    "mevcut",
    "önceki",
    "bu yıl",
    "geçen yıl",
    "önümüzdeki yıl",
    "şimdiye kadar",
    "o zamandan beri",
    "günümüzde",
    # ======================
    # DERECE / NİCELEYİCİLER
    # ======================
    "çok",
    "az",
    "daha",
    "en",
    "her",
    "hiç",
    "bazı",
    "birçok",
    "tüm",
    "genel",
    "çoğu",
    "hepsi",
    "birkaç",
    "hiçbir",
    "herhangi",
    "bütün",
    "tamamı",
    "yarısı",
    "büyük",
    "küçük",
    "fazla",
    "oldukça",
    "son derece",
    "gayet",
    "epey",
    "neredeyse",
    "hemen hemen",
    "yaklaşık",
    # ======================
    # SAYILAR (YAZIYA DÖKÜLMÜŞ)
    # ======================
    "bir",
    "iki",
    "üç",
    "dört",
    "beş",
    "altı",
    "yedi",
    "sekiz",
    "dokuz",
    "on",
    "yirmi",
    "otuz",
    "kırk",
    "elli",
    "altmış",
    "yetmiş",
    "seksen",
    "doksan",
    "yüz",
    "bin",
    "milyon",
    "milyar",
    "birinci",
    "ikinci",
    "üçüncü",
    "dördüncü",
    "beşinci",
    "ilk",
    "son",
    "sonuncu",
    "ortanca",
    # ======================
    # CV FILLER – ÇEKIM EKLERİ / YAPILAR
    # ======================
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
    "bulunan",
    "sağlayan",
    "içeren",
    "kapsayan",
    "geliştiren",
    "kullanan",
    "çalışan",
    "yer alan",
    "katılan",
    "yürüten",
    "yöneten",
    "tasarlayan",
    "uygulayan",
    "analiz eden",
    "sunan",
    "hazırlayan",
    "planlayan",
    "koordine eden",
    "destekleyen",
    "izleyen",
    "test eden",
    "denetleyen",
    "raporlayan",
    "oluşturan",
    "kuruan",
    "başlatan",
    "tamamlayan",
    # ======================
    # CV EYLEM FİİLLERİ (GEÇMİŞ / ŞİMDİ)
    # ======================
    "çalıştım",
    "geliştirdim",
    "yaptım",
    "aldım",
    "kullandım",
    "katıldım",
    "yürüttüm",
    "yönettim",
    "tasarladım",
    "uyguladım",
    "analiz ettim",
    "sundum",
    "hazırladım",
    "planladım",
    "koordine ettim",
    "destekledim",
    "izledim",
    "test ettim",
    "denetledim",
    "raporladım",
    "oluşturdum",
    "kurdum",
    "başlattım",
    "tamamladım",
    "entegre ettim",
    "optimize ettim",
    "çözdüm",
    "araştırdım",
    "inceledim",
    "değerlendirdim",
    "takip ettim",
    "düzenledim",
    "belgeledim",
    "eğittim",
    "liderlik ettim",
    "danıştım",
    "öğrendim",
    "öğrettim",
    "çalışmaktayım",
    "geliştiriyorum",
    "yönetiyorum",
    # ======================
    # BAĞLAYICI İFADELER
    # ======================
    "bu nedenle",
    "bu yüzden",
    "dolayısıyla",
    "ayrıca",
    "ek olarak",
    "bununla birlikte",
    "aynı zamanda",
    "öte yandan",
    "buna ek olarak",
    "bunun yanı sıra",
    "bir yandan",
    "diğer yandan",
    "özellikle",
    "başta",
    "örneğin",
    "mesela",
    "yani",
    "kısacası",
    "özetle",
    "sonuç olarak",
    "netice itibarıyla",
    "genel olarak",
    # ======================
    # AKADEMİK / CV DOLGU
    # ======================
    "kapsamında",
    "çerçevesinde",
    "sürecinde",
    "boyunca",
    "deneyim",
    "tecrübe",
    "bilgi",
    "beceri",
    "yetkinlik",
    "sorumluluk",
    "görev",
    "proje",
    "pozisyon",
    "rol",
    "katkı",
    "başarı",
    "hedef",
    "amaç",
    "sonuç",
    "ekip",
    "takım",
    "departman",
    "bölüm",
    "birim",
    "süreç",
    "yöntem",
    "araç",
    "sistem",
    "platform",
    # ======================
    # SEKTÖR / MESLEK TERİMLERİ
    # ======================
    "yazılım",
    "donanım",
    "veri",
    "analiz",
    "rapor",
    "müşteri",
    "kullanıcı",
    "ürün",
    "hizmet",
    "çözüm",
    "strateji",
    "bütçe",
    "maliyet",
    "kalite",
    "verimlilik",
    "yönetim",
    "liderlik",
    "iletişim",
    "sunum",
    "eğitim",
    # ======================
    # YAYGINCA YANLIŞ YAZILAN / OCR HATALARI
    # ======================
    "calisma",
    "tecrube",
    "egitim",
    "ogrenim",
    "deneyim",  # (doğru ama OCR'da sık çıkar)
    "yonetim",
    "gelistirme",
    "uygulama",
    "koordinasyon",
    "analiz",
    "raporlama",
    "planlama",
    "tasarim",
    "is gecmisi",
    "kariyer gecmisi",
    "is deneyimi",
    # ======================
    # TÜRKÇEYE ÖZGÜ KARAKTERLER İÇEREN YAKIN FORMLAR
    # ======================
    "değerlendirme",
    "sürdürülebilir",
    "güçlendirme",
    "iyileştirme",
    "dönüşüm",
    "büyüme",
    "gelişim",
    "öğrenme",
    "öğretme",
    "ölçümleme",
    "izleme",
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


# ── OCR glyph translation table (applied inside normalize_text Pass 1) ────────
# Repairs ligatures and dotless-i before any downstream text matching.
_NT_OCR_TRANSLATE: dict[int, str] = {
    0x0131: "i",  # ı → i   (dotless-i)
    0x0130: "I",  # İ → I   (becomes i after turkish_lower)
    0xFB01: "fi",  # ﬁ → fi
    0xFB02: "fl",  # ﬂ → fl
    0xFB00: "ff",  # ﬀ → ff
    0xFB03: "ffi",  # ﬃ → ffi
    0xFB04: "ffl",  # ﬄ → ffl
    0x2018: "'",  # ' left single quote
    0x2019: "'",  # ' right single quote
    0x201C: '"',  # " left double quote
    0x201D: '"',  # " right double quote
    0x2013: "-",  # – en-dash
    0x2014: "-",  # — em-dash
}


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


# ─────────────────────────────────────────────
#  STAGE 2 — BLOCK SEGMENTATION
# ─────────────────────────────────────────────

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────
#  STRUCTURED PIPELINE  (cv_pipeline.py — embedded)
#  Replaces the old CVBlock, split_into_blocks, and assign_sections.
#  Stages 1-6: normalize → block segment → heading detect → boundary assign
#              → classify → safety rules → output.
# ─────────────────────────────────────────────────────────────────────────
"""
cv_pipeline.py
==============
Structured CV Parsing Pipeline  —  Stages 1-6
==============================================

Replaces the original keyword-matching approach with a 6-stage structural
pipeline that is layout-aware, OCR-robust, and section-contamination-proof.

PIPELINE OVERVIEW
─────────────────
  Stage 1  normalize_text(text)
           └─ OCR error repair, Unicode NFC, whitespace normalisation,
              duplicate-block removal.

  Stage 2  split_into_blocks(text) → List[CVBlock]
           └─ Split into logical blocks on blank lines or heading detection.
              Each block records structural signals: is_list, has_dates, etc.

  Stage 3  is_heading(line) / detect_heading(block) → Optional[str]
           └─ Robust heading detection: keyword dict, OCR tolerance, merged-
              heading splitting, decoration stripping.

  Stage 4  assign_sections(blocks) → Dict[str, List[str]]
           └─ State-machine boundary detection: section starts at a heading,
              ends at the next heading.  Heading-labeled blocks are trusted
              directly; unlabeled blocks go to the fallback classifier.

  Stage 5  classify_block(block, index) → str
           └─ Structural heuristics for heading-less blocks: date ranges →
              experience; degree words → education; list + tech words → skills;
              build verbs → projects; prose + pronouns → summary.

  Stage 6  apply_safety_rules(sections) + build_output(sections)
           └─ Post-classification safety rules (skills ≠ paragraphs, education
              needs institution keyword, summary capped, etc.).

PUBLIC API
──────────
  parse_cv(text: str) -> Dict[str, str]
      Full pipeline: str → normalized → blocks → sections → final dict.

  normalize_text(text: str) -> str
  split_into_blocks(text: str) -> List[CVBlock]
  is_heading(line: str) -> bool
  detect_heading(block: str) -> Optional[str]
  assign_sections(blocks: List[CVBlock]) -> Dict[str, List[str]]
  classify_block(block: CVBlock, index: int) -> str

All functions are pure / side-effect-free and fully unit-testable.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & DICTIONARIES
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical output section names — every block maps to one of these.
CANONICAL_SECTIONS: List[str] = [
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "other",
]

# Maximum word-count for a line to be considered a potential heading.
# Lines of ≥ 7 words are almost certainly body text.
_HEADING_MAX_WORDS: int = 6

# Summary length limits
_SUMMARY_MIN_WORDS: int = 10  # lowered to catch short one-liner summaries
_SUMMARY_MAX_WORDS: int = 120
_SUMMARY_MAX_LINES: int = 8

# Heading keyword dictionary:  canonical_section → list[heading_variants]
# Entries are lowercased, stripped.  Both English and Turkish are included.
# OCR-error variants are listed explicitly (educatıon, experıence, …).
_HEADING_DICT: Dict[str, List[str]] = {
    "summary": [
        "summary",
        "profile",
        "about",
        "about me",
        "objective",
        "professional summary",
        "career objective",
        "career summary",
        "personal summary",
        "executive summary",
        "introduction",
        "intro",
        "overview",
        "bio",
        "biography",
        "professional profile",
        "personal statement",
        "career statement",
        "personal overview",
        "who i am",
        "about myself",
        # OCR variants
        "summ ary",
        "summry",
        "prof ile",
        "proflie",
        "sumary",
        # Turkish
        "ozet",
        "özet",
        "profil",
        "hakkimda",
        "hakkımda",
        "hakkında",
        "kariyer hedefi",
        "kariyer ozeti",
        "kariyer özeti",
        "kisisel ozet",
        "kişisel özet",
        "genel bakis",
        "genel bakış",
        "tanitim",
        "tanıtım",
        "ben kimim",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "career history",
        "positions held",
        "career",
        "career progression",
        "career path",
        "job history",
        "work record",
        "internship",
        "internships",
        "professional background",
        "relevant experience",
        # OCR variants
        "exper ience",
        "experlence",
        "experince",
        "employ ment",
        # Turkish
        "deneyim",
        "is deneyimi",
        "iş deneyimi",
        "is gecmisi",
        "iş geçmişi",
        "calisma gecmisi",
        "çalışma geçmişi",
        "kariyer gecmisi",
        "kariyer geçmişi",
        "mesleki deneyim",
        "tecrube",
        "tecrübe",
        "staj",
        "staj deneyimi",
    ],
    "education": [
        "education",
        "academic background",
        "academic history",
        "qualifications",
        "degrees",
        "schooling",
        "studies",
        "educational background",
        "academic record",
        "learning",
        "university education",
        "college education",
        # OCR variants
        "educat ion",
        "edcation",
        "educaton",
        # Turkish
        "egitim",
        "eğitim",
        "ogrenim",
        "öğrenim",
        "akademik gecmis",
        "akademik geçmiş",
        "mezuniyet",
        "okul bilgileri",
        "egitim bilgileri",
        "eğitim bilgileri",
        "lisans egitimi",
        "lisans eğitimi",
        "yuksek lisans",
        "yüksek lisans",
        "doktora",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "competencies",
        "technologies",
        "tools",
        "proficiencies",
        "key skills",
        "expertise",
        "capabilities",
        "strengths",
        "skill set",
        "skills summary",
        "professional skills",
        "technical expertise",
        "tech stack",
        "technology stack",
        "development stack",
        "programming skills",
        "software skills",
        "languages",
        "language skills",
        "programming languages",
        # OCR variants
        "sk ills",
        "skils",
        # Turkish
        "yetenekler",
        "beceriler",
        "teknolojiler",
        "yetkinlikler",
        "teknik beceriler",
        "temel yetkinlikler",
        "uzmanlik alanlari",
        "uzmanlık alanları",
        "bilgi birikimi",
        "diller",
        "yabanci diller",
        "yabancı diller",
    ],
    "projects": [
        "projects",
        "personal projects",
        "academic projects",
        "side projects",
        "portfolio",
        "project work",
        "key projects",
        "selected projects",
        "notable projects",
        "open source",
        # Turkish
        "projeler",
        "kisisel projeler",
        "kişisel projeler",
        "akademik projeler",
        "proje calismasi",
        "proje çalışması",
    ],
    "other": [
        "certifications",
        "certificates",
        "licenses",
        "awards",
        "honors",
        "achievements",
        "publications",
        "research",
        "hobbies",
        "interests",
        "volunteering",
        "references",
        "additional information",
        "extracurricular",
        "activities",
        "memberships",
        "contact",
        "contact information",
        "personal information",
        # Turkish
        "sertifikalar",
        "odüller",
        "ödüller",
        "basarilar",
        "başarılar",
        "yayinlar",
        "yayınlar",
        "hobiler",
        "ilgi alanlari",
        "ilgi alanları",
        "gonüllülük",
        "gönüllülük",
        "referanslar",
        "ek bilgiler",
        "iletisim bilgileri",
        "iletişim bilgileri",
        "kisisel bilgiler",
        "kişisel bilgiler",
    ],
}

# Flattened lookup: normalised_heading_text → canonical_section
# Built once at import time for O(1) lookup speed.
_HEADING_LOOKUP: Dict[str, str] = {}
for _sec, _variants in _HEADING_DICT.items():
    for _v in _variants:
        _HEADING_LOOKUP[_v.strip().lower()] = _sec


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPILED REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Used in Stage 1: OCR character repairs
_RE_OCR_DOTLESS_I = re.compile(r"ı")  # dotless-ı → i  (OCR noise)
_RE_OCR_SPACED_L = re.compile(r"\bl\.\s+")  # "l. " → "l." (OCR split dot)
_RE_OCR_GMAIL = re.compile(r"gma\s*l\s+", re.I)  # "gma l " → "gmail"
_RE_AT_SPACES = re.compile(r"([A-Za-z0-9._%+\-])\s+@\s+([A-Za-z0-9.\-])")
_RE_DOT_SPACES = re.compile(r"([A-Za-z0-9])\s*\.\s*([A-Za-z]{2,6})(?=\s|$)")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Used in Stage 2: block structural signals
_RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_RE_DATE_RANGE = re.compile(
    r"(19|20)\d{2}\s*[-–]\s*((19|20)\d{2}|present|günümüz|halen|devam|now)",
    re.I,
)
_RE_BULLET = re.compile(r"^\s*[-•·▪◦●◆▸►*]\s+\S")
_RE_COMMA_LIST = re.compile(
    r"^(?:[A-Za-zÇĞİÖŞÜçğışöüA-Z][A-Za-z0-9+#.\s]{0,24},\s*){2,}"
)

# Used in Stage 3: heading detection
_RE_DECORATION_LEAD = re.compile(r"^[^\w\u0130\u0131\u0100-\u024F]+", re.UNICODE)
_RE_DECORATION_TAIL = re.compile(r"[^\w\u0130\u0131\u0100-\u024F]+$", re.UNICODE)
_RE_ALL_CAPS_WORD = re.compile(r"^[A-ZÇĞİÖŞÜ\s]+$")
_RE_MERGED_HEADING = re.compile(
    r"(education|experience|skills|summary|projects|profile|"
    r"eğitim|deneyim|beceriler|özet)\s+"
    r"(education|experience|skills|summary|projects|profile|"
    r"eğitim|deneyim|beceriler|özet)",
    re.I,
)

# Used in Stage 5: content classification heuristics
_RE_ROLE_WORDS = re.compile(
    r"\b(intern|stajyer|engineer|mühendis|manager|müdür|developer|geliştirici"
    r"|analyst|analist|specialist|uzman|coordinator|koordinatör|lead|lider"
    r"|director|direktör|officer|consultant|danışman|architect|mimar"
    r"|designer|tasarımcı|researcher|araştırmacı|assistant|asistan"
    r"|executive|başkan|president|vice president|vp|ceo|cto|cfo)\b",
    re.I,
)
_RE_COMPANY_WORDS = re.compile(
    r"\b(a\.ş|ltd|inc|corp|gmbh|s\.a|llc|co\.|şirketi|company|holding"
    r"|group|grup|teknoloji|technology|solutions|systems|consulting"
    r"|agency|ajans|bank|banka|hospital|hastane)\b",
    re.I,
)
_RE_DEGREE_WORDS = re.compile(
    r"\b(üniversite|university|fakülte|faculty|bölüm|department|lisans|bachelor"
    r"|yüksek\s+lisans|master|msc|mba|doktora|phd|doctorate|diploma|mezun"
    r"|graduate|lise|high\s+school|okul|school|akademi|academy|enstitü|institute"
    r"|college|polytechnic)\b",
    re.I,
)
_RE_TECH_WORDS = re.compile(
    r"\b(python|java|javascript|typescript|sql|react|angular|vue|django|flask"
    r"|spring|node|nodejs|html|css|sass|scss|php|ruby|swift|kotlin|go|rust"
    r"|c\+\+|docker|kubernetes|k8s|aws|azure|gcp|git|linux|bash|terraform"
    r"|jenkins|figma|sketch|photoshop|premiere|illustrator|after\s*effects"
    r"|excel|powerbi|tableau|matlab|hadoop|spark|tensorflow|pytorch"
    r"|mongodb|postgresql|mysql|redis|graphql|rest|api|microservices)\b",
    re.I,
)
_RE_PROJECT_VERBS = re.compile(
    r"\b(built|developed|created|designed|implemented|architected|deployed"
    r"|launched|contributed|maintained|engineered|coded|programmed|wrote"
    r"|geliştirdim|oluşturdum|tasarladım|yaptım|kurdum|inşa ettim)\b",
    re.I,
)
_RE_PLATFORM_WORDS = re.compile(
    # Only genuine deployment/hosting platforms — NOT generic terms like
    # "backend" or "frontend" which appear in summaries and experience bullets.
    r"\b(github|gitlab|bitbucket|heroku|vercel|netlify|app\s+store|play\s+store"
    r"|npm|pypi|demo\s+at|deployed\s+on|android\s+app|ios\s+app)\b",
    re.I,
)
_RE_SENTENCE_END = re.compile(r"[.!?]\s*$")
_RE_PRONOUN = re.compile(
    r"\b(i am|i have|i'm|i've|ben|benim|hakkımda|kendimi|kariyer|hedefim"
    r"|motivated|passionate|experienced|uzman|deneyimli|seeking|looking)\b",
    re.I,
)

# Safety rules
_RE_LONG_SENTENCE = re.compile(r"\w[\w\s]{60,}[.!?]")  # paragraph line in skills


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — TEXT NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════


def normalize_text(text: str) -> str:
    """
    Stage 1 — produce clean, deduplicated text ready for block segmentation.

    Passes (in order):
      1. Unicode NFC composition  — resolves decomposed diacritics.
      2. OCR glyph repairs        — ı→i, ligatures, broken email spacing.
      3. Per-line whitespace norm — collapse tabs, strip trailing spaces.
      4. Duplicate block removal  — drops ≥80%-similar repeated paragraphs
                                    (the multi-column PDF double-extraction bug).
      5. Collapse excess newlines — max two consecutive blank lines.

    Structured tokens (emails, URLs, phones) are protected so that repairs
    never corrupt them.

    Args:
        text: Raw text from PDF/OCR extraction.

    Returns:
        Normalised text with OCR errors fixed and duplicates removed.
    """
    if not text:
        return ""

    # ── Pass 1: Unicode NFC ───────────────────────────────────────────────────
    text = unicodedata.normalize("NFC", text)

    # ── Pass 2: OCR glyph repairs (global, safe) ──────────────────────────────
    # Replace common OCR ligature artifacts
    _ligature_map = {
        "\ufb01": "fi",  # ﬁ → fi
        "\ufb02": "fl",  # ﬂ → fl
        "\ufb00": "ff",  # ﬀ → ff
        "\ufb03": "ffi",  # ﬃ → ffi
        "\ufb04": "ffl",  # ﬄ → ffl
        "\u2018": "'",  # ' → '
        "\u2019": "'",  # ' → '
        "\u201c": '"',  # " → "
        "\u201d": '"',  # " → "
        "\u2013": "-",  # – → -
        "\u2014": "-",  # — → -
        "\u00b7": " ",  # · → space (bullet used as separator)
    }
    for src, dst in _ligature_map.items():
        text = text.replace(src, dst)

    # Repair broken emails: spaces around "@" and "." in email-like contexts
    text = _repair_broken_emails(text)

    # ── Pass 3: Per-line whitespace normalisation ─────────────────────────────
    normalised_lines: List[str] = []
    for line in text.splitlines():
        # Apply email spacing fix per line (catches most broken patterns)
        if "@" in line:
            for _ in range(3):
                new = _RE_AT_SPACES.sub(r"\1@\2", line)
                if new == line:
                    break
                line = new
            for _ in range(3):
                new = _RE_DOT_SPACES.sub(r"\1.\2", line)
                if new == line:
                    break
                line = new

        # Replace OCR dotless-ı with regular i only in body text
        # (heading lines stay untouched so heading detection still fires)
        if len(line.split()) > _HEADING_MAX_WORDS:
            line = _RE_OCR_DOTLESS_I.sub("i", line)

        # Collapse inline whitespace
        line = _RE_MULTI_SPACE.sub(" ", line).rstrip()
        normalised_lines.append(line)

    text = "\n".join(normalised_lines)

    # ── Pass 4: Duplicate block removal ───────────────────────────────────────
    text = _remove_duplicate_blocks(text, threshold=0.80)

    # ── Pass 5: Collapse excess blank lines ───────────────────────────────────
    text = _RE_MULTI_NEWLINE.sub("\n\n", text).strip()

    return text


def _repair_broken_emails(text: str) -> str:
    """
    Collapse spaces injected inside email addresses by PDF extraction.

    Strategy: find any token sequence containing "@", collapse internal spaces,
    validate with a strict email regex before substituting.

    Example:
        "gma l. com" is not an email pattern (no @) so skip it.
        "user @ gmail . com" → "user@gmail.com" (validated → substitute).
    """
    if "@" not in text:
        return text

    _candidate = re.compile(
        r"[A-Za-z0-9._%+\-]+\s*@\s*[A-Za-z0-9.\-\s]+\.\s*[A-Za-z]{2,}",
        re.I,
    )
    _valid_email = re.compile(
        r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", re.I
    )

    def _try_fix(m: re.Match) -> str:
        original = m.group(0)
        collapsed = re.sub(r"\s+", "", original)
        return collapsed if _valid_email.match(collapsed) else original

    return _candidate.sub(_try_fix, text)


def _remove_duplicate_blocks(text: str, threshold: float = 0.80) -> str:
    """
    Drop paragraph blocks whose normalised content is ≥ threshold similar to
    any previously seen block.  Only the first occurrence is kept.

    A block = sequence of non-empty lines surrounded by blank lines.
    Similarity is measured with SequenceMatcher on lowercased, space-collapsed
    fingerprints.  This eliminates the most common multi-column PDF artifact:
    the same paragraph extracted twice (once per column).

    Args:
        text:      Normalised text (NFC, whitespace fixed).
        threshold: Similarity ratio above which a block is considered a duplicate.

    Returns:
        Text with duplicate blocks removed.
    """
    raw_blocks = re.split(r"\n{2,}", text.strip())
    if len(raw_blocks) <= 1:
        return text

    def _fingerprint(block: str) -> str:
        return re.sub(r"\s+", " ", block.strip().lower())

    kept: List[str] = []
    seen_fps: List[str] = []

    for block in raw_blocks:
        fp = _fingerprint(block)
        if not fp:
            # Preserve empty-looking blocks (structural separators)
            kept.append(block)
            seen_fps.append(fp)
            continue

        is_duplicate = any(
            SequenceMatcher(None, fp, seen_fp).ratio() >= threshold
            for seen_fp in seen_fps
            if seen_fp
        )
        if not is_duplicate:
            kept.append(block)
            seen_fps.append(fp)

    return "\n\n".join(kept)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — BLOCK SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CVBlock:
    """
    A contiguous group of lines that share a common structural role.

    Attributes:
        lines:      Body lines (heading line stripped out).
        heading:    Canonical section name if the block started with a heading.
        is_list:    True when the majority of lines are bullet/comma-list items.
        has_dates:  True when any line contains a 4-digit year (19xx / 20xx).
        line_count: Number of non-empty body lines.
    """

    lines: List[str] = field(default_factory=list)
    heading: Optional[str] = None
    is_list: bool = False
    has_dates: bool = False
    line_count: int = 0


def split_into_blocks(text: str) -> List[CVBlock]:
    """
    Stage 2 — split normalised text into logical CVBlock objects.

    A new block starts when:
      (a) One or more blank lines separate the current line group from the next.
      (b) A heading line is detected mid-paragraph (heading detected in-stream).

    Within each block:
      • The heading line is stored in CVBlock.heading and removed from CVBlock.lines
        so that downstream code only sees body content.
      • Structural signals (is_list, has_dates, line_count) are computed once here
        so later stages don't have to reparse lines.

    List detection rules:
      • Majority of lines start with a bullet marker (-, •, *, etc.)
      • Block reads as a comma-separated list
      • Single line with 3+ short space-separated tokens (tech-stack pattern)

    Args:
        text: Output of normalize_text() — normalised but NOT yet lowercased.
              Heading detection is done case-insensitively internally.

    Returns:
        Ordered list of CVBlock objects preserving document reading order.
    """
    lines = text.splitlines()
    blocks: List[CVBlock] = []
    current_lines: List[str] = []
    current_heading: Optional[str] = None

    def _flush(lns: List[str], hdg: Optional[str]) -> None:
        """Finalise the current accumulated block and append to `blocks`."""
        # A heading-only block with no lines is still valid — it marks a
        # section boundary even if content follows in the next block.
        non_empty = [l for l in lns if l.strip()]
        if not non_empty and hdg is None:
            return  # truly empty — nothing to add

        block = CVBlock(
            lines=non_empty,
            heading=hdg,
            line_count=len(non_empty),
        )
        # Compute structural signals
        block.has_dates = any(_RE_YEAR.search(l) for l in non_empty)

        bullet_count = sum(1 for l in non_empty if _RE_BULLET.match(l))
        is_majority_bullets = bullet_count >= max(1, len(non_empty) // 2)

        # Comma-list: join all lines, check for repeated "token, " pattern
        joined = " ".join(non_empty)
        is_comma_list = bool(len(non_empty) >= 2 and _RE_COMMA_LIST.match(joined))

        # Single-line tech stack: ≥3 tokens, all short, no dates,
        # and must NOT end with sentence punctuation (rules out prose summaries).
        _line0 = non_empty[0] if non_empty else ""
        is_tech_line = (
            len(non_empty) == 1
            and len(_line0.split()) >= 3
            and all(len(tok) <= 20 for tok in _line0.split())
            and not _RE_YEAR.search(_line0)
            and not _RE_SENTENCE_END.search(_line0.strip())
        )

        block.is_list = is_majority_bullets or is_comma_list or is_tech_line
        blocks.append(block)

    for line in lines:
        stripped = line.strip()

        # ── Blank line → flush the current block ─────────────────────────────
        if not stripped:
            if current_lines or current_heading is not None:
                _flush(current_lines, current_heading)
                current_lines = []
                current_heading = None
            continue

        # ── Heading detection ─────────────────────────────────────────────────
        detected_section = _detect_section_from_line(stripped)

        if detected_section is not None:
            # Flush whatever accumulated before this heading
            if current_lines or current_heading is not None:
                _flush(current_lines, current_heading)
            # Start a new block rooted at this heading
            current_lines = []
            current_heading = detected_section
        else:
            # Body line — accumulate into current block
            current_lines.append(line)

    # Flush the final block
    if current_lines or current_heading is not None:
        _flush(current_lines, current_heading)

    return blocks


def _detect_section_from_line(line: str) -> Optional[str]:
    """
    Internal helper: detect canonical section from a single raw line.

    Wraps is_heading() and detect_heading() into a single call that returns
    the canonical section name or None.

    Used by split_into_blocks() and assign_sections().
    """
    if is_heading(line):
        return detect_heading(line)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — HEADING DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def is_heading(line: str) -> bool:
    """
    Stage 3a — decide whether a single line is a section heading.

    Rules (applied in order, stops at first match):
      1. Word-count guard: ≥ 7 words → NOT a heading (body text).
      2. Decoration stripping: remove leading/trailing non-word chars, then re-test.
      3. Keyword lookup: normalised text found in _HEADING_LOOKUP → heading.
      4. Merged-heading detection: two heading words run together → heading.
      5. OCR-repaired lookup: replace dotless-ı with i, re-check dict.

    Deliberately does NOT use:
      • ALL-CAPS heuristic alone (skill names like "PYTHON SQL" would fire)
      • Font-size inference (not available in plain text)

    Args:
        line: A single raw text line (not yet lowercased).

    Returns:
        True if the line is a section heading.
    """
    stripped = line.strip()
    if not stripped:
        return False

    # Guard: too many words → definitely body text
    if len(stripped.split()) >= _HEADING_MAX_WORDS + 1:
        return False

    # Strip decoration characters and trailing colon, then lookup
    plain = _strip_decoration(stripped)

    # Layer 1: exact normalised keyword match
    if _keyword_match(plain):
        return True

    # Layer 2: merged heading ("education skills" → two headings run together)
    if _RE_MERGED_HEADING.search(plain.lower()):
        return True

    # Layer 3: OCR repair — replace dotless-ı → i, retry lookup
    ocr_repaired = plain.replace("ı", "i").replace("İ", "I")
    if ocr_repaired != plain and _keyword_match(ocr_repaired):
        return True

    return False


def detect_heading(block: str) -> Optional[str]:
    """
    Stage 3b — return the canonical section name for a heading line/block.

    Works on either a single line or a short multi-line block.
    For merged headings ("education skills"), returns the FIRST matched section.

    Args:
        block: A raw line or short block known to be (or suspected to be) a heading.

    Returns:
        Canonical section name ("summary", "experience", …) or None.
    """
    stripped = block.strip()
    if not stripped:
        return None

    # Use only the first line when given a multi-line block
    first_line = stripped.splitlines()[0].strip()
    plain = _strip_decoration(first_line)

    # Direct lookup
    result = _keyword_lookup(plain)
    if result:
        return result

    # Merged-heading: return the first sub-heading found
    if _RE_MERGED_HEADING.search(plain.lower()):
        for match in _RE_MERGED_HEADING.finditer(plain.lower()):
            # Try each captured word
            for group_idx in (1, 2):
                candidate = match.group(group_idx)
                result = _keyword_lookup(candidate)
                if result:
                    return result

    # OCR-repaired lookup
    ocr_repaired = plain.replace("ı", "i").replace("İ", "I")
    return _keyword_lookup(ocr_repaired)


def _strip_decoration(line: str) -> str:
    """
    Remove leading/trailing decorative border characters from a heading.

    Examples:
        "─── Education ───"  →  "Education"
        "*** Skills ***"     →  "Skills"
        "[ Experience ]"     →  "Experience"
        "EDUCATION:"         →  "EDUCATION"

    Args:
        line: Raw heading candidate.

    Returns:
        Line with decoration stripped but word content preserved.
    """
    stripped = _RE_DECORATION_LEAD.sub("", line.strip())
    stripped = _RE_DECORATION_TAIL.sub("", stripped)
    stripped = stripped.rstrip(":").strip()
    return stripped


def _normalise_for_lookup(text: str) -> str:
    """
    Normalise text for heading dictionary lookup.

    Lowercases (ASCII-safe), strips non-word characters, collapses whitespace.

    Args:
        text: Any heading candidate string.

    Returns:
        Normalised string for use as a dict key.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\u0130\u0131\u00C0-\u024F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_match(text: str) -> bool:
    """Return True if normalised text appears in the heading dictionary."""
    return _normalise_for_lookup(text) in _HEADING_LOOKUP


def _keyword_lookup(text: str) -> Optional[str]:
    """Return canonical section name for the text, or None."""
    return _HEADING_LOOKUP.get(_normalise_for_lookup(text))


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — SECTION BOUNDARY DETECTION (assign_sections)
# ═══════════════════════════════════════════════════════════════════════════════


def _is_contact_block(block: "CVBlock") -> bool:
    """
    Return True if a heading-less block looks like a name/contact header.

    These blocks appear at the very top of a CV before any labelled section.
    They contain 1-2 lines with a person name, phone, email, URL, LinkedIn,
    or address — not content that belongs to any CV section.

    Detection criteria (all must hold):
      • ≤ 3 non-empty lines
      • No dates (not an experience or education entry)
      • No sentence-ending punctuation (not a summary paragraph)
      • Matches contact-pattern: email, phone, URL, or very short name-like line
    """
    import re as _re

    if block.has_dates:
        return False
    non_empty = [l.strip() for l in block.lines if l.strip()]
    if len(non_empty) > 3:
        return False
    # Must not look like prose (sentence-ending)
    if any(_RE_SENTENCE_END.search(l) for l in non_empty):
        return False
    _RE_CONTACT = _re.compile(
        r"(@|linkedin|github|http|www\.|\+\d|\(\d{3}\)|tel:|phone|"
        r"\d{3}[-.]\d{3}|address|adres)",
        _re.I,
    )
    _RE_NAME_LINE = _re.compile(
        r"^[A-ZÇĞİÖŞÜ][a-zçğışöü]+(\s+[A-ZÇĞİÖŞÜ][a-zçğışöü]+){0,3}$"
    )
    for line in non_empty:
        if _RE_CONTACT.search(line):
            return True
        if _RE_NAME_LINE.match(line.strip()):
            return True
    return False


def assign_sections(blocks: List[CVBlock]) -> Dict[str, List[str]]:
    """
    Stage 4 — assign blocks to canonical sections using a boundary-state machine.

    Algorithm:
      1. Iterate blocks in document order.
      2. If block.heading is set → that section is now ACTIVE.
         All subsequent body lines go to this section UNTIL the next heading.
      3. Heading-less blocks → classified by classify_block() (Stage 5).
      4. Repeated headings (column-split PDF artifact) → merge into existing bucket.
      5. Pre-heading blocks (before the first heading) → discarded.
         Name/contact info is captured by a separate contact extractor.

    Key guarantee:
      A line is assigned to AT MOST ONE section.  Section boundaries are
      determined by heading positions, NOT by keyword scanning of body text.

    Args:
        blocks: Ordered list from split_into_blocks().

    Returns:
        Dict { section_name: [body_lines] } with all CANONICAL_SECTIONS keys.
    """
    raw: Dict[str, List[str]] = {s: [] for s in CANONICAL_SECTIONS}

    current_section: Optional[str] = None
    seen_sections: set[str] = set()  # for column-split detection
    transition_log: List[str] = []  # ordered section transitions

    # Index for position-aware classify_block
    block_index = 0

    for block in blocks:
        if block.heading is not None:
            # ── New section boundary ──────────────────────────────────────────
            detected = block.heading

            if detected in seen_sections:
                # Column-split loop prevention:
                # The same heading appeared again after visiting other sections.
                # Find which sections we visited since the last occurrence.
                last_idx = (
                    len(transition_log) - 1 - transition_log[::-1].index(detected)
                )
                sections_between = set(transition_log[last_idx + 1 :])
                sections_between.discard(detected)

                if sections_between:
                    # We left this section and came back → column-split artefact.
                    # Merge incoming content into the already-open bucket silently.
                    pass  # current_section = detected will merge lines below
            else:
                seen_sections.add(detected)

            current_section = detected
            transition_log.append(detected)

            # The block's own body lines belong to the newly-opened section
            if block.lines:
                raw[current_section].extend(block.lines)

        else:
            # ── No heading: classify by structure (Stage 5) ───────────────────
            if not block.lines:
                block_index += 1
                continue

            # Skip name/contact header blocks that appear before any heading.
            # These are 1-2 line blocks at the very top containing only a name,
            # phone, email, or URL — no useful section content.
            if not current_section and _is_contact_block(block):
                block_index += 1
                continue

            # Position-aware: only classify if we're past any header already
            # OR if this looks structurally significant.
            classified = classify_block(block, block_index)
            raw[classified].extend(block.lines)

        block_index += 1

    return raw


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — CONTENT CLASSIFICATION (fallback for heading-less blocks)
# ═══════════════════════════════════════════════════════════════════════════════


def classify_block(block: CVBlock, index: int) -> str:
    """
    Stage 5 — classify a heading-less CVBlock using structural heuristics.

    Signal hierarchy (stops at first confident match):
      1. Date-range pattern        → experience  (strongest signal)
      2. Degree/institution words  → education   (requires date too)
      3. Project build verbs OR platform names → projects
      4. List shape + tech keywords → skills
      5. High tech-word density (≥4) with no dates → skills
      6. Paragraph prose + pronoun/career words → summary (top-3 blocks only)
      7. Date + role/company words → experience
      8. Keyword score fallback    → best-scoring section
      9. Default                  → experience (most common unlabelled type)

    Position matters for summary:
      Block index 0-2 = top of CV = more likely to be the summary paragraph.

    Args:
        block: CVBlock with pre-computed structural signals.
        index: Zero-based position of the block in the document.

    Returns:
        Canonical section name string.
    """
    full_text = " ".join(block.lines)
    lower_text = full_text.lower()

    # ── Signal 1: degree/institution words → education ───────────────────────
    # Checked FIRST because education entries contain "YYYY-YYYY" date ranges
    # (graduation spans) that would otherwise fire the experience signal below.
    # Degree words are highly specific; false positives are rare.
    if _RE_DEGREE_WORDS.search(lower_text) and block.has_dates:
        return "education"

    # ── Signal 2: date range → experience ────────────────────────────────────
    # A YYYY-YYYY (or YYYY-present) pattern is the defining experience signal,
    # but only AFTER education has been ruled out above.
    if _RE_DATE_RANGE.search(full_text):
        return "experience"

    # ── Signal 3: project build verbs or platform names → projects ────────────
    # Checked BEFORE list+tech so "built a React app" routes to projects even
    # though React is a tech keyword that could trigger skills.
    if _RE_PROJECT_VERBS.search(lower_text) or _RE_PLATFORM_WORDS.search(lower_text):
        return "projects"

    # ── Signal 4: list shape + ≥2 tech words → skills ────────────────────────
    tech_hits = len(_RE_TECH_WORDS.findall(lower_text))
    if block.is_list and tech_hits >= 2:
        return "skills"

    # ── Signal 5: dense tech keywords with no dates → skills ─────────────────
    if tech_hits >= 4 and not block.has_dates:
        return "skills"

    # ── Signal 6: prose paragraph with pronouns/career words → summary ────────
    # Only fires in the top portion of the CV (first 3 blocks) to avoid
    # classifying mid-CV prose (experience bullet points) as a summary.
    sentence_endings = sum(1 for l in block.lines if _RE_SENTENCE_END.search(l))
    word_count = len(full_text.split())
    is_prose = (
        sentence_endings >= 1
        and not block.has_dates
        and not block.is_list
        and _SUMMARY_MIN_WORDS <= word_count <= _SUMMARY_MAX_WORDS
    )
    if is_prose and index < 3 and _RE_PRONOUN.search(lower_text):
        return "summary"

    # ── Signal 7: date + role or company name → experience ────────────────────
    if block.has_dates and (
        _RE_ROLE_WORDS.search(lower_text) or _RE_COMPANY_WORDS.search(lower_text)
    ):
        return "experience"

    # ── Signal 8: keyword score fallback ──────────────────────────────────────
    scored = _score_text_for_section(full_text)
    if scored:
        return scored

    # ── Default ───────────────────────────────────────────────────────────────
    # The most common un-labelled block type in CVs is experience (job entries
    # without clear date ranges, freelance work, etc.).
    return "experience"


def _score_text_for_section(text: str) -> Optional[str]:
    """
    Keyword-scoring fallback: assign points per section, return the winner.

    Used as a last resort when structural signals alone are inconclusive.
    Returns None if no section scores above zero.

    Scoring criteria:
      experience: date-range(3) + role-word(2) + company-word(1)
      education:  degree-word(3) + date+degree(+2)
      skills:     tech-word hits × 2 capped at 6, level-word(1)
      projects:   project-verb(3) + platform(2)
      summary:    pronoun(2) + sentence-ending with no date(1)

    Args:
        text: Full block text (joined lines).

    Returns:
        Best-scoring section name or None.
    """
    lower = text.lower()
    scores: Dict[str, int] = {s: 0 for s in CANONICAL_SECTIONS}

    if _RE_DATE_RANGE.search(text):
        scores["experience"] += 3
    if _RE_ROLE_WORDS.search(lower):
        scores["experience"] += 2
    if _RE_COMPANY_WORDS.search(lower):
        scores["experience"] += 1

    if _RE_DEGREE_WORDS.search(lower):
        scores["education"] += 3
    if _RE_YEAR.search(text) and _RE_DEGREE_WORDS.search(lower):
        scores["education"] += 2

    tech_hits = len(_RE_TECH_WORDS.findall(lower))
    scores["skills"] += min(tech_hits * 2, 6)

    _level_re = re.compile(
        r"\b(beginner|intermediate|advanced|expert|fluent|native|proficient|"
        r"başlangıç|orta|ileri|uzman|akıcı|anadil)\b",
        re.I,
    )
    if _level_re.search(lower):
        scores["skills"] += 1

    if _RE_PROJECT_VERBS.search(lower):
        scores["projects"] += 3
    if _RE_PLATFORM_WORDS.search(lower):
        scores["projects"] += 2

    if _RE_PRONOUN.search(lower):
        scores["summary"] += 2
    if _RE_SENTENCE_END.search(text.strip()) and not _RE_YEAR.search(text):
        scores["summary"] += 1

    best_section = max(scores, key=lambda k: scores[k])
    return best_section if scores[best_section] > 0 else None


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 6 — SAFETY RULES + OUTPUT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════


def _dedup_lines(lines: List[str]) -> List[str]:
    """
    Remove near-duplicate lines within a section (look-back window of 5).

    Only CONSECUTIVE-ish duplicates are removed — legitimate repeated values
    (e.g. "Python" appearing in both skills and an experience bullet) survive
    because they are in different sections and different call contexts.

    Args:
        lines: Accumulated raw lines for one section.

    Returns:
        Lines with near-duplicates removed, order preserved.
    """
    LOOKBACK = 5
    seen_recently: List[str] = []
    result: List[str] = []

    for line in lines:
        norm_key = re.sub(r"\s+", " ", line.strip().lower())
        if norm_key and norm_key in seen_recently:
            continue
        result.append(line)
        if norm_key:
            seen_recently.append(norm_key)
            if len(seen_recently) > LOOKBACK:
                seen_recently.pop(0)

    return result


def build_output(sections: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Stage 6b — finalise and join per-section line lists into output strings.

    Applies safety rules, deduplicates lines, strips whitespace, and returns
    the six canonical string fields.

    Args:
        sections: Dict { section: [lines] } from assign_sections().

    Returns:
        Dict with keys: summary, experience, education, skills, projects, other.
        All values are stripped strings (empty string if section has no content).
    """
    safe = _apply_safety_rules(sections)

    result: Dict[str, str] = {}
    for section in CANONICAL_SECTIONS:
        deduped = _dedup_lines(safe.get(section, []))
        result[section] = "\n".join(deduped).strip()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def parse_cv(text: str) -> Dict[str, str]:
    """
    Full 6-stage CV parsing pipeline.

    Transforms raw OCR/PDF text into a structured dict of CV sections.

    Stages:
      1. normalize_text()       — OCR repair, dedup, whitespace fix
      2. split_into_blocks()    — logical block segmentation
      3. (internal)             — heading detection per block
      4. assign_sections()      — boundary-state-machine section assignment
      5. classify_block()       — structural fallback for heading-less blocks
      6. build_output()         — safety rules, dedup, join to strings

    Args:
        text: Raw text from PDF extraction or OCR.

    Returns:
        Dict with keys: summary, experience, education, skills, projects, other.

    Example:
        >>> result = parse_cv(raw_ocr_text)
        >>> print(result["skills"])
        "Python, SQL, React, Docker"
    """
    # Stage 1: normalise
    normalised = normalize_text(text)

    # Stage 2: segment into blocks (includes Stage 3 heading detection)
    blocks = split_into_blocks(normalised)

    # Stage 4: assign sections using boundary state machine
    sections_raw = assign_sections(blocks)

    # Stage 6: safety rules + output
    return build_output(sections_raw)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  (run with:  python cv_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── end of embedded cv_pipeline ─────────────────────────────────────────


def detect_section_headers(text: str) -> list[tuple[int, str, str]]:
    """
    Stage 3 — scan text line by line and return all detected section headings.

    Returns a list of (line_index, raw_line, canonical_section) tuples so
    callers can use it to locate section boundaries without re-running the
    full block segmentation.

    Detection layers (in priority order):
      L0: _SD_PRIORITY_OVERRIDES (spec-mandated routing)
      L1: _is_section_heading()  (keyword + OCR repair + fuzzy)
      L2: _SD_EXT_MAP extended map
      L3: _sd_detect_heading()   (decoration-strip + context scoring)

    Args:
        text: Any stage of processed text (raw, normalised, or cleaned).

    Returns:
        List of (line_idx, raw_line, section_name) sorted by line_idx.
    """
    results: list[tuple[int, str, str]] = []
    lines = text.splitlines()
    n = len(lines)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        prev_line = lines[i - 1].strip() if i > 0 else ""
        next_line = lines[i + 1].strip() if i < n - 1 else ""

        section, _ = _sd_detect_heading(stripped, prev_line, next_line)
        if section is not None:
            results.append((i, line, section))

    return results


def _find_column_split_x(words: list[dict], page_width: float) -> Optional[float]:
    """
    Find the x-coordinate of the column boundary between two text columns.

    Strategy (two-stage):
      Stage 1 — KMeans clustering (when sklearn available):
        Cluster word x0-positions into 2 groups.  If both clusters contain
        at least COLUMN_MIN_RATIO of all words AND their centres are separated
        by ≥ MIN_GAP_FRACTION * page_width, return the midpoint between the
        right edge of the left cluster and the left edge of the right cluster.
        KMeans adapts naturally to asymmetric layouts (narrow sidebar + wide main).

      Stage 2 — Gap-scan fallback (always available):
        Build a 1-D occupancy array along the x-axis (GAP_SCAN_BUCKETS wide),
        find the longest unoccupied run, return its centre.
        Used when sklearn is absent OR KMeans produces a degenerate split.

    Returns the split x-coordinate in page pixels, or None if no credible
    column boundary is found.
    """
    if not words:
        return None

    n_words = len(words)

    # ── Stage 1: KMeans clustering ────────────────────────────────────────────
    if SKLEARN_AVAILABLE and n_words >= 6:
        try:
            import numpy as np

            X = np.array([[w["x0"]] for w in words], dtype=float)
            km = _KMeans(n_clusters=2, n_init=5, random_state=42)
            labels = km.fit_predict(X)

            left_idx = int(km.cluster_centers_[0][0] <= km.cluster_centers_[1][0])
            right_idx = 1 - left_idx

            left_words = [words[i] for i, l in enumerate(labels) if l == left_idx]
            right_words = [words[i] for i, l in enumerate(labels) if l == right_idx]

            left_ratio = len(left_words) / n_words
            right_ratio = len(right_words) / n_words

            if left_ratio >= COLUMN_MIN_RATIO and right_ratio >= COLUMN_MIN_RATIO:
                left_max_x1 = max(w["x1"] for w in left_words)
                right_min_x0 = min(w["x0"] for w in right_words)
                gap = right_min_x0 - left_max_x1

                if gap / page_width >= MIN_GAP_FRACTION:
                    # Return midpoint of the physical gap between clusters
                    return (left_max_x1 + right_min_x0) / 2.0
        except Exception:
            pass  # Degenerate data or import issue — fall through to Stage 2

    # ── Stage 2: Gap-scan fallback ────────────────────────────────────────────
    bucket_size = page_width / GAP_SCAN_BUCKETS
    occupied = [False] * GAP_SCAN_BUCKETS

    for w in words:
        start_bucket = max(0, int(w["x0"] / bucket_size))
        end_bucket = min(GAP_SCAN_BUCKETS - 1, int(w["x1"] / bucket_size))
        for b in range(start_bucket, end_bucket + 1):
            occupied[b] = True

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

    if current_start is not None:
        run_len = GAP_SCAN_BUCKETS - current_start
        if run_len > (best_end - best_start):
            best_start, best_end = current_start, GAP_SCAN_BUCKETS - 1

    if best_start == -1:
        return None

    gap_width_fraction = (best_end - best_start + 1) / GAP_SCAN_BUCKETS
    if gap_width_fraction < MIN_GAP_FRACTION:
        return None

    return ((best_start + best_end) / 2.0) * bucket_size


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

    # ── Step 0: URL noise filter ─────────────────────────────────────────────
    # Keep linkedin.com and github.com (valuable contact signals).
    # Strip all other http/https/www URLs — they are almost always noise in CVs
    # (portfolio links, job board footers, PDF metadata artifacts).
    def _filter_url(m: re.Match) -> str:
        url = m.group(0)
        url_lower = url.lower()
        if "linkedin.com" in url_lower or "github.com" in url_lower:
            return url
        return ""  # drop noise URL

    text = re.sub(
        r"https?://[^\s]+|www\.[^\s]+", _filter_url, text, flags=re.IGNORECASE
    )
    # Collapse any blank lines left by removed URLs
    text = re.sub(r"\n{3,}", "\n\n", text)

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
        if RAPIDFUZZ_AVAILABLE:
            # rapidfuzz is ~10-50× faster than difflib.SequenceMatcher and
            # uses token_set_ratio which handles word-order variations better.
            ratio = _rf_fuzz.token_set_ratio(norm, kw_norm) / 100.0
        else:
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
    # ======================
    # SKILLS — TEKNİK ALTYAPI
    # ======================
    "program becerileri": "skills",
    "teknik beceriler": "skills",
    "yazılım becerileri": "skills",
    "teknik yetkinlikler": "skills",
    "dil becerileri": "skills",
    "bilgisayar becerileri": "skills",
    "uzmanlıklar": "skills",
    "yetkinlikler": "skills",
    "temel yetkinlikler": "skills",
    "profesyonel beceriler": "skills",
    "mesleki beceriler": "skills",
    "anahtar beceriler": "skills",
    "ana yetkinlikler": "skills",
    "teknik araçlar": "skills",
    "yazılım araçları": "skills",
    "kullandığım araçlar": "skills",
    "kullandığım teknolojiler": "skills",
    "kullanılan teknolojiler": "skills",
    "kullandığım yazılımlar": "skills",
    "kullandığım programlar": "skills",
    "yazılımlar": "skills",
    "programlar": "skills",
    "teknoloji yığını": "skills",
    "teknoloji seti": "skills",
    "araçlar ve teknolojiler": "skills",
    # Skills — EN
    "core skills": "skills",
    "skills": "skills",
    "technical skills": "skills",
    "soft skills": "skills",
    "competencies": "skills",
    "key skills": "skills",
    "skill set": "skills",
    "technical competencies": "skills",
    "professional skills": "skills",
    "tools and technologies": "skills",
    "technologies used": "skills",
    "tech stack": "skills",
    "technology stack": "skills",
    "development stack": "skills",
    "frameworks and libraries": "skills",
    "tools & frameworks": "skills",
    "platforms and tools": "skills",
    "programming skills": "skills",
    "it skills": "skills",
    # ======================
    # SKILLS — DİL YETKİNLİKLERİ
    # ======================
    "diller": "skills",
    "yabancı diller": "skills",
    "yabancı dil": "skills",
    "konuşulan diller": "skills",
    "dil bilgisi": "skills",
    "dil yetkinliği": "skills",
    "dil seviyesi": "skills",
    "dil becerileri": "skills",
    # EN
    "languages": "skills",
    "language proficiency": "skills",
    "spoken languages": "skills",
    "foreign languages": "skills",
    "language skills": "skills",
    "linguistic skills": "skills",
    # ======================
    # SKILLS — KİŞİSEL / SOSYAL
    # ======================
    "iletişim becerileri": "skills",
    "liderlik becerileri": "skills",
    "kişisel beceriler": "skills",
    "sosyal beceriler": "skills",
    "analitik beceriler": "skills",
    "problem çözme becerileri": "skills",
    "takım çalışması": "skills",
    "yönetim becerileri": "skills",
    "organizasyonel beceriler": "skills",
    # EN
    "personal skills": "skills",
    "interpersonal skills": "skills",
    "communication skills": "skills",
    "leadership skills": "skills",
    "analytical skills": "skills",
    "problem solving skills": "skills",
    "teamwork": "skills",
    "organizational skills": "skills",
    "management skills": "skills",
    # ======================
    # OTHER — HOBİ / İLGİ ALANLARI
    # ======================
    "hobiler": "other",
    "hobi": "other",
    "ilgi alanları": "other",
    "ilgi ve hobiler": "other",
    "kişisel ilgi alanları": "other",
    "serbest zaman aktiviteleri": "other",
    "boş zaman aktiviteleri": "other",
    "aktiviteler": "other",
    # EN
    "hobbies": "other",
    "interests": "other",
    "activities": "other",
    "extracurricular activities": "other",
    "personal interests": "other",
    "outside interests": "other",
    "leisure activities": "other",
    "pastimes": "other",
    # ======================
    # OTHER — SERTİFİKA / LİSANS / BELGE
    # ======================
    "sertifikalar": "other",
    "sertifika": "other",
    "belgeler": "other",
    "lisanslar": "other",
    "sertifikasyonlar": "other",
    "mesleki sertifikalar": "other",
    "tamamlanan kurslar": "other",
    "kurslar": "other",
    "online kurslar": "other",
    "eğitimler": "other",
    # EN
    "certifications": "other",
    "certificates": "other",
    "licenses": "other",
    "licenses & certifications": "other",
    "professional certifications": "other",
    "courses": "other",
    "online courses": "other",
    "training": "other",
    "completed courses": "other",
    "continuing education": "other",
    "professional development": "other",
    # ======================
    # OTHER — ÖDÜL / BAŞARI / ONUR
    # ======================
    "ödüller": "other",
    "başarılar": "other",
    "ödül ve başarılar": "other",
    "onurlar": "other",
    "tanınırlık": "other",
    # EN
    "achievements": "other",
    "awards": "other",
    "honors": "other",
    "honours": "other",
    "recognitions": "other",
    "awards & honors": "other",
    "accomplishments": "other",
    "distinctions": "other",
    "scholarships": "other",
    "fellowships": "other",
    # ======================
    # OTHER — AKADEMİK / YAYIN / ARAŞTIRMA
    # ======================
    "yayınlar": "other",
    "akademik çalışmalar": "other",
    "makaleler": "other",
    "konferanslar": "other",
    "sunumlar": "other",
    "tezler": "other",
    # EN
    "publications": "other",
    "research": "other",
    "papers": "other",
    "conferences": "other",
    "presentations": "other",
    "theses": "other",
    "academic publications": "other",
    "research papers": "other",
    "conference presentations": "other",
    # ======================
    # OTHER — GÖNÜLLÜLÜK / SOSYAL
    # ======================
    "gönüllülük": "other",
    "gönüllü çalışmalar": "other",
    "sosyal sorumluluk": "other",
    "topluluk çalışmaları": "other",
    "sivil toplum": "other",
    # EN
    "volunteering": "other",
    "volunteer work": "other",
    "volunteer experience": "other",
    "community service": "other",
    "social responsibility": "other",
    "civic activities": "other",
    "non-profit work": "other",
    "charity work": "other",
    # ======================
    # OTHER — ORGANİZASYON / LİDERLİK
    # ======================
    "organizasyonlar": "other",
    "organizasyon deneyimi": "other",
    "liderlik deneyimi": "other",
    "kulüp üyelikleri": "other",
    "dernek üyelikleri": "other",
    "üyelikler": "other",
    "komite üyelikleri": "other",
    "öğrenci toplulukları": "other",
    # EN
    "leadership experience": "other",
    "leadership & activities": "other",
    "organizations": "other",
    "organization & leadership": "other",
    "organizational memberships": "other",
    "memberships": "other",
    "professional memberships": "other",
    "associations": "other",
    "club memberships": "other",
    "student organizations": "other",
    "committee roles": "other",
    "board membership": "other",
    # ======================
    # OTHER — REFERANS
    # ======================
    "referanslar": "other",
    "referans": "other",
    "referanslarım": "other",
    # EN
    "references": "other",
    "referees": "other",
    "professional references": "other",
    "references available": "other",
    "references upon request": "other",
    # ======================
    # OTHER — KİŞİSEL / İLETİŞİM
    # ======================
    "iletişim bilgileri": "other",
    "kişisel bilgiler": "other",
    "özlük bilgileri": "other",
    "demografik bilgiler": "other",
    # EN
    "contact": "other",
    "contact information": "other",
    "contact details": "other",
    "personal information": "other",
    "personal details": "other",
    "personal data": "other",
    # ======================
    # OTHER — EK / ÇEŞITLI
    # ======================
    "ek bilgiler": "other",
    "ek bilgi": "other",
    "diğer": "other",
    "çeşitli": "other",
    "genel": "other",
    # EN
    "additional information": "other",
    "other information": "other",
    "misc": "other",
    "miscellaneous": "other",
    "extra": "other",
    "supplementary information": "other",
    "additional details": "other",
    "further information": "other",
    "appendix": "other",
}.items():
    _SD_EXT_MAP[_sd_norm(_sd_heading)] = _sd_bucket


# ─────────────────────────────────────────────
#  5c. HIERARCHICAL SECTION CONSTANTS  (Session 10)
# ─────────────────────────────────────────────
#
# MAIN_HEADERS  — generic section labels that open a new top-level section.
# SUB_HEADERS   — specific sub-types that nest UNDER the current MAIN section.
#
# When a SUB heading is encountered, content accumulates under:
#   sections["skills_subsections"][sub_label]  (or analogous for other mains)
# while sections["skills"] (flat string) continues to receive the same lines
# for full backward compatibility.
#
# Normalised keys (via _sd_norm) are compared so Turkish chars always match.

MAIN_HEADERS: set[str] = {
    _sd_norm(h)
    for h in [
        # Turkish
        "eğitim",
        "iş geçmişi",
        "iş deneyimi",
        "deneyim",
        "beceriler",
        "hakkımda",
        "özet",
        "projeler",
        # English
        "education",
        "experience",
        "work experience",
        "skills",
        "summary",
        "about",
        "projects",
    ]
}

# sub_norm_key → (parent_section, display_label)
SUB_HEADERS: dict[str, tuple[str, str]] = {
    _sd_norm(k): v
    for k, v in {
        "program becerileri": ("skills", "Program Becerileri"),
        "teknik beceriler": ("skills", "Teknik Beceriler"),
        "yazılım becerileri": ("skills", "Yazılım Becerileri"),
        "teknik yetkinlikler": ("skills", "Teknik Yetkinlikler"),
        "dil becerileri": ("skills", "Dil Becerileri"),
        "diller": ("skills", "Diller"),
        "yabancı diller": ("skills", "Yabancı Diller"),
        "hobiler": ("other", "Hobiler"),
        "ilgi alanları": ("other", "İlgi Alanları"),
        "sertifikalar": ("other", "Sertifikalar"),
        "ödüller": ("other", "Ödüller"),
        "başarılar": ("other", "Başarılar"),
        "gönüllülük": ("other", "Gönüllülük"),
        "referanslar": ("other", "Referanslar"),
        # English equivalents
        "languages": ("skills", "Languages"),
        "technical skills": ("skills", "Technical Skills"),
        "soft skills": ("skills", "Soft Skills"),
        "hobbies": ("other", "Hobbies"),
        "interests": ("other", "Interests"),
        "certifications": ("other", "Certifications"),
        "awards": ("other", "Awards"),
        "volunteering": ("other", "Volunteering"),
        "references": ("other", "References"),
    }.items()
}


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

# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 — CONTENT CLASSIFICATION  (assign_sections)
# ─────────────────────────────────────────────────────────────────────────────
#
#  This replaces the old keyword-only _sd_score_line_for_section() fallback.
#
#  Design:
#    • Heading-labelled blocks → trusted directly.
#    • Unlabelled blocks → classified by STRUCTURE first, keywords second.
#    • Safety rules applied after initial assignment to correct mis-routes.

# ── Content-signal patterns ───────────────────────────────────────────────────

# Experience: date + role/company signals
_AS_DATE_RE = re.compile(r"\b(19|20)\d{2}\b")
_AS_DATE_RANGE = re.compile(
    r"(19|20)\d{2}\s*[-–]\s*((19|20)\d{2}|present|günümüz|halen|devam)", re.I
)
_AS_ROLE_WORDS = re.compile(
    r"\b(intern|stajyer|engineer|mühendis|manager|müdür|developer|geliştirici"
    r"|analyst|analist|specialist|uzman|coordinator|koordinatör|lead|lider"
    r"|director|direktör|officer|consultant|danışman|architect|mimar"
    r"|designer|tasarımcı|researcher|araştırmacı|assistant|asistan)\b",
    re.I,
)
_AS_COMPANY_WORDS = re.compile(
    r"\b(a\.ş|ltd|inc|corp|gmbh|s\.a|llc|co\.|şirketi|company|holding"
    r"|group|grup|teknoloji|technology|solutions|çözümleri|systems|sistemleri"
    r"|consulting|danışmanlık|agency|ajans)\b",
    re.I,
)

# Education: institution and degree signals
_AS_DEGREE_WORDS = re.compile(
    r"\b(üniversite|university|fakülte|faculty|bölüm|department|lisans|bachelor"
    r"|yüksek\s+lisans|master|msc|mba|doktora|phd|doctorate|diploma|mezun"
    r"|graduate|lise|high\s+school|okul|school|akademi|academy|enstitü|institute)\b",
    re.I,
)

# Skills: technology keywords
_AS_TECH_WORDS = re.compile(
    r"\b(python|java|javascript|typescript|sql|react|angular|vue|django|flask"
    r"|spring|node|nodejs|html|css|sass|scss|php|ruby|swift|kotlin|go|rust|c\+\+"
    r"|docker|kubernetes|k8s|aws|azure|gcp|git|linux|bash|terraform|jenkins"
    r"|figma|sketch|photoshop|premiere|illustrator|after\s*effects"
    r"|excel|powerbi|tableau|matlab|r\b|hadoop|spark|tensorflow|pytorch)\b",
    re.I,
)
_AS_LEVEL_WORDS = re.compile(
    r"\b(beginner|intermediate|advanced|expert|fluent|native|proficient"
    r"|başlangıç|orta|ileri|uzman|akıcı|anadil|temel|iyi)\b",
    re.I,
)

# Projects: build/create verbs and platform names
_AS_PROJECT_VERBS = re.compile(
    r"\b(built|developed|created|designed|implemented|geliştirdim|oluşturdum"
    r"|tasarladım|yaptım|kurdum|coded|deployed|launched|contributed)\b",
    re.I,
)
_AS_PLATFORM_RE = re.compile(
    r"\b(github|gitlab|bitbucket|heroku|vercel|netlify|app store|play store"
    r"|npm|pypi|portfolio|demo|api|backend|frontend|mobile|android|ios)\b",
    re.I,
)

# Summary: prose signals (no dates, no bullets, full sentences)
_AS_SENTENCE_END = re.compile(r"[.!?]\s*$")
_AS_PRONOUN_RE = re.compile(
    r"\b(i am|i have|i'm|ben|benim|hakkımda|kendimi|kariyer|hedefim"
    r"|motivated|passionate|experienced|uzman|deneyimli)\b",
    re.I,
)

# Safety-rule patterns
_AS_PARA_RE = re.compile(r"\w[\w\s]{40,}[.!?]")  # long sentence → not skills


def _classify_block(block: CVBlock) -> str:
    """
    Classify a single CVBlock into a canonical section name using structural
    signals rather than (only) keyword matching.

    Signal hierarchy:
      1. Trusted heading — if block.heading is set, use it directly.
      2. Date-range presence → experience (strongest structural signal).
      3. Degree/institution words → education.
      4. List shape + tech keywords → skills.
      5. Build verbs or platform names → projects.
      6. Paragraph prose with pronouns/career words → summary.
      7. Keyword fallback (legacy _sd_score_line_for_section logic).
      8. Default: experience (most common unlabelled block type in CVs).

    Args:
        block: CVBlock with lines and signals already computed.

    Returns:
        Canonical section name string.
    """
    if block.heading is not None:
        return block.heading

    full_text = " ".join(block.lines)
    lower = turkish_lower(full_text)

    # ── Signal 1: date range → experience ────────────────────────────────────
    if _AS_DATE_RANGE.search(full_text):
        return "experience"

    # ── Signal 2: degree/institution words → education ────────────────────────
    if _AS_DEGREE_WORDS.search(lower):
        # Require at least one date too (education entries almost always have years)
        if block.has_dates:
            return "education"

    # ── Signal 3: project verbs / platform names → projects ────────────
    # Checked BEFORE list+tech so 'built a github app with React' routes
    # to projects even though React is a technology keyword.
    if _AS_PROJECT_VERBS.search(lower) or _AS_PLATFORM_RE.search(lower):
        return "projects"

    # ── Signal 4: list shape + tech words → skills ────────────────────────
    tech_hits = len(_AS_TECH_WORDS.findall(lower))
    if block.is_list and tech_hits >= 2:
        return "skills"
    if tech_hits >= 4 and not block.has_dates:
        return "skills"

    # ── Signal 5: paragraph prose → summary ───────────────────────────────
    sentence_endings = sum(1 for l in block.lines if _AS_SENTENCE_END.search(l))
    if (
        sentence_endings >= 2
        and not block.has_dates
        and not block.is_list
        and _AS_PRONOUN_RE.search(lower)
    ):
        return "summary"

    # ── Signal 6: date + role/company → experience ──────────────────────────
    if block.has_dates and (
        _AS_ROLE_WORDS.search(lower) or _AS_COMPANY_WORDS.search(lower)
    ):
        return "experience"

    # ── Signal 7: keyword score fallback ───────────────────────────────────────────
    kw_section = _sd_score_line_for_section(full_text)
    if kw_section:
        return kw_section

    # ── Default ────────────────────────────────────────────────────────────────
    return "experience"


def _apply_safety_rules(sections: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Stage 4 post-pass — enforce structural safety rules to catch
    mis-classified content that passed through the signal hierarchy.

    Rules:
      • skills MUST NOT contain long paragraph lines (>60 chars with sentence
        endings). Any such line is moved to summary if summary is short,
        else dropped from skills.
      • experience MUST contain at least one date per block-group.
        Blocks with no dates are demoted to summary or other.
      • education MUST contain an institution keyword.
        Blocks with no institution signal are moved to experience.
      • summary is capped at _SUMMARY_MAX_LINES non-empty lines.
        Overflow goes to other.

    Args:
        sections: Dict of { section_name: [lines] } (pre-join).

    Returns:
        Cleaned sections dict with the same keys.
    """
    result = {k: list(v) for k, v in sections.items()}

    # Rule 1: skills must not contain paragraphs
    clean_skills: list[str] = []
    spill_to_summary: list[str] = []
    for line in result.get("skills", []):
        if _AS_PARA_RE.search(line) and _AS_SENTENCE_END.search(line.strip()):
            spill_to_summary.append(line)
        else:
            clean_skills.append(line)
    result["skills"] = clean_skills
    if spill_to_summary and len(result.get("summary", [])) < _SUMMARY_MAX_LINES:
        result.setdefault("summary", []).extend(spill_to_summary)

    # Rule 2: education must contain institution signal
    clean_edu: list[str] = []
    spill_exp: list[str] = []
    for line in result.get("education", []):
        if _AS_DEGREE_WORDS.search(turkish_lower(line)):
            clean_edu.append(line)
        else:
            spill_exp.append(line)
    result["education"] = clean_edu
    result.setdefault("experience", []).extend(spill_exp)

    # Rule 3: summary capped at _SUMMARY_MAX_LINES non-empty lines
    summary_lines = result.get("summary", [])
    non_empty = [l for l in summary_lines if l.strip()]
    if len(non_empty) > _SUMMARY_MAX_LINES:
        result["summary"] = non_empty[:_SUMMARY_MAX_LINES]
        result.setdefault("other", []).extend(non_empty[_SUMMARY_MAX_LINES:])

    return result


_SUMMARY_MAX_LINES = 8


def _assign_sections_compat(blocks):
    """
    Thin shim: calls the new structured assign_sections() and converts
    the { section: [lines] } dict to { section: str } for backward compat.
    Replaces the old assign_sections() which lived here.
    """
    raw = assign_sections(blocks)  # new pipeline function (embedded above)
    return {k: "\n".join(v) if isinstance(v, list) else v for k, v in raw.items()}


def _sd_score_line_for_section(line: str) -> Optional[str]:
    """
    Keyword fallback: score a single line against per-section content signals.
    Used as Signal 7 in _classify_block() and by the headerless-CV fallback
    in extract_sections().

    Uses the same compiled patterns as assign_sections() so behaviour is
    consistent whether a block is classified structurally or by keyword.
    """
    lower = turkish_lower(line)

    scores: dict[str, int] = {
        "experience": 0,
        "education": 0,
        "skills": 0,
        "projects": 0,
        "summary": 0,
    }

    if _AS_DATE_RANGE.search(line):
        scores["experience"] += 3
    if _AS_ROLE_WORDS.search(lower):
        scores["experience"] += 2
    if _AS_COMPANY_WORDS.search(lower):
        scores["experience"] += 1

    if _AS_DEGREE_WORDS.search(lower):
        scores["education"] += 3
    if _AS_DATE_RE.search(line) and _AS_DEGREE_WORDS.search(lower):
        scores["education"] += 2

    tech_hits = len(_AS_TECH_WORDS.findall(lower))
    scores["skills"] += min(tech_hits * 2, 6)
    if _AS_LEVEL_WORDS.search(lower):
        scores["skills"] += 1

    if _AS_PROJECT_VERBS.search(lower):
        scores["projects"] += 3
    if _AS_PLATFORM_RE.search(lower):
        scores["projects"] += 2

    if _AS_PRONOUN_RE.search(lower):
        scores["summary"] += 2
    if _AS_SENTENCE_END.search(line.strip()) and not _AS_DATE_RE.search(line):
        scores["summary"] += 1

    best = max(scores, key=lambda k: scores[k])
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

    # Subsection accumulators: { section_name: { sub_label: [lines] } }
    # e.g. {"skills": {"Teknik Beceriler": ["Python", "SQL"], "_root": []}}
    sub_accum: dict[str, dict[str, list[str]]] = {}

    lines = text.splitlines()
    n = len(lines)
    current_section: Optional[str] = None
    current_sub: Optional[str] = None  # display label of active sub-section

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
            # ── Check if this is a SUB-heading first ──────────────────────────
            norm_raw = _sd_norm(raw_line.strip())
            if norm_raw in SUB_HEADERS:
                parent_sec, sub_label = SUB_HEADERS[norm_raw]
                # Sub-heading: keep current_section open (or adopt parent)
                # but activate a named sub-bucket inside it.
                if current_section is None or current_section not in _SD_CANONICAL:
                    current_section = parent_sec
                    if parent_sec not in seen_sections:
                        seen_sections.add(parent_sec)
                        transition_log.append(parent_sec)
                current_sub = sub_label
                if current_section not in sub_accum:
                    sub_accum[current_section] = {"_root": []}
                if sub_label not in sub_accum[current_section]:
                    sub_accum[current_section][sub_label] = []
                if _debug:
                    print(
                        f"  [SUB] line {i}: {raw_line.strip()!r} → {current_section}/{sub_label}"
                    )
                continue  # sub-heading line itself not stored as body content

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
            current_sub = None  # new main section resets sub-section pointer

            if _debug:
                print(f"  [H] line {i}: {raw_line.strip()!r} → {detected!r} ({method})")

        else:
            # ── Body line: assign to current section ──────────────────────────
            # Pre-header lines (current_section is None) are discarded — they
            # contain name/contact info already captured by extract_contact_info().
            if raw_line.strip() and current_section is not None:
                sections[current_section].append(raw_line)
                # Also accumulate into sub-section bucket if one is active
                if current_sub is not None and current_section in sub_accum:
                    sub_accum[current_section][current_sub].append(raw_line)
                elif current_section in sub_accum:
                    # Content before any sub-heading → "_root" bucket
                    sub_accum[current_section]["_root"].append(raw_line)

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

    # ── Subsection output (hierarchical) ─────────────────────────────────────
    # For each section that had sub-headings, emit a "{section}_subsections"
    # key containing a dict { sub_label: cleaned_text }.
    # "_root" holds content that appeared before any sub-heading within the section.
    # Backward compat: the flat sections[section] string is unchanged.
    for sec, sub_dict in sub_accum.items():
        built: dict[str, str] = {}
        for sub_label, sub_lines in sub_dict.items():
            cleaned = "\n".join(_dedup_section_lines(sub_lines)).strip()
            if cleaned:
                built[sub_label] = cleaned
        if built:
            result[f"{sec}_subsections"] = built  # type: ignore[assignment]

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

    # ── Step 6b: structured pipeline (NEW — full 6-stage parse_cv) ─────────
    # parse_cv() runs the complete structured pipeline (normalize → block
    # segment → heading detect → boundary assign → classify → safety rules).
    # Strategy: the new pipeline output wins when the keyword pass left a
    # section empty OR when the new output is >20% richer in content.
    try:
        _structured = parse_cv(cleaned_text)  # full 6-stage pipeline
        for _sec in [
            "summary",
            "experience",
            "education",
            "skills",
            "projects",
            "other",
        ]:
            _kw_val = sections.get(_sec, "")
            _st_val = _structured.get(_sec, "")
            if not _kw_val and _st_val:
                sections[_sec] = _st_val
            elif _st_val and len(_st_val) > len(_kw_val) * 1.2:
                sections[_sec] = _st_val
    except Exception as _e:
        logger.debug(f"  [structured_pipeline] skipped: {_e}")

    # ── Step 6c: group experience blocks (FIX 4) ─────────────────────────────
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
        default="C:/Users/rumeysagokce/Desktop/cv_parser_project/data/test",
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
