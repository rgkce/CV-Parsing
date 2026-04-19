import fitz
import pdfplumber
from langdetect import detect, DetectorFactory, LangDetectException
import re


class MetadataExtractor:
    def __init__(self):
        DetectorFactory.seed = 0

        # CV-specific language hints
        self.tr_indicators = {
            "ve",
            "bir",
            "bu",
            "için",
            "ile",
            "ben",
            "deneyim",
            "eğitim",
            "yetenek",
            "proje",
            "çalışma",
            "öğrenci",
        }

        self.en_indicators = {
            "and",
            "the",
            "for",
            "with",
            "experience",
            "education",
            "skills",
            "project",
            "student",
            "work",
            "engineer",
        }

    # -----------------------------
    # MAIN
    # -----------------------------
    def extract(self, raw_text: str, pdf_path: str) -> dict:

        language = self.detect_language(raw_text)
        has_photo = self.detect_photo(pdf_path)

        return {"language": language, "has_photo": has_photo}

    # -----------------------------
    # IMPROVED LANGUAGE DETECTION
    # -----------------------------
    def detect_language(self, text: str) -> str:

        if not text:
            return "eng"

        text = text.lower()

        # sample better (middle section is more stable than start)
        mid = len(text) // 2
        sample = text[mid : mid + 1500]

        try:
            lang_detected = detect(sample)
        except LangDetectException:
            lang_detected = "unknown"

        # indicator scoring (IMPORTANT FIX)
        tr_score = sum(1 for w in self.tr_indicators if w in sample)
        en_score = sum(1 for w in self.en_indicators if w in sample)

        # decision logic
        if tr_score > en_score:
            return "tr"

        if en_score > tr_score:
            return "eng"

        # fallback to langdetect
        if lang_detected == "tr":
            return "tr"

        return "eng"

    # -----------------------------
    # IMPROVED PHOTO DETECTION
    # -----------------------------
    def detect_photo(self, pdf_path: str) -> bool:

        images = self._extract_images(pdf_path)

        if not images:
            return False

        # filter out small icons/logos
        valid_images = [img for img in images if self._is_likely_photo(img)]

        return len(valid_images) > 0

    # -----------------------------
    # IMAGE EXTRACTION
    # -----------------------------
    def _extract_images(self, pdf_path):

        images = []

        try:
            doc = fitz.open(pdf_path)

            for page in doc:
                for img in page.get_images(full=True):
                    images.append(img)

        except Exception:
            pass

        return images

    # -----------------------------
    # FILTER LOGOS / ICONS
    # -----------------------------
    def _is_likely_photo(self, img):

        # img tuple: (xref, smask, width, height, ...)
        try:
            width = img[2]
            height = img[3]

            # heuristic:
            # real photos usually larger than icons/logos
            if width > 80 and height > 80:
                return True

        except Exception:
            pass

        return False
