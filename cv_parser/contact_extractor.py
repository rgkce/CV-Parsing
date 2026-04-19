import re
from typing import Dict


class ContactExtractor:
    def __init__(self):

        self.common_domains = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com"]

    # -----------------------------
    # MAIN
    # -----------------------------
    def extract(self, raw_text: str) -> Dict[str, str]:

        text = self._preprocess(raw_text)

        return {
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "linkedin": self._extract_linkedin(text),
            "github": self._extract_github(text),
        }

    # -----------------------------
    # SAFE PREPROCESS (FIXED)
    # -----------------------------
    def _preprocess(self, text: str) -> str:

        if not text:
            return ""

        # DO NOT destroy structure fully
        text = re.sub(r"[\u200b-\u200d\uFEFF]", "", text)

        text = re.sub(r"\s+", " ", text)

        # fix spaced separators but carefully
        text = re.sub(r"\s*@\s*", "@", text)
        text = re.sub(r"\s*\.\s*", ".", text)

        return text

    # -----------------------------
    # EMAIL EXTRACTION (IMPROVED)
    # -----------------------------
    def _extract_email(self, text: str) -> str:

        # 1. direct match
        pattern = r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
        matches = re.findall(pattern, text.lower())

        if matches:
            return matches[0]

        # 2. smart reconstruction
        return self._reconstruct_email(text)

    # -----------------------------
    # SMART EMAIL REPAIR (KEY FIX)
    # -----------------------------
    def _reconstruct_email(self, text: str) -> str:

        tokens = text.lower().split()

        for i in range(len(tokens)):
            # sliding window (1–5 tokens)
            window = tokens[i : i + 5]

            candidate = "".join(window)

            if "@" in candidate and "." in candidate:
                if self._is_valid_email(candidate):
                    return candidate

        return ""

    def _is_valid_email(self, email: str) -> bool:

        pattern = r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$"
        return bool(re.match(pattern, email))

    # -----------------------------
    # PHONE EXTRACTION (FIXED)
    # -----------------------------
    def _extract_phone(self, text: str) -> str:

        # extract candidates first
        candidates = re.findall(r"\+?\d[\d\s\-()]{7,}\d", text)

        for c in candidates:
            normalized = re.sub(r"[^\d+]", "", c)

            if self._is_valid_phone(normalized):
                return self._normalize_phone(normalized)

        return ""

    def _is_valid_phone(self, phone: str) -> bool:

        digits = re.sub(r"[^\d]", "", phone)

        # CV-safe constraints
        return 10 <= len(digits) <= 13

    def _normalize_phone(self, phone: str) -> str:

        digits = re.sub(r"[^\d+]", "", phone)

        if digits.startswith("0") and len(digits) == 11:
            return "+90" + digits[1:]

        if digits.startswith("90") and not digits.startswith("+"):
            return "+" + digits

        if not digits.startswith("+"):
            return "+" + digits

        return digits

    # -----------------------------
    # LINKEDIN (STRONGER)
    # -----------------------------
    def _extract_linkedin(self, text: str) -> str:

        pattern = r"(https?://)?(www\.)?linkedin\.com/[a-z0-9\-_/]+"

        match = re.search(pattern, text.lower())

        if match:
            return self._normalize_url(match.group())

        return ""

    # -----------------------------
    # GITHUB (STRONGER)
    # -----------------------------
    def _extract_github(self, text: str) -> str:

        pattern = r"(https?://)?(www\.)?github\.com/[a-z0-9\-_/]+"

        match = re.search(pattern, text.lower())

        if match:
            return self._normalize_url(match.group())

        return ""

    # -----------------------------
    # URL NORMALIZER
    # -----------------------------
    def _normalize_url(self, url: str) -> str:

        url = url.strip().rstrip(".,;")

        if not url.startswith("http"):
            url = "https://" + url

        return url
