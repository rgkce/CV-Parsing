from typing import Dict


class CVQualityScorer:
    def __init__(self):
        self.required_sections = [
            "summary",
            "experience",
            "education",
            "skills",
            "projects",
        ]

    # -----------------------------
    # MAIN ENTRY
    # -----------------------------
    def score(self, parsed_cv: Dict) -> Dict:
        section_score = self._score_sections(parsed_cv.get("sections", {}))
        contact_score = self._score_contact(parsed_cv.get("contact", {}))
        text_score = self._score_text(parsed_cv.get("raw_text", ""))
        metadata_score = self._score_metadata(parsed_cv.get("metadata", {}))

        final_score = (
            0.4 * section_score
            + 0.2 * contact_score
            + 0.2 * text_score
            + 0.2 * metadata_score
        )

        return {
            "score": round(final_score, 3),
            "details": {
                "sections_score": round(section_score, 3),
                "contact_score": round(contact_score, 3),
                "text_score": round(text_score, 3),
                "metadata_score": round(metadata_score, 3),
                "weights": {
                    "sections": 0.4,
                    "contact": 0.2,
                    "text": 0.2,
                    "metadata": 0.2,
                },
            },
        }

    # -----------------------------
    # 1. SECTION COMPLETENESS
    # -----------------------------
    def _score_sections(self, sections: Dict) -> float:
        if not sections:
            return 0.0

        present = 0
        quality_bonus = 0

        for sec in self.required_sections:
            content = sections.get(sec, "")

            if content and len(content.strip()) > 10:
                present += 1

                # bonus for meaningful content
                if len(content.split()) > 30:
                    quality_bonus += 0.05

        base = present / len(self.required_sections)

        return min(base + quality_bonus, 1.0)

    # -----------------------------
    # 2. CONTACT EXTRACTION QUALITY
    # -----------------------------
    def _score_contact(self, contact: Dict) -> float:
        if not contact:
            return 0.0

        fields = ["email", "phone", "linkedin", "github"]

        found = 0

        for f in fields:
            value = contact.get(f, "")
            if value and value.strip():
                found += 1

        base = found / len(fields)

        # bonus if email + phone exist (high value signals)
        bonus = 0
        if contact.get("email") and contact.get("phone"):
            bonus = 0.1

        return min(base + bonus, 1.0)

    # -----------------------------
    # 3. TEXT QUALITY
    # -----------------------------
    def _score_text(self, text: str) -> float:
        if not text:
            return 0.0

        word_count = len(text.split())

        # length score
        if word_count < 100:
            length_score = 0.3
        elif word_count < 300:
            length_score = 0.7
        else:
            length_score = 1.0

        # coherence score (simple heuristic)
        weird_chars = sum(1 for c in text if not c.isalnum() and c not in " .,\n")
        coherence = 1 - min(weird_chars / max(len(text), 1), 1.0)

        return (length_score + coherence) / 2

    # -----------------------------
    # 4. METADATA QUALITY
    # -----------------------------
    def _score_metadata(self, metadata: Dict) -> float:
        if not metadata:
            return 0.0

        score = 0

        # language detected
        if metadata.get("language") in ["tr", "eng"]:
            score += 0.5

        # photo detection
        if isinstance(metadata.get("has_photo"), bool):
            score += 0.5

        return score
