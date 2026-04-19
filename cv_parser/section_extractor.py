import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher


class SectionExtractor:
    def __init__(self):

        self.section_map = {
            "summary": [
                # EN - core
                "summary",
                "professional summary",
                "profile",
                "about",
                "about me",
                "career summary",
                "personal summary",
                "executive summary",
                "professional profile",
                # EN - variations
                "overview",
                "career overview",
                "bio",
                "introduction",
                "who i am",
                "career objective",
                "objective",
                "personal profile",
                "professional overview",
                "summary of qualifications",
                "qualifications summary",
                # TR - core
                "özet",
                "profil",
                "hakkımda",
                "kariyer özeti",
                "kişisel özet",
                "profesyonel özet",
                # TR - variations
                "genel bakış",
                "kariyer profili",
                "tanıtım",
                "kendim hakkında",
                "kişisel bilgiler",
                "özgeçmiş özeti",
                "mesleki özet",
                "nitelikler özeti",
                "kariyer hedefi",
                "hedef",
                "amaç",
            ],
            "experience": [
                # EN - core
                "experience",
                "work experience",
                "professional experience",
                "employment",
                "career",
                "background",
                # EN - variations (very common in CVs)
                "work history",
                "professional background",
                "career history",
                "employment history",
                "job experience",
                "work record",
                "professional track record",
                "relevant experience",
                "industry experience",
                "experience summary",
                "professional journey",
                "career path",
                # EN - alternative headings
                "positions held",
                "work experience summary",
                "employment background",
                "career overview",
                "previous experience",
                "past experience",
                "work profile",
                # TR - core
                "deneyim",
                "iş deneyimi",
                "çalışma deneyimi",
                "profesyonel deneyim",
                "kariyer",
                # TR - variations
                "iş tecrübesi",
                "çalışma geçmişi",
                "kariyer geçmişi",
                "mesleki deneyim",
                "mesleki geçmiş",
                "iş geçmişi",
                "deneyimler",
                "tecrübeler",
                "kariyer özeti",
                "profesyonel geçmiş",
                "çalıştığı yerler",
                "görevler",
                "görev geçmişi",
                "iş tecrübe özeti",
                # TR - alternative CV styles
                "çalışma hayatı",
                "iş hayatı",
                "kariyer yolculuğu",
                "profesyonel yolculuk",
            ],
            "education": [
                # EN - core
                "education",
                "academic background",
                "qualifications",
                # EN - very common CV variants
                "education and training",
                "academic qualifications",
                "educational background",
                "academic history",
                "education history",
                "schooling",
                "study background",
                "studies",
                "academic profile",
                # EN - degree-based headings
                "degrees",
                "degree",
                "academic degrees",
                "certification",
                "certifications",
                "diploma",
                "academic record",
                "formal education",
                "educational qualifications",
                # EN - alternative CV styles
                "education summary",
                "training and education",
                "learning background",
                "education details",
                "academic journey",
                # TR - core
                "eğitim",
                "akademik bilgiler",
                # TR - common variants
                "eğitim bilgileri",
                "öğrenim durumu",
                "öğrenim bilgileri",
                "eğitim geçmişi",
                "akademik geçmiş",
                "okul bilgileri",
                "okul geçmişi",
                # TR - degree & diploma oriented
                "diploma",
                "derece",
                "dereceler",
                "sertifikalar",
                "eğitim ve sertifikalar",
                "akademik derece",
                "mezuniyet",
                "öğrenim",
                "öğrenim hayatı",
                # TR - alternative CV phrasing
                "eğitim hayatı",
                "eğitim özeti",
                "öğrenim süreci",
                "akademik yolculuk",
            ],
            "skills": [
                # EN - core
                "skills",
                "technical skills",
                "competencies",
                # EN - very common CV variants
                "core competencies",
                "key skills",
                "professional skills",
                "expertise",
                "areas of expertise",
                "technical expertise",
                "core skills",
                "skill set",
                "abilities",
                "capabilities",
                "proficiencies",
                # EN - tech CV specific
                "technologies",
                "tech stack",
                "technology stack",
                "tools",
                "tools & technologies",
                "programming skills",
                "development skills",
                "software skills",
                "it skills",
                "engineering skills",
                # EN - modern CV variations
                "what i know",
                "what i can do",
                "my skills",
                "my expertise",
                "strengths",
                "core strengths",
                # TR - core
                "yetenekler",
                "beceriler",
                "uzmanlıklar",
                # TR - common variants
                "teknik beceriler",
                "mesleki beceriler",
                "temel beceriler",
                "yetenek seti",
                "uzmanlık alanları",
                "teknik yetenekler",
                "profesyonel beceriler",
                # TR - tech CV variants
                "teknolojiler",
                "kullandığım teknolojiler",
                "teknoloji seti",
                "araçlar",
                "kullandığım araçlar",
                "yazılım becerileri",
                "programlama becerileri",
                "it becerileri",
                # TR - alternative phrasing
                "neler biliyorum",
                "yapabildiklerim",
                "güçlü yönler",
                "yetkinlikler",
            ],
            "projects": [
                # EN - core
                "projects",
                "personal projects",
                "academic projects",
                # EN - very common CV variants
                "project experience",
                "project work",
                "selected projects",
                "notable projects",
                "key projects",
                "relevant projects",
                "portfolio",
                "my projects",
                "project portfolio",
                # EN - dev / tech CV variants
                "side projects",
                "open source projects",
                "software projects",
                "development projects",
                "engineering projects",
                "research projects",
                "technical projects",
                # EN - alternative headings (VERY IMPORTANT)
                "what i built",
                "things i built",
                "work samples",
                "case studies",
                "case study",
                "portfolio projects",
                "builds",
                # TR - core
                "projeler",
                "kişisel projeler",
                "akademik projeler",
                # TR - variants
                "proje deneyimi",
                "proje çalışmaları",
                "seçilmiş projeler",
                "önemli projeler",
                "ilgili projeler",
                "yaptığım projeler",
                "portföy",
                "proje portföyü",
                # TR - tech CV variants
                "yazılım projeleri",
                "geliştirme projeleri",
                "araştırma projeleri",
                "teknik projeler",
                # TR - alternative phrasing
                "neler yaptım",
                "yaptıklarım",
                "çalışmalar",
                "uygulamalar",
                "case study",
            ],
        }

    # -----------------------------
    # MAIN
    # -----------------------------
    def extract_sections(self, raw_text: str) -> Tuple[Dict, Dict]:

        lines = self._preprocess(raw_text)

        headers = self._detect_headers(lines)

        sections = self._build_sections(lines, headers)

        confidence = self._compute_confidence(sections, headers)

        return sections, confidence

    # -----------------------------
    # PREPROCESS (IMPORTANT FIX)
    # -----------------------------
    def _preprocess(self, text: str) -> List[str]:

        # split more intelligently
        text = re.sub(r"\n+", "\n", text)

        lines = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # split inline headers like "skills: python, java"
            if ":" in line:
                parts = line.split(":", 1)
                lines.append(parts[0])
                lines.append(parts[1])
            else:
                lines.append(line)

        return lines

    # -----------------------------
    # HEADER DETECTION (FIXED + FUZZY)
    # -----------------------------
    def _detect_headers(self, lines: List[str]) -> List[Dict]:

        headers = []

        for i, line in enumerate(lines):
            norm = self._normalize(line)

            section, score = self._match_section(norm)

            if section and score > 0.72:
                headers.append(
                    {"index": i, "section": section, "text": line, "score": score}
                )

        return headers

    # -----------------------------
    # FUZZY MATCH ENGINE (CRITICAL UPGRADE)
    # -----------------------------
    def _match_section(self, text: str):

        best_section = None
        best_score = 0.0

        for section, keywords in self.section_map.items():
            for kw in keywords:
                sim = SequenceMatcher(None, text, kw).ratio()

                # boost exact / contains
                if kw in text:
                    sim = max(sim, 0.9)

                if sim > best_score:
                    best_score = sim
                    best_section = section

        return best_section, best_score

    # -----------------------------
    # SECTION BUILDER (FIXED)
    # -----------------------------
    def _build_sections(self, lines: List[str], headers: List[Dict]) -> Dict:

        sections = {k: "" for k in self.section_map.keys()}

        if not headers:
            return sections

        headers = sorted(headers, key=lambda x: x["index"])

        for i, h in enumerate(headers):
            start = h["index"] + 1
            end = headers[i + 1]["index"] if i + 1 < len(headers) else len(lines)

            content = " ".join(lines[start:end]).strip()

            section = h["section"]

            # merge instead of overwrite (IMPORTANT FIX)
            if sections[section]:
                sections[section] += " " + content
            else:
                sections[section] = content

        return sections

    # -----------------------------
    # CONFIDENCE (IMPROVED LOGIC)
    # -----------------------------
    def _compute_confidence(self, sections: Dict, headers: List[Dict]) -> Dict:

        header_scores = {}

        for h in headers:
            sec = h["section"]
            header_scores.setdefault(sec, []).append(h["score"])

        confidence = {}

        for sec, text in sections.items():
            header_score = sum(header_scores.get(sec, [0])) / max(
                len(header_scores.get(sec, [1])), 1
            )

            length_score = min(len(text.split()) / 80, 1.0)

            keyword_score = self._keyword_density(sec, text)

            confidence[sec] = round(
                0.5 * header_score + 0.3 * length_score + 0.2 * keyword_score, 3
            )

        return confidence

    # -----------------------------
    # KEYWORD DENSITY
    # -----------------------------
    def _keyword_density(self, section: str, text: str) -> float:

        keywords = self.section_map.get(section, [])
        text = text.lower()

        if not keywords:
            return 0.0

        hits = sum(1 for k in keywords if k in text)

        return hits / len(keywords)

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    def _normalize(self, text: str) -> str:

        text = text.lower()

        text = re.sub(r"[^a-zçğıöşü0-9\s]", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()
