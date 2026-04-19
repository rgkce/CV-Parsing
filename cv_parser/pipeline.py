import os
import uuid
import logging
from typing import List, Dict

from cv_parser.contact_extractor import ContactExtractor
from cv_parser.cv_scorer import CVQualityScorer
from cv_parser.metadata_extractor import MetadataExtractor
from cv_parser.pdf_parser import PDFParser
from cv_parser.section_extractor import SectionExtractor


# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
class CVParsingPipeline:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.section_extractor = SectionExtractor()
        self.contact_extractor = ContactExtractor()
        self.metadata_extractor = MetadataExtractor()
        self.scorer = CVQualityScorer()

    def parse(self, pdf_path: str) -> dict:

        resume_id = str(uuid.uuid4())
        logging.info(f"Processing: {pdf_path}")

        try:
            # 1. PDF → RAW TEXT
            raw_text = self.pdf_parser.parse(pdf_path)
            raw_text = raw_text.lower().strip() if raw_text else ""

            # 2. SECTION EXTRACTION
            sections, section_confidence = self.section_extractor.extract_sections(
                raw_text
            )

            # safety fallback
            sections = sections or {}
            section_confidence = section_confidence or {}

            # 3. CONTACT EXTRACTION
            contact = self.contact_extractor.extract(raw_text) or {}

            # 4. METADATA EXTRACTION
            metadata = self.metadata_extractor.extract(raw_text, pdf_path) or {}

            # 5. QUALITY SCORING
            score_input = {
                "raw_text": raw_text,
                "sections": sections,
                "contact": contact,
                "metadata": metadata,
            }

            quality_score = self.scorer.score(score_input)

            # 6. FINAL OUTPUT
            return {
                "resume_id": resume_id,
                "file_path": pdf_path,
                "raw_text": raw_text,
                "sections": {
                    "summary": sections.get("summary", ""),
                    "experience": sections.get("experience", ""),
                    "education": sections.get("education", ""),
                    "skills": sections.get("skills", ""),
                    "projects": sections.get("projects", ""),
                },
                "section_confidence": {
                    "summary": section_confidence.get("summary", 0.0),
                    "experience": section_confidence.get("experience", 0.0),
                    "education": section_confidence.get("education", 0.0),
                    "skills": section_confidence.get("skills", 0.0),
                    "projects": section_confidence.get("projects", 0.0),
                },
                "contact": {
                    "email": contact.get("email", ""),
                    "phone": contact.get("phone", ""),
                    "linkedin": contact.get("linkedin", ""),
                    "github": contact.get("github", ""),
                },
                "has_photo": metadata.get("has_photo", False),
                "language": metadata.get("language", "eng"),
                "source_format": "pdf",
                "score": quality_score,
            }

        except Exception as e:
            logging.error(f"Failed to parse {pdf_path}: {str(e)}")

            return {"resume_id": resume_id, "file_path": pdf_path, "error": str(e)}


# -----------------------------
# BATCH PROCESSOR (FIXED)
# -----------------------------
class CVBatchProcessor:
    def __init__(self):
        self.pipeline = CVParsingPipeline()

    def process(self, pdf_paths: List[str]) -> List[Dict]:
        results = []

        for idx, path in enumerate(pdf_paths):
            logging.info(f"[{idx + 1}/{len(pdf_paths)}] Processing file")

            result = self.pipeline.parse(path)
            results.append(result)

        return results

    # 🔥 NEW: folder support (senin asıl ihtiyacın buydu)
    def process_folder(self, folder_path: str) -> List[Dict]:
        pdf_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".pdf")
        ]

        return self.process(pdf_files)


# -----------------------------
# TEST RUN (FIXED)
# -----------------------------
if __name__ == "__main__":
    pipeline = CVParsingPipeline()

    # ❌ WRONG (senin eski kodun): folder path tek PDF gibi verilmiş
    # pdf_file = "C:/.../PDF"
    # result = pipeline.parse(pdf_file)

    # ✅ CORRECT TEST
    test_file = (
        "C:/Users/rumeysagokce/Desktop/cv_parser_project/data/PDF/rumeysa gokce 1.pdf"
    )
    result = pipeline.parse(test_file)

    print(result)

    # Batch test
    batch = CVBatchProcessor()
    results = batch.process_folder(
        "C:/Users/rumeysagokce/Desktop/cv_parser_project/data/PDF"
    )

    print(len(results))
