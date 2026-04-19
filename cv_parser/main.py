import os
import json
import time
import uuid
import logging
import argparse
from typing import List, Dict, Union

from pdf_parser import PDFParser
from section_extractor import SectionExtractor
from contact_extractor import ContactExtractor
from metadata_extractor import MetadataExtractor
from cv_scorer import CVQualityScorer


# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("cv_pipeline")


# -----------------------------
# INIT COMPONENTS (SAFE SINGLE PROCESS ONLY)
# -----------------------------
pdf_parser = PDFParser()
section_extractor = SectionExtractor()
contact_extractor = ContactExtractor()
metadata_extractor = MetadataExtractor()
cv_scorer = CVQualityScorer()


# -----------------------------
# SAFE TEXT NORMALIZATION
# -----------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


# -----------------------------
# SAFE DICT MERGE
# -----------------------------
def safe_merge(base: Dict, update: Dict) -> Dict:
    if not update:
        return base
    for k, v in update.items():
        if v is not None:
            base[k] = v
    return base


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_single_cv(file_path: str) -> Dict:

    start_time = time.time()
    resume_id = str(uuid.uuid4())

    logger.info(f"Processing: {file_path}")

    output = {
        "resume_id": resume_id,
        "file_path": file_path,
        "raw_text": "",
        "sections": {
            "summary": "",
            "experience": "",
            "education": "",
            "skills": "",
            "projects": "",
        },
        "section_confidence": {
            "summary": 0.0,
            "experience": 0.0,
            "education": 0.0,
            "skills": 0.0,
            "projects": 0.0,
        },
        "contact": {
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": "",
        },
        "has_photo": False,
        "language": "eng",
        "source_format": "pdf",
        "score": 0.0,
    }

    try:
        # -----------------------------
        # 1. PDF PARSE
        # -----------------------------
        raw_text = normalize_text(pdf_parser.parse(file_path))
        output["raw_text"] = raw_text

        # -----------------------------
        # 2. SECTIONS
        # -----------------------------
        try:
            sections, conf = section_extractor.extract_sections(raw_text)
            output["sections"] = safe_merge(output["sections"], sections)
            output["section_confidence"] = safe_merge(
                output["section_confidence"], conf
            )
        except Exception as e:
            logger.warning(f"Section error: {file_path} | {e}")

        # -----------------------------
        # 3. CONTACT
        # -----------------------------
        try:
            contact = contact_extractor.extract(raw_text)
            output["contact"] = safe_merge(output["contact"], contact)
        except Exception as e:
            logger.warning(f"Contact error: {file_path} | {e}")

        # -----------------------------
        # 4. METADATA
        # -----------------------------
        try:
            metadata = metadata_extractor.extract(raw_text, file_path) or {}
            output["language"] = metadata.get("language", "eng")
            output["has_photo"] = metadata.get("has_photo", False)
        except Exception as e:
            logger.warning(f"Metadata error: {file_path} | {e}")

        # -----------------------------
        # 5. SCORING
        # -----------------------------
        try:
            score = cv_scorer.score(
                {
                    "raw_text": raw_text,
                    "sections": output["sections"],
                    "contact": output["contact"],
                    "metadata": {
                        "language": output["language"],
                        "has_photo": output["has_photo"],
                    },
                }
            )

            output["score"] = score.get("score", 0.0)
        except Exception as e:
            logger.warning(f"Scoring error: {file_path} | {e}")

    except Exception as e:
        logger.error(f"FATAL ERROR: {file_path} | {e}")
        output["error"] = str(e)

    logger.info(f"Done: {file_path} in {time.time() - start_time:.2f}s")

    return output


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def get_pdf_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]

    pdfs = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))

    return pdfs


# -----------------------------
# BATCH PROCESS (SAFE)
# -----------------------------
def process_batch(file_paths: List[str]) -> List[Dict]:
    results = []

    for i, path in enumerate(file_paths):
        logger.info(f"[{i + 1}/{len(file_paths)}]")
        results.append(process_single_cv(path))

    return results


# -----------------------------
# SAVE OUTPUT
# -----------------------------
def save_results(results: Union[List[Dict], Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved → {output_path}")


# -----------------------------
# MAIN
# -----------------------------
def main(input_path: str, output_path: str):

    pdf_files = get_pdf_files(input_path)

    if not pdf_files:
        logger.error("No PDFs found")
        return

    logger.info(f"Total PDFs: {len(pdf_files)}")

    results = process_batch(pdf_files)

    save_results(results, output_path)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="output.json")

    args = parser.parse_args()

    main(args.input, args.output)
