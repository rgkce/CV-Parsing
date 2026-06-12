import sys
import logging
from pathlib import Path

sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
cv_parser8.logger.setLevel(logging.INFO)

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\sudenaz boyalı.pdf")
if not pdf_path.exists():
    # Maybe the filename is sudenaz boyali.pdf
    pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\sudenaz boyali.pdf")

record = cv_parser8.process_cv(pdf_path)

print("\n=== FINAL ALL ===")
for k, v in record.items():
    if k in ["summary", "title", "years_of_experience", "experience", "education", "skills", "projects", "languages", "certificates", "interests", "organizations", "other"]:
        print(f"[{k}]: {repr(v)}")

raw, _ = cv_parser8.extract_text_pdf(str(pdf_path))
print("\n--- RAW TEXT ---")
print(raw[:1500])
