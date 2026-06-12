import sys
import logging
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path
import pprint

cv_parser8.logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
cv_parser8.logger.addHandler(handler)

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\vedat acat.pdf")
cv_parser8._debug = True

print("=== RAW TEXT ===")
raw, _ = cv_parser8.extract_text_pdf(str(pdf_path))
print(raw)

print("\n=== CLEAN TEXT ===")
clean = cv_parser8.clean_text(cv_parser8.fix_ocr_spacing(cv_parser8.normalize_column_spacing(raw)), 'tr')
for i, line in enumerate(clean.split('\n')):
    print(f"{i}: {repr(line)}")

print("\n=== FINAL SECTIONS ===")
record = cv_parser8.process_cv(pdf_path)
for k in ["summary", "skills", "projects", "organizations", "languages"]:
    print(f"[{k}]: {repr(record['sections'].get(k))}")
