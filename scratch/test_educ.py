import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\sudenaz boyali.pdf")
raw, _ = cv_parser8.extract_text_pdf(str(pdf_path))
clean = cv_parser8.clean_text(cv_parser8.fix_ocr_spacing(cv_parser8.normalize_column_spacing(raw)), 'tr')

print("CLEAN TEXT LINES:")
lines = clean.split('\n')
for i, line in enumerate(lines):
    if "beyazıt" in line:
        print(f"Line {i}: {repr(line)}")

cv_parser8._debug = True
print("--- RUNNING EXTRACT SECTIONS ---")
s = cv_parser8.extract_sections(clean)
for k, v in s.items():
    if "beyazıt" in str(v):
        print(f"Section {k} has beyazit")
