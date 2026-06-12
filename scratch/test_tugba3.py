import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\tugba zengin.pdf")
cv_parser8._debug = True
raw, _ = cv_parser8.extract_text_pdf(str(pdf_path))
clean = cv_parser8.clean_text(cv_parser8.fix_ocr_spacing(cv_parser8.normalize_column_spacing(raw)), 'en')
print("--- CLEAN LINES ---")
for i, line in enumerate(clean.split('\n')):
    print(f"{i}: {line}")
s = cv_parser8.extract_sections(clean)
print("\n--- EXTRACT SECTIONS OUTPUT ---")
print("LANGUAGES:", repr(s.get('languages', '')))
print("SKILLS:", repr(s.get('skills', '')))
