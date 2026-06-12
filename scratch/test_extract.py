import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\sudenaz boyali.pdf")
raw, _ = cv_parser8.extract_text_pdf(str(pdf_path))
clean = cv_parser8.clean_text(cv_parser8.fix_ocr_spacing(cv_parser8.normalize_column_spacing(raw)), 'tr')

cv_parser8._debug = True
print("--- RUNNING EXTRACT SECTIONS ---")
s = cv_parser8.extract_sections(clean)
