import sys
import logging
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

cv_parser8.logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
cv_parser8.logger.addHandler(handler)

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\tugba zengin.pdf")
cv_parser8._debug = True
raw, _ = cv_parser8.extract_text_pdf(str(pdf_path))
clean = cv_parser8.clean_text(cv_parser8.fix_ocr_spacing(cv_parser8.normalize_column_spacing(raw)), 'en')
s = cv_parser8.extract_sections(clean)
