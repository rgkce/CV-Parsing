import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\omer bugra karakoc.pdf")
raw_text, _ = cv_parser8.ocr_fallback(str(pdf_path))
print("RAW TEXT FROM OCR:")
print(repr(raw_text))
