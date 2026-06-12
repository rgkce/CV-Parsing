import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\omer bugra karakoc.pdf")
raw_text, _ = cv_parser8.extract_text_pdf(str(pdf_path))
raw_text = cv_parser8.sanitize_raw_text(raw_text)

lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
print("LINES:")
for i, l in enumerate(lines[:30]):
    print(f"{i}: {l}")
