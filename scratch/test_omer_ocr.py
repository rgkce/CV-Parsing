import json, sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\omer bugra karakoc.pdf")
print("Testing OCR on omer...")
res = cv_parser8.process_cv(pdf_path, force_ocr=True)

for k, v in res.items():
    print(f"[{k}]: {repr(v)}")
