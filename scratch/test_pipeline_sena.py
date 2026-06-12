import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from pathlib import Path
from cv_parser8 import extract_text_pdf, parse_cv

file_path = Path(r"C:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\sena nur morbel.pdf")
raw_text, clean_txt = extract_text_pdf(str(file_path))
print("SOURCE FORMAT:", "ocr" if "ocr" in clean_txt else "pdf")
print("RAW TEXT:")
print(repr(raw_text[:1000]))

result = parse_cv(raw_text)
print("PARSED CONTACT:")
print(json.dumps(result.get("contact", {}), indent=2, ensure_ascii=False))
