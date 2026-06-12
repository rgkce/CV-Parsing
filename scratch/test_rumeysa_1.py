import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\rumeysa gokce 1.pdf")
record = cv_parser8.process_cv(pdf_path)

print("\n=== FINAL SECTIONS ===")
for sec, content in record.get("sections", {}).items():
    print(f"[{sec}]: {repr(content)}")

print(f"\nTitle: {repr(record.get('sections', {}).get('title', ''))}")
print(f"Language: {repr(record.get('language', ''))}")
