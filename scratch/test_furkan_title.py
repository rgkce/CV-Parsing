import json, sys, os
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
pdf_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\muhammed furkan ozcan mekatronik.pdf"

print("Testing single CV extraction...")
result = cv_parser8.process_cv(Path(pdf_path))

s = result.get("sections", {})
print("\n=== FINAL PARSED TITLE ===")
print(repr(s.get("title", "")))
print("\n=== FINAL YoE ===")
print(repr(s.get("years_of_experience", "")))
