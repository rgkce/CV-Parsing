import json, sys, glob
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_paths = list(Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF").glob("*ömer buğra*.pdf"))
if not pdf_paths:
    pdf_paths = list(Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF").glob("*omer bugra*.pdf"))

if not pdf_paths:
    print("CV NOT FOUND")
    sys.exit(1)

pdf_path = pdf_paths[0]
print(f"Testing {pdf_path}")
result = cv_parser8.process_cv(pdf_path)

s = result.get("sections", {})
print("\n=== FINAL SECTIONS ===")
for k, v in s.items():
    if v.strip():
        print(f"[{k}]: {repr(v)}")
