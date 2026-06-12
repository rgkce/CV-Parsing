import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path
import pprint

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\tugba zengin.pdf")
cv_parser8._debug = True
record = cv_parser8.process_cv(pdf_path)

print("\n=== RAW TEXT ===")
print(record['raw_text'])

print("\n=== FINAL ALL ===")
for k in ["skills", "languages", "other"]:
    print(f"[{k}]: {repr(record['sections'].get(k))}")
