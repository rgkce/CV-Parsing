import sys
import json
sys.path.append("cv-parser-script")
from cv_parser8 import process_cv
from pathlib import Path

file_path = Path("data/PDF/emre celik.pdf")
record = process_cv(file_path)
print("=== PROCESS_CV EMRE CELIK ===")
print("Contact Info:", json.dumps(record.get("contact"), indent=2))
print("Other Sections:", json.dumps(record.get("sections"), indent=2))
