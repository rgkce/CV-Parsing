import json, sys
from pathlib import Path
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8

data_dir = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF")
output_file = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json")

all_results = {}
for pdf in data_dir.glob("*.pdf"):
    print(f"Processing {pdf.name}...")
    try:
        all_results[pdf.name] = cv_parser8.process_cv(pdf)
    except Exception as e:
        print(f"Error on {pdf.name}: {e}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Successfully processed {len(all_results)} CVs.")
