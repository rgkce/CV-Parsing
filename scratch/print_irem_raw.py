import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for candidate in data:
    fp = candidate.get("file_path", "").lower()
    if "irem celik" in fp:
        print("=" * 60)
        print(f"File Path: {candidate.get('file_path')}")
        print("Raw text:")
        print(candidate.get("raw_text"))
