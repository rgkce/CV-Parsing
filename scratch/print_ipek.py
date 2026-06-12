import json
from pathlib import Path

json_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for c in data:
    if "ipek" in c.get("file_path", "").lower() and "sarialp" in c.get("file_path", "").lower():
        print(json.dumps(c, ensure_ascii=False, indent=2))
        break
