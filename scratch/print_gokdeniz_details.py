import json
from pathlib import Path

json_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for c in data:
    if "gokdeniz can" in c.get("file_path", "").lower():
        print("Contact info:", c.get("contact"))
        break
