import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ulasli = None
for candidate in data:
    if "ulasli" in candidate.get("file_path", "").lower():
        ulasli = candidate
        break

if ulasli:
    print(json.dumps(ulasli, indent=2, ensure_ascii=False))
else:
    print("Not found")
