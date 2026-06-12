import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for candidate in data:
    fp = candidate.get("file_path", "").lower()
    if "nilsu ak 1" in fp:
        print("Raw Languages in JSON:")
        print(repr(candidate.get("sections", {}).get("languages")))
        print("Raw text in JSON:")
        print(repr(candidate.get("raw_text")))
