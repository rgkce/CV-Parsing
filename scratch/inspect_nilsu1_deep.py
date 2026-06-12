import json, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for candidate in data:
    fp = candidate.get("file_path", "").lower()
    if "nilsu ak 1" in fp:
        s = candidate.get("sections", {})
        print("Title:", repr(s.get("title", "")))
        print("Education:", repr(s.get("education", "")))
        print("Summary:", repr(s.get("summary", "")))
        print("Raw text first 500:", repr(candidate.get("raw_text", "")[:500]))
        break
