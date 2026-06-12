import json, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for candidate in data:
    fp = candidate.get("file_path", "").lower()
    if "muhammed furkan ozcan mekatronik" in fp:
        s = candidate.get("sections", {})
        print("=== TITLE ===")
        print(repr(s.get("title", "")))
        print("\n=== EDUCATION ===")
        print(repr(s.get("education", "")))
        print("\n=== SUMMARY ===")
        print(repr(s.get("summary", "")))
        print("\n=== EXPERIENCE ===")
        print(repr(s.get("experience", "")))
        print("\n=== RAW TEXT (first 800 chars) ===")
        print(repr(candidate.get("raw_text", "")[:800]))
        print("\n=== RAW TEXT (full) ===")
        print(candidate.get("raw_text", ""))
        break
