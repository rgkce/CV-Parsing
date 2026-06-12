import json, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for candidate in data:
    fp = candidate.get("file_path", "").lower()
    if "omer bugra karakoc" in fp or "ömer buğra karakoç" in fp.replace("ö", "o").replace("ğ", "g").replace("ç", "c"):
        s = candidate.get("sections", {})
        print("=== TITLE ===")
        print(repr(s.get("title", "")))
        print("\n=== PROJECTS ===")
        print(repr(s.get("projects", "")))
        print("\n=== RAW TEXT (first 2000 chars) ===")
        print(repr(candidate.get("raw_text", "")[:2000]))
        print("\n=== RAW TEXT (full) ===")
        print(candidate.get("raw_text", ""))
        break
