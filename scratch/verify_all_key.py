import json, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

targets = ["irem celik", "nilsu ak 1", "nilsu ak 2", "muhammed furkan ozcan mekatronik", 
           "muhammed furkan ozcan.pdf", "muhammed fatih ulasli"]

for candidate in data:
    fp = candidate.get("file_path", "").lower()
    stem = fp.replace("\\", "/").split("/")[-1]
    
    for t in targets:
        if t in stem:
            s = candidate.get("sections", {})
            print("=" * 60)
            print(f"File: {stem}")
            print(f"  Title: {s.get('title', '')}")
            print(f"  YoE: {s.get('years_of_experience', '')}")
            print(f"  Languages: {s.get('languages', '')}")
            print(f"  Interests: {s.get('interests', '')[:120]}")
            print(f"  Skills (first 120): {s.get('skills', '')[:120]}")
            break
