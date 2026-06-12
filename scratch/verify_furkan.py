import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for name_key in ["ulasli", "furkan ozcan", "furkan özcan"]:
    print("=" * 60)
    print(f"Checking matches for: {name_key}")
    found = False
    for candidate in data:
        fp = candidate.get("file_path", "").lower()
        if name_key in fp:
            found = True
            print(f"File Path: {candidate.get('file_path')}")
            print(f"YoE: {candidate.get('sections', {}).get('years_of_experience')}")
            print(f"Title: {candidate.get('sections', {}).get('title')}")
            print(f"Skills: {candidate.get('sections', {}).get('skills')[:150]}...")
            print(f"Interests: {candidate.get('sections', {}).get('interests')}")
    if not found:
        print("No match found")
