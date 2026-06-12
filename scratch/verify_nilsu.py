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
    if "nilsu ak" in fp:
        print("=" * 60)
        print(f"File Path: {candidate.get('file_path')}")
        print(f"YoE: {candidate.get('sections', {}).get('years_of_experience')}")
        print(f"Title: {candidate.get('sections', {}).get('title')}")
        print(f"Languages: {candidate.get('sections', {}).get('languages')}")
        print(f"Skills: {candidate.get('sections', {}).get('skills')}")
        print(f"Other (Volunteering): {candidate.get('sections', {}).get('other')}")
        print(f"Experience: {candidate.get('sections', {}).get('experience')}")
        print(f"Organizations: {candidate.get('sections', {}).get('organizations')}")
