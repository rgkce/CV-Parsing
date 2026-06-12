import json
import sys

# Reconfigure stdout to use utf-8 if possible
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ulasli = None
for candidate in data:
    if "ulasli" in candidate.get("file_path", "").lower():
        ulasli = candidate
        break

if ulasli:
    print("--- Fatih Ulaşlı Parsed Data ---")
    print(f"File Path: {ulasli.get('file_path')}")
    print(f"Name: {ulasli.get('contact', {}).get('name')}")
    print(f"Email: {ulasli.get('contact', {}).get('email')}")
    print(f"Phone: {ulasli.get('contact', {}).get('phone')}")
    print(f"YoE: {ulasli.get('sections', {}).get('years_of_experience')}")
    print(f"Skills: {ulasli.get('sections', {}).get('skills')}")
    print(f"Interests (Hobbies): {ulasli.get('sections', {}).get('interests')}")
    print(f"Experience: {ulasli.get('sections', {}).get('experience')}")
else:
    print("Fatih Ulaşlı not found in final_dataset.json!")
