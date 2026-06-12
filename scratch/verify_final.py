import json
from pathlib import Path

# Load compiled final_dataset.json
json_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json")
if not json_path.exists():
    print("final_dataset.json does not exist yet at:", json_path)
    exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Find Gökdeniz Can
gokdeniz = None
for candidate in data:
    if "gokdeniz can" in candidate.get("file_path", "").lower():
        gokdeniz = candidate
        break

if gokdeniz:
    print("--- Gökdeniz Can ---")
    print("Title:", gokdeniz.get("sections", {}).get("title"))
    print("Years of Experience:", gokdeniz.get("sections", {}).get("years_of_experience"))
    print("Experience length:", len(gokdeniz.get("sections", {}).get("experience", "")))
    print("Success status:", gokdeniz.get("source_format") != "failed")
else:
    print("Gökdeniz Can not found in dataset!")

# Check other Phase 3 candidates as regression tests
regression_candidates = [
    ("gizem kilinc", "Gizem Kılınç"),
    ("sena nur morbel", "Sena Nur Mörbel"),
    ("yagiz tokgoz", "Yağız Tokgöz"),
    ("irem sude uslu", "İrem Sude Uslu"),
    ("burcu kuzucu", "Burcu Kuzucu")
]

print("\n--- Regression Checks ---")
for slug, name in regression_candidates:
    match = None
    for candidate in data:
        if slug in candidate.get("file_path", "").lower():
            match = candidate
            break
    if match:
        print(f"{name}: Title={match.get('sections', {}).get('title')}, YoE={match.get('sections', {}).get('years_of_experience')}, Success={match.get('source_format') != 'failed'}")
    else:
        print(f"{name} not found in dataset!")
