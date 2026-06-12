import json
from pathlib import Path

json_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

targets = ["arda gungor", "ilteris cansiz", "ipek nur sarialp", "suat bilgay", "yagiz tokgoz", "yuksel cosgun Backend"]

print("=== Empty Languages Section Analysis ===")
for candidate in data:
    fp = candidate.get("file_path", "").lower()
    match_name = next((t for t in targets if t in fp), None)
    if match_name:
        sections = candidate.get("sections", {})
        print(f"\n--- {Path(candidate['file_path']).name} ---")
        print("Languages:", repr(sections.get("languages")))
        print("Skills   :", repr(sections.get("skills")))
        print("Summary  :", repr(sections.get("summary")[:150] + "..."))
        print("Other    :", repr(sections.get("other")[:150] + "..."))
        print("Education:", repr(sections.get("education")[:150] + "..."))
