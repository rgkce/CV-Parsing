import json

json_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for candidate in data:
    fp = candidate.get("file_path", "").lower()
    if "furkan ozcan" in fp or "furkan özcan" in fp:
        print("=" * 60)
        print(f"File Path: {candidate.get('file_path')}")
        print("Raw text snippet:")
        print(candidate.get("raw_text")[:1500])
