import json

with open("final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for idx, item in enumerate(data):
    linkedin = item.get("contact", {}).get("linkedin", "")
    if linkedin:
        name = item.get("file_path", "").rsplit("\\", 1)[-1]
        print(f"Index {idx:2d}: {name:<30} -> {linkedin}")
