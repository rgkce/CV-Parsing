import json
import re

with open("final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

turkish_chars = re.compile(r"[ıİğĞüşŞöÖçÇ]")

print("Scanning for LinkedIn links with Turkish characters:")
found_any = False
for idx, item in enumerate(data):
    linkedin = item.get("contact", {}).get("linkedin", "")
    if linkedin and turkish_chars.search(linkedin):
        name = item.get("file_path", "").rsplit("\\", 1)[-1]
        print(f"Index {idx}: File={name}, LinkedIn={repr(linkedin)}")
        found_any = True

if not found_any:
    print("None found!")
