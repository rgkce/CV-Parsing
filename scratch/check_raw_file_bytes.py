import json

with open("final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "gizem" in item["file_path"].lower():
        title = item["sections"]["title"]
        print("Title text:", title)
        print("Title codepoints:", [ord(c) for c in title])
        linkedin = item["contact"]["linkedin"]
        print("Linkedin text:", linkedin)
        print("Linkedin codepoints:", [ord(c) for c in linkedin])
        break
