import json

with open("cv-parser-script/final_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

for r in data:
    fp = r.get("file_path", "")
    email = r.get("contact", {}).get("email", "")
    phone = r.get("contact", {}).get("phone", "")
    print(f"{fp:<50} | {email:<35} | {phone}")
