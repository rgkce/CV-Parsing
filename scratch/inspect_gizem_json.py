import json

with open("final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "gizem" in item["file_path"].lower():
        print("NAME:", item["file_path"])
        print("TITLE:", repr(item["sections"]["title"]))
        print("EDUCATION:", repr(item["sections"]["education"]))
        print("LINKEDIN:", repr(item["contact"]["linkedin"]))
        print("SKILLS:", repr(item["sections"]["skills"]))
        print("CERTIFICATES:", repr(item["sections"]["certificates"]))
        print("OTHER:", repr(item["sections"]["other"]))
        print("EMAIL:", repr(item["contact"]["email"]))
        break
