import json
import os

root_json = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json"
script_json = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"

if not os.path.exists(root_json):
    print("Root final_dataset.json does not exist!")
    root_data = []
else:
    with open(root_json, "r", encoding="utf-8") as f:
        root_data = json.load(f)

if not os.path.exists(script_json):
    print("Script final_dataset.json does not exist!")
    script_data = []
else:
    with open(script_json, "r", encoding="utf-8") as f:
        script_data = json.load(f)

print(f"Root dataset length: {len(root_data)}")
print(f"Script dataset length: {len(script_data)}")

# Compare specific fields for Fatih Ulaşlı
def find_ulasli(data):
    for c in data:
        if "ulasli" in c.get("file_path", "").lower():
            return c
    return None

ulasli_root = find_ulasli(root_data)
ulasli_script = find_ulasli(script_data)

if ulasli_root:
    print("\n--- Fatih Ulaşlı in Root JSON ---")
    print(f"YoE: {ulasli_root.get('sections', {}).get('years_of_experience')}")
    print(f"Skills: {ulasli_root.get('sections', {}).get('skills')}")
    print(f"Interests: {ulasli_root.get('sections', {}).get('interests')}")
else:
    print("\nFatih Ulaşlı not found in Root JSON")

if ulasli_script:
    print("\n--- Fatih Ulaşlı in Script JSON ---")
    print(f"YoE: {ulasli_script.get('sections', {}).get('years_of_experience')}")
    print(f"Skills: {ulasli_script.get('sections', {}).get('skills')}")
    print(f"Interests: {ulasli_script.get('sections', {}).get('interests')}")
else:
    print("\nFatih Ulaşlı not found in Script JSON")
