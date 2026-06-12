import os
import json

root_json = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json"
script_json = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"

with open(root_json, "r", encoding="utf-8") as f:
    root_data = json.load(f)

with open(script_json, "r", encoding="utf-8") as f:
    script_data = json.load(f)

if json.dumps(root_data, sort_keys=True) == json.dumps(script_data, sort_keys=True):
    print("Verification Success: Root and Script final_dataset.json files are identical!")
else:
    print("Verification Failure: Files differ!")
