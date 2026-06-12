import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def print_safe(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        safe_msg = msg.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(safe_msg)

root_json = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json"
script_json = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json"

with open(root_json, "r", encoding="utf-8") as f:
    root_data = json.load(f)

with open(script_json, "r", encoding="utf-8") as f:
    script_data = json.load(f)

root_map = {os.path.basename(c["file_path"]).lower(): c for c in root_data}
script_map = {os.path.basename(c["file_path"]).lower(): c for c in script_data}

all_keys = set(root_map.keys()) | set(script_map.keys())

diff_count = 0
for filename in sorted(all_keys):
    if filename not in root_map:
        print_safe(f"[DIFF] {filename} is missing from root JSON!")
        diff_count += 1
        continue
    if filename not in script_map:
        print_safe(f"[DIFF] {filename} is missing from script JSON!")
        diff_count += 1
        continue
        
    c_root = root_map[filename]
    c_script = script_map[filename]
    
    # Compare fields
    cand_diffs = []
    
    # 1. Compare top-level contact fields
    for field in ["name", "email", "phone"]:
        v_root = c_root.get("contact", {}).get(field)
        v_script = c_script.get("contact", {}).get(field)
        if v_root != v_script:
            cand_diffs.append(f"contact.{field}: '{v_root}' vs '{v_script}'")
            
    # 2. Compare sections
    sec_root = c_root.get("sections", {})
    sec_script = c_script.get("sections", {})
    for sec_key in ["summary", "title", "years_of_experience", "skills", "languages", "certificates", "interests", "organizations", "other"]:
        v_root = sec_root.get(sec_key, "")
        v_script = sec_script.get(sec_key, "")
        if v_root != v_script:
            cand_diffs.append(f"sections.{sec_key}: '{v_root}' vs '{v_script}'")
            
    if cand_diffs:
        print_safe(f"\n--- Diffs for {filename} ---")
        for d in cand_diffs:
            print_safe(f"  * {d}")
        diff_count += 1

print_safe(f"\nTotal differing candidates: {diff_count}")
