import json
from pathlib import Path

# Load compiled final_dataset.json from root and cv-parser-script
root_json = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json")
script_json = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json")

def check_file(json_path):
    if not json_path.exists():
        print(f"File {json_path} does not exist!")
        return False
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    gorkem = None
    for candidate in data:
        if "görkem" in candidate.get("file_path", "").lower() or "gorkem" in candidate.get("file_path", "").lower():
            gorkem = candidate
            break
            
    if not gorkem:
        print(f"Görkem Tolu not found in {json_path.name}!")
        return False
        
    sections = gorkem.get("sections", {})
    title = sections.get("title")
    yoe = sections.get("years_of_experience")
    langs = sections.get("languages")
    exp = sections.get("experience")
    
    print(f"\n--- Görkem Tolu in {json_path.name} ---")
    print("Title:", title)
    print("Years of Experience:", yoe)
    print("Languages:", repr(langs))
    print("Experience:\n", exp)
    
    success = True
    if title != "Makine Teknikeri":
        print("FAIL: Title should be 'Makine Teknikeri'!")
        success = False
    if yoe != "2":
        print("FAIL: Years of Experience should be '2'!")
        success = False
    if "İngilizce - A2" not in langs:
        print("FAIL: Languages should include 'İngilizce - A2'!")
        success = False
    if "Türkçe" not in langs:
        print("FAIL: Languages should include 'Türkçe'!")
        success = False
        
    if success:
        print("SUCCESS: All assertions passed!")
    return success

print("Verifying script folder JSON:")
check_file(script_json)

print("\nVerifying root folder JSON:")
check_file(root_json)
