import json

with open("final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for idx, item in enumerate(data):
    if "gizem" in item["file_path"].lower() or "kilin" in item["file_path"].lower():
        print(f"Index: {idx}")
        print(f"File Path: {item['file_path']}")
        print(f"Source Format: {item['source_format']}")
        print("Contact:")
        print(json.dumps(item["contact"], indent=2, ensure_ascii=False))
        print("Sections:")
        for k, v in item["sections"].items():
            print(f"  {k}: {repr(v[:200])}...")
        print("Raw text:")
        print(item["raw_text"][:500])
        print("="*40)
