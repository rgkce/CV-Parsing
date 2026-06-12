import json

with open("cv-parser-script/final_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

targets = ["emre celik", "ahmet berat bulduk", "arda gungor"]
records = {}

for r in data:
    fp = r.get("file_path", "").lower()
    for t in targets:
        if t in fp:
            records[t] = r

for name, r in records.items():
    print(f"=== {name.upper()} ===")
    print(f"File Path: {r.get('file_path')}")
    print(f"Title: {r.get('sections', {}).get('title')}")
    print(f"Years of Experience: {r.get('sections', {}).get('years_of_experience')}")
    print(f"Contact Info: {json.dumps(r.get('contact'), indent=2)}")
    print(f"Other Section:\n{r.get('sections', {}).get('other')}")
    print(f"Experience Section:\n{r.get('sections', {}).get('experience')}")
    print(f"Certificates Section:\n{r.get('sections', {}).get('certificates')}")
    print("=" * 60)
