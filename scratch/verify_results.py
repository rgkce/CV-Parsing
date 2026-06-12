import json
import os

with open("final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

targets = ["gizem kilinc", "sena nur morbel", "yagiz tokgoz", "irem sude uslu", "burcu kuzucu"]

print("=== VERIFYING TARGET CANDIDATES ===")
for target in targets:
    found = False
    for item in data:
        if target in item["file_path"].replace("-", " ").lower():
            name = item["file_path"].rsplit("\\", 1)[-1]
            print(f"Candidate: {name}")
            print(f"  Title: {repr(item['sections']['title'])}")
            print(f"  Education: {repr(item['sections']['education'])}")
            print(f"  LinkedIn: {repr(item['contact']['linkedin'])}")
            print(f"  Skills: {repr(item['sections']['skills'][:150])}...")
            print()
            found = True
            break
    if not found:
        print(f"Target '{target}' NOT found in dataset!\n")
