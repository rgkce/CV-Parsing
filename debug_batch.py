import json

data = json.load(open('final_dataset.json', encoding='utf-8'))
targets = ['ebrar kadir cetin', 'elif bilgili', 'esra yaman', 'esranur cinar', 'hakan ozguler', 'hasan can gul', 'kaan olmez', 'koray ozturk', 'mustafa narin', 'ozan ahmet dede', 'rumeysa gokce', 'sena yildiz', 'sevval salman', 'yusuf yaman', 'zeynep tugsem camlica', 'zeynep tugsem camlıca']

with open('debug_batch.txt', 'w', encoding='utf-8') as f:
    for item in data:
        if any(t in item['file_path'].lower() for t in targets):
            f.write(f'\n\n=== {item["file_path"]} ===\n')
            f.write(f'SECTIONS: {json.dumps(item["sections"], ensure_ascii=False, indent=2)}\n')
            f.write(f'EMAILS: {item.get("contact", {}).get("emails", [])}\n')
            f.write(f'YEARS_EXP: {item.get("years_of_experience", 0)}\n')
            f.write(f'RAW:\n{item["raw_text"][:1000]}\n...\n{item["raw_text"][-1000:]}\n')
