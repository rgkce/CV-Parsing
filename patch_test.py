import re

with open('cv_parser_script/cv_parser8.py', 'r', encoding='utf-8') as f:
    content = f.read()

count1 = content.count('"deneyim",')
count2 = content.count('replacement_count = text.count("\\ufffd")\n        if replacement_count > 5:')
count3 = content.count('        if _AS_PARA_RE.search(line) and _AS_SENTENCE_END.search(line.strip()):\n            spill_to_summary.append(line)\n        else:\n            clean_skills.append(line)')
count4 = content.count('        if len(parts) > 1:\n            first_token = parts[0].strip().rstrip("_:;.,-|")\n            if first_token.lower() in {"sj", "lo", "q", "e", "o"}:')

print(f'Match counts: 1:{count1} 2:{count2} 3:{count3} 4:{count4}')
