import json
import re
from pathlib import Path

json_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\final_dataset.json")
output_report = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\scratch\audit_report_clean.txt")

if not json_path.exists():
    print("final_dataset.json does not exist!")
    exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

lines = []
lines.append(f"Total candidates: {len(data)}")

# 1. Check for generic/empty/suspicious titles
lines.append("\n=== 1. Suspicious/Generic/Empty Titles ===")
for i, c in enumerate(data):
    title = c.get("sections", {}).get("title", "")
    file_name = Path(c.get("file_path", "")).name
    
    if not title.strip() or title.lower() in ["öğrenci", "ogrenci", "student"]:
        lines.append(f"[{i+1}] {file_name}: Generic/Empty title -> '{title}'")
    elif any(k in title.lower() for k in ["staj", "intern", "astsubay", "barista", "kasiyer"]):
        lines.append(f"[{i+1}] {file_name}: Suspicious title containing student/military keywords -> '{title}'")

# 2. Check for YoE anomalies
lines.append("\n=== 2. YoE Anomalies ===")
for i, c in enumerate(data):
    yoe = c.get("sections", {}).get("years_of_experience", "0")
    exp = c.get("sections", {}).get("experience", "")
    file_name = Path(c.get("file_path", "")).name
    
    # 0 YoE but has non-internship experience
    if yoe == "0" and len(exp.strip()) > 30:
        # Check if the experience contains staj/intern keywords. If not, it's very suspicious!
        if not re.search(r'\b(staj|stajyer[a-z]*|staj[ıi][a-z]*|intern|interns|internship|trainee|trainees)\b', exp.lower()):
            lines.append(f"[{i+1}] {file_name}: YoE is '0' but experience section does not mention internships!")
            lines.append(f"   Exp: {exp[:200]}...")
            
    # YoE is exceptionally high (> 8)
    try:
        yoe_int = int(yoe)
        if yoe_int > 8:
            lines.append(f"[{i+1}] {file_name}: Exceptionally high YoE -> {yoe_int}")
            lines.append(f"   Exp: {exp[:200]}...")
    except ValueError:
        pass

# 3. Check for potential language parsing omissions
lines.append("\n=== 3. Language Parsing Omissions ===")
for i, c in enumerate(data):
    langs = c.get("sections", {}).get("languages", "")
    raw = c.get("raw_text", "")
    file_name = Path(c.get("file_path", "")).name
    
    if not langs.strip():
        lang_indicators = ["ingilizce", "english", "almanca", "deutsch", "yabancı dil", "diller", "languages", "türkçe", "turkish"]
        found = [w for w in lang_indicators if w in raw.lower()]
        if found:
            # Check if raw lines look like language declarations
            lang_lines = []
            for line in raw.split('\n'):
                if any(w in line.lower() for w in ["ingilizce", "english", "almanca", "deutsch", "türkçe", "turkish", "yabancı dil", "diller"]):
                    if len(line.strip()) < 80:
                        lang_lines.append(line.strip())
            if lang_lines:
                lines.append(f"[{i+1}] {file_name}: Languages section is empty, but raw text contains language keywords: {found}")
                lines.append(f"   Raw lines match: {lang_lines}")

# 4. Check for leaked reference keywords in non-experience/non-summary sections
lines.append("\n=== 4. Leaked References ===")
ref_keywords = ["referans", "reference", "istek üzerine", "upon request", "uam references", "uum references", "references"]
for i, c in enumerate(data):
    file_name = Path(c.get("file_path", "")).name
    for sec_name, sec_val in c.get("sections", {}).items():
        if sec_name in ["experience", "summary", "other", "contact_discard"]:
            continue
        if isinstance(sec_val, str) and any(k in sec_val.lower() for k in ref_keywords):
            lines.append(f"[{i+1}] {file_name}: Possible leaked reference in '{sec_name}' -> '{sec_val}'")

# Write report to file
with open(output_report, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Clean audit report written successfully to {output_report.name}")
