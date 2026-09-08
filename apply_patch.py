import re

with open('cv_parser_script/cv_parser8.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update SECTION_KEYWORDS (Targeting specific arrays)
def add_to_list(match):
    return match.group(1) + match.group(2) + match.group(3)

content = re.sub(
    r'("experience":\s*\[)(.*?)(\s*\])', 
    lambda m: m.group(1) + m.group(2).rstrip() + ',\n        "deneyimler",\n        "stajlar",\n        "staj deneyimleri",\n' + m.group(3).lstrip(), 
    content, flags=re.DOTALL
)

content = re.sub(
    r'("education":\s*\[)(.*?)(\s*\])', 
    lambda m: m.group(1) + m.group(2).rstrip() + ',\n        "egitimler",\n        "eğitimler",\n        "akademik geçmiş",\n        "akademik gecmis",\n' + m.group(3).lstrip(), 
    content, flags=re.DOTALL
)

content = re.sub(
    r'("projects":\s*\[)(.*?)(\s*\])', 
    lambda m: m.group(1) + m.group(2).rstrip() + ',\n        "projelerim",\n        "projelerimiz",\n' + m.group(3).lstrip(), 
    content, flags=re.DOTALL
)

content = re.sub(
    r'("skills":\s*\[)(.*?)(\s*\])', 
    lambda m: m.group(1) + m.group(2).rstrip() + ',\n        "yeteneklerim",\n' + m.group(3).lstrip(), 
    content, flags=re.DOTALL
)


# 2. Update _is_text_broken
old_broken = """    # 1. Check for the replacement character (garbage)
    if text.count('\\ufffd') > 0:
        logger.info("  [broken_check] Detected too many replacement characters.")
        return True"""
new_broken = """    # 1. Check for the replacement character (garbage)
    if text.count('\\ufffd') > 0 or text.count('\\u01ec') > 0 or 'Ǭ' in text or '' in text:
        logger.info("  [broken_check] Detected too many replacement/mojibake characters.")
        return True"""
content = content.replace(old_broken, new_broken)

# 3. Update clean_skills to strip star ratings
old_skills = """        if _AS_PARA_RE.search(line) and _AS_SENTENCE_END.search(line.strip()):
            spill_to_summary.append(line)
        else:
            clean_skills.append(line)"""
new_skills = """        if _AS_PARA_RE.search(line) and _AS_SENTENCE_END.search(line.strip()):
            spill_to_summary.append(line)
        else:
            # Strip rating patterns like "* * * * x" or "o o o o +"
            cleaned_skill = re.sub(r'^(?:[\\*xXoO\\-+]\\s*){3,}', '', line).strip()
            if cleaned_skill:
                clean_skills.append(cleaned_skill)"""
content = content.replace(old_skills, new_skills)

# 4. Update repair_broken_emails
old_repair = """        if len(parts) > 1:
            first_token = parts[0].strip().rstrip("_:;.,-|")
            if first_token.lower() in {"sj", "lo", "q", "e", "o"}:"""
new_repair = """        if len(parts) > 1:
            first_token = parts[0].strip().rstrip("_:;.,-|")
            if len(first_token) <= 2 or first_token.lower() in {"sj", "lo", "q", "e", "o", "ba", "s", "m", "mm", "mmee", "mmee."}:"""
content = content.replace(old_repair, new_repair)

with open('cv_parser_script/cv_parser8.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch applied.")
