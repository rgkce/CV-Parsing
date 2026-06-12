import sys
import re
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\beyza aktas.pdf")

# We will instrument process_cv locally by reproducing it and printing
file_path = pdf_path
file_path_str = str(file_path)
raw_text, source_format = cv_parser8.extract_text_pdf(file_path_str)
raw_text = cv_parser8.sanitize_raw_text(raw_text)
original_raw = raw_text

# Protect email
_email_placeholders = {}
def _protect_email_for_split(m):
    place = f"__EMAIL_PLACEHOLDER_{len(_email_placeholders)}__"
    _email_placeholders[place] = m.group(0)
    return place
raw_text_protected = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', _protect_email_for_split, raw_text) if hasattr(cv_parser8, 're') else raw_text

import re
raw_text_protected = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', _protect_email_for_split, raw_text)
# We will just run cv_parser8.process_cv but with custom prints if possible, or run the steps manually:
print("\n--- Running steps manually ---")
cleaned_text = cv_parser8.clean_text(cv_parser8.normalize_column_spacing(raw_text))
print("1. After clean_text:")
sections_raw = cv_parser8.extract_sections(cleaned_text, debug=False)
sections = sections_raw
print("  education:", repr(sections.get("education")))

# step 6b
try:
    _structured = cv_parser8.parse_cv(cleaned_text)
    print("  structured education:", repr(_structured.get("education")))
    # override?
    _kw_val = sections.get("education", "")
    _st_val = _structured.get("education", "")
    if not _kw_val and _st_val:
        sections["education"] = _st_val
except Exception as e:
    print("  structured failed:", e)

print("2. After step 6b:")
print("  education:", repr(sections.get("education")))

# final fixes
_final_fixes = cv_parser8._final_fixes if hasattr(cv_parser8, '_final_fixes') else []
_RE_BULLET_CLEAN = re.compile(r"^[a-zçğıöşü•▪▫\-\*\+·~]\s+", re.I)

for k, v in list(sections.items()):
    if k != "education":
        continue
    if isinstance(v, str):
        v = v.replace("===COLUMN_BREAK===", "").replace(" \n", "\n").replace("\n\n\n", "\n\n")
        v_lines = []
        for line in v.split("\n"):
            line_clean = line.strip()
            if k == "education" and line_clean.lower() == "lise":
                continue
            while True:
                next_line = _RE_BULLET_CLEAN.sub("", line_clean).strip()
                if next_line == line_clean:
                    break
                line_clean = next_line
            v_lines.append(line_clean)
        sections[k] = "\n".join(v_lines).strip()

print("3. After final fixes:")
print("  education:", repr(sections.get("education")))

# step 8a contact strip
_header_lines_to_strip = set()
if raw_text:
    for _hl in raw_text.split("\n")[:6]:
        _hl_clean = _hl.strip().lower()
        if not _hl_clean:
            continue
        if cv_parser8._is_section_heading(_hl_clean):
            break
        if len(_hl_clean) > 50:
            continue
        _header_lines_to_strip.add(_hl_clean)
print("  _header_lines_to_strip:", _header_lines_to_strip)

val = sections.get("education", "")
clean_lines = []
for line in val.split("\n"):
    line_stripped = line.strip()
    line_lower = line_stripped.lower()
    if line_lower in _header_lines_to_strip:
        # Our new check:
        if re.search(r"\b(mühendis|muhendis|bölüm|bolum|lisans|okul|lise|üniversite|university|faculty|fakülte)\b", line_lower):
            pass
        else:
            print("  Stripping header duplicate:", repr(line))
            continue
    clean_lines.append(line)
sections["education"] = "\n".join(clean_lines).strip()

print("4. After Step 8a contact strip:")
print("  education:", repr(sections.get("education")))

# step 9 - correct typos
for sec_key in ["education"]:
    if isinstance(sections[sec_key], str):
        sections[sec_key] = cv_parser8.correct_turkish_ocr_typos(sections[sec_key])

print("5. After correct_turkish_ocr_typos:")
print("  education:", repr(sections.get("education")))

# step 10 - unified clean/rescue
# Let's run step 2 of rescue education
education = sections.get("education", "").strip()
if education:
    edu_lines = []
    skills_rescued = []
    certs_rescued = []
    for line in education.split("\n"):
        line_clean = line.strip()
        line_lower = line_clean.lower()
        if not line_clean:
            continue
        # Rescue Certificates
        if any(k in line_lower for k in ["sertifika", "certificate", "katılım belgesi", "katilim belgesi", "kursu", "eğitimi", "egitimi"]):
            certs_rescued.append(line_clean)
            continue
        # Rescue Skills
        if any(k in line_lower for k in [
            "react", "kotlin", "java", "python", "javascript", "html", "css", "sql", "git", "c#", "c++", 
            "figma", "photoshop", "shopify", "dropshipping", "fastapi", "docker", "excel", "word"
        ]) and ("," in line_clean or len(line_clean.split()) > 3):
            skills_rescued.append(line_clean)
            continue
        edu_lines.append(line_clean)
    sections["education"] = "\n".join(edu_lines).strip()

print("6. After rescue education:")
print("  education:", repr(sections.get("education")))
