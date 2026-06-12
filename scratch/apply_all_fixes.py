import sys
from pathlib import Path

f_path = Path(r'c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\cv_parser8.py')
code = f_path.read_text('utf-8')

# 1. Yağız Tokgöz: Add foreign language
code = code.replace(
    '"foreign languages",',
    '"foreign languages",\n        "foreign language",'
)
code = code.replace(
    '"yabancı diller": "languages",',
    '"yabancı diller": "languages",\n    "foreign language": "languages",'
)
code = code.replace(
    '"yabancı diller": ("languages", "Yabancı Diller"),',
    '"yabancı diller": ("languages", "Yabancı Diller"),\n        "foreign language": ("languages", "Foreign Language"),'
)

# 2. Yağız Tokgöz: Fix Title Fallback scanning past headings
old_title_logic = '''            # Skip lines that look like section headings
            if l_lower in _TITLE_SKIP_HEADINGS:
                continue'''
new_title_logic = '''            # Skip lines that look like section headings
            if l_lower in _TITLE_SKIP_HEADINGS or _sd_norm(l) in _SD_EXT_MAP:
                break'''
if old_title_logic in code:
    code = code.replace(old_title_logic, new_title_logic)
else:
    print("WARNING: Could not find old_title_logic")

# 3. Vedat Acat: Add projeler ve başarımlar, topluluk ve aktiviteler
code = code.replace(
    '"projects": ["projeler", "projects", "proje", "projelerim", "akademik projeler", "academic projects", "personal projects", "kişisel projeler"],',
    '"projects": ["projeler", "projects", "proje", "projelerim", "akademik projeler", "academic projects", "personal projects", "kişisel projeler", "projeler ve başarımlar"],'
)
code = code.replace(
    '"organizations": ["kulüpler", "clubs", "sosyal", "gönüllü", "volunteer", "dernek", "vakıf", "topluluk", "organizations", "faaliyetler", "okul dışı faaliyetler"],',
    '"organizations": ["kulüpler", "clubs", "sosyal", "gönüllü", "volunteer", "dernek", "vakıf", "topluluk", "organizations", "faaliyetler", "okul dışı faaliyetler", "topluluk ve aktiviteler", "topluluk ve aktıvıteler"],'
)

code = code.replace(
    '"kişisel projeler": "projects",',
    '"kişisel projeler": "projects",\n    "projeler ve başarımlar": "projects",'
)
code = code.replace(
    '"okul dışı faaliyetler": "organizations",',
    '"okul dışı faaliyetler": "organizations",\n    "topluluk ve aktiviteler": "organizations",\n    "topluluk ve aktıvıteler": "organizations",'
)

code = code.replace(
    '"mezuniyet projesi": ("projects", "Mezuniyet Projesi"),',
    '"mezuniyet projesi": ("projects", "Mezuniyet Projesi"),\n        "projeler ve başarımlar": ("projects", "Projeler Ve Başarımlar"),'
)

# 4. Tuğba/Vedat: Fix Upper-Intermediate language extraction
old_regex = r"m_level = re.search(r'\b([abc][12]|fluent|advanced|intermediate|beginner|akıcı|akici|iyi|orta|ileri|seviye)\b', l_lower)"
new_regex = r"m_level = re.search(r'\b([abc][12]|fluent|advanced|upper[\s\-]?intermediate|upper[\s\-]?\u0131ntermediate|intermediate|\u0131ntermediate|beginner|akıcı|akici|iyi|orta|ileri|seviye)\b', l_lower)"
if old_regex in code:
    code = code.replace(old_regex, new_regex)
else:
    print("WARNING: Could not find m_level regex")

old_lvl_logic = '''                if m_level and m_level.group(1) != m_lang.group(1).lower() and current_lang:
                    lang_levels[current_lang].append(m_level.group(1).upper())'''
new_lvl_logic = '''                if m_level and m_level.group(1) != m_lang.group(1).lower() and current_lang:
                    lvl = m_level.group(1).title()
                    if lvl.lower().startswith("upper"):
                        lvl = "Upper-Intermediate"
                    lang_levels[current_lang].append(lvl)'''
if old_lvl_logic in code:
    code = code.replace(old_lvl_logic, new_lvl_logic)
else:
    print("WARNING: Could not find lvl logic")

f_path.write_text(code, 'utf-8')
print("Successfully applied all fixes safely.")
