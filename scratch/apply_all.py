import sys
from pathlib import Path

f_path = Path(r'c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\cv_parser8.py')
code = f_path.read_text('utf-8')

# 1 & 2 & 3: Mappings
# _HEADING_DICT
code = code.replace(
    '"projects": ["projeler", "projects", "proje", "projelerim", "akademik projeler", "academic projects", "personal projects", "kişisel projeler"],',
    '"projects": ["projeler", "projects", "proje", "projelerim", "akademik projeler", "academic projects", "personal projects", "kişisel projeler", "projeler ve başarımlar"],'
)
code = code.replace(
    '"organizations": ["kulüpler", "clubs", "sosyal", "gönüllü", "volunteer", "dernek", "vakıf", "topluluk", "organizations", "faaliyetler", "okul dışı faaliyetler"],',
    '"organizations": ["kulüpler", "clubs", "sosyal", "gönüllü", "volunteer", "dernek", "vakıf", "topluluk", "organizations", "faaliyetler", "okul dışı faaliyetler", "topluluk ve aktiviteler", "topluluk ve aktıvıteler"],'
)
code = code.replace(
    '"foreign languages",',
    '"foreign languages",\n        "foreign language",'
)

# _SD_EXT_MAP
code = code.replace(
    '"kişisel projeler": "projects",',
    '"kişisel projeler": "projects",\n    "projeler ve başarımlar": "projects",'
)
code = code.replace(
    '"okul dışı faaliyetler": "organizations",',
    '"okul dışı faaliyetler": "organizations",\n    "topluluk ve aktiviteler": "organizations",\n    "topluluk ve aktıvıteler": "organizations",'
)
code = code.replace(
    '"yabancı diller": "languages",',
    '"yabancı diller": "languages",\n    "foreign language": "languages",'
)

# SUB_HEADERS
code = code.replace(
    '"mezuniyet projesi": ("projects", "Mezuniyet Projesi"),',
    '"mezuniyet projesi": ("projects", "Mezuniyet Projesi"),\n        "projeler ve başarımlar": ("projects", "Projeler Ve Başarımlar"),'
)
code = code.replace(
    '"yabancı diller": ("languages", "Yabancı Diller"),',
    '"yabancı diller": ("languages", "Yabancı Diller"),\n        "foreign language": ("languages", "Foreign Language"),'
)

# 4: Regex update for ıntermediate
old_regex = r'''        r'(native|ana\s*dil|fluent|advanced|intermediate|\u0131ntermediate'
        r'|beginner|upper[\s\-]?intermediate|pre[\s\-]?intermediate'
        r'|[abc][12]|orta|iyi|ileri|başlangıç|baslangic|temel'''
new_regex = r'''        r'(native|ana\s*dil|fluent|advanced|intermediate|\u0131ntermediate'
        r'|beginner|upper[\s\-]?intermediate|upper[\s\-]?\u0131ntermediate|pre[\s\-]?intermediate|pre[\s\-]?\u0131ntermediate'
        r'|[abc][12]|orta|iyi|ileri|başlangıç|baslangic|temel'''
code = code.replace(old_regex, new_regex)

# 5: Level extraction logic
old_logic = '''            elif "orta" in l_lower or "intermediate" in l_lower or "b1" in l_lower or "b2" in l_lower:
                level_str = "Intermediate" if is_english_cv else "Orta Düzey"'''
new_logic = '''            elif "upper" in l_lower or "iyi" in l_lower or "b2" in l_lower:
                level_str = "Upper-Intermediate" if is_english_cv else "İyi Düzey"
            elif "orta" in l_lower or "intermediate" in l_lower or "ıntermediate" in l_lower or "b1" in l_lower:
                level_str = "Intermediate" if is_english_cv else "Orta Düzey"'''
code = code.replace(old_logic, new_logic)

# 6: Title fallback logic
old_title_logic = '''    if not sections["title"]:
        title_candidates = []
        for i, (line, is_h) in enumerate(parsed_lines[:20]): # Look in first 20 lines
            if is_h: continue'''
new_title_logic = '''    if not sections["title"]:
        title_candidates = []
        for i, (line, is_h) in enumerate(parsed_lines[:20]): # Look in first 20 lines
            if is_h: break'''
code = code.replace(old_title_logic, new_title_logic)

f_path.write_text(code, 'utf-8')
print('Restored and applied ALL fixes successfully.')
