import sys
from pathlib import Path
sys.path.append(r'c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script')
import cv_parser8

raw = cv_parser8.normalize_text(cv_parser8.extract_text_pdf(Path(r'c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\zeliha ayca aydemir.pdf'))[0])

import inspect
source = inspect.getsource(cv_parser8.extract_title_and_experience)
source = source.replace('title = _title_candidate_fallback', 'print("Assigned fallback:", repr(_title_candidate_fallback)); title = _title_candidate_fallback')
source = source.replace('title = potential_title', 'print("Assigned from line 1:", repr(potential_title)); title = potential_title')
source = source.replace('title = l\n', 'print("Assigned from regex match:", repr(l)); title = l\n')
source = source.replace('title = parts[1]', 'print("Assigned parts[1]:", repr(parts[1])); title = parts[1]')

with open('temp_module.py', 'w', encoding='utf-8') as f:
    f.write('import re\nimport datetime\n')
    f.write('from cv_parser8 import *\n')
    f.write(source)

sys.path.insert(0, '')
import temp_module

title, years = temp_module.extract_title_and_experience(raw, '', '')
print("FINAL RETURNED TITLE:", repr(title))
