import sys
from pathlib import Path
sys.path.append(r'c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script')
import cv_parser8
import re

raw = cv_parser8.normalize_text(cv_parser8.extract_text_pdf(Path(r'c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\zeliha ayca aydemir.pdf'))[0])
print('Raw line 1:', raw.splitlines()[0])
print('Raw line 2:', raw.splitlines()[1])

contact_info = cv_parser8.extract_contact_info(raw)
raw_text = raw

if contact_info.get("email"):
    raw_text = raw_text.replace(contact_info["email"], "")
    raw_text = cv_parser8._RE_EMAIL.sub(" ", raw_text)

raw_text = cv_parser8._RE_PHONE_CONTACT.sub(" ", raw_text)

print('Masked line 1:', raw_text.splitlines()[0])
print('Masked line 2:', raw_text.splitlines()[1])

title, years = cv_parser8.extract_title_and_experience(raw_text, '', '')
print('Extracted title from masked text:', repr(title))
