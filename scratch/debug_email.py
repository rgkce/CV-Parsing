import sys
from pathlib import Path
import re

# Add cv-parser-script to sys.path
sys.path.append(str(Path(__file__).parent.parent / "cv-parser-script"))
from cv_parser8 import extract_contact_info, process_cv, _RE_EMAIL, ocr_fallback

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gokdeniz can.pdf")
raw_text, format_str = ocr_fallback(pdf_path)

print("--- Original Raw Text (first 500 chars) ---")
print(raw_text[:500])
print("----------------")

contact_search_text = raw_text

_has_valid_email_already = _RE_EMAIL.search(contact_search_text)
print("Has valid email already:", bool(_has_valid_email_already))

if _has_valid_email_already:
    print("Direct match:", _has_valid_email_already.group(0))
    email_search_text = contact_search_text
else:
    email_search_text = re.sub(
        r"([A-Za-z0-9._%+\-])\s+@\s+([A-Za-z0-9])", r"\1@\2", contact_search_text
    )

email_search_text = re.sub(
    r"([A-Za-z0-9])\s*\.\s*(com|net|org|edu|gov|info|online|site|link|app|dev|me|io|co|tr|in|biz|[a-z]{2})(?=\s|$|[,;\)])",
    r"\1.\2",
    email_search_text,
    flags=re.I
)

print("--- Preprocessed Email Search Text ---")
# Find line containing "@" in email_search_text
for line in email_search_text.splitlines():
    if "@" in line:
        print("Line with @:", repr(line))

print("----------------")
email_match = _RE_EMAIL.search(email_search_text)
if email_match:
    print("Match:", repr(email_match.group(0)))
else:
    print("No email match found in preprocessed text!")
