import sys
from pathlib import Path
import re

# Add cv-parser-script to sys.path
sys.path.append(str(Path(__file__).parent.parent / "cv-parser-script"))
from cv_parser8 import ocr_fallback, normalize_text, extract_contact_info

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gokdeniz can.pdf")
raw_text, format_str = ocr_fallback(pdf_path)

raw_text_norm = normalize_text(raw_text)
contact = extract_contact_info(raw_text_norm)
print("Immediately after extract_contact_info:", contact.get("email"))

# Wait! Does contact["email"] get modified?
# Let's check if contact["email"] is modified anywhere in the overrides in cv_parser8.py
# Let's inspect the code of cv_parser8.py around lines 7479 to 9500 to see if there is any other place.
# Let's print the actual process_cv call's contact:
from cv_parser8 import process_cv
res = process_cv(pdf_path)
print("Final processed contact email:", res.get("contact", {}).get("email"))
