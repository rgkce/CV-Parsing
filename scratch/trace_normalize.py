import sys
from pathlib import Path
import re

# Add cv-parser-script to sys.path
sys.path.append(str(Path(__file__).parent.parent / "cv-parser-script"))
from cv_parser8 import ocr_fallback, normalize_text, extract_contact_info

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gokdeniz can.pdf")
raw_text, format_str = ocr_fallback(pdf_path)

print("Before normalize_text:")
# Search for lines containing "gmail" in raw_text
for line in raw_text.splitlines():
    if "gmail" in line:
        print("  Line:", repr(line))

normalized = normalize_text(raw_text)

print("After normalize_text:")
# Search for lines containing "gmail" in normalized
for line in normalized.splitlines():
    if "gmail" in line:
        print("  Line:", repr(line))

contact = extract_contact_info(normalized)
print("Extracted contact from normalized:", contact)
