import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\omer bugra karakoc.pdf")
raw_text, _ = cv_parser8.extract_text_pdf(str(pdf_path))
print("RAW TEXT:")
print(repr(raw_text))

t = raw_text.lower()
import re
dropped_i_words = [
    r"\bünverste", r"\bdeneym", r"\bçekm\b", r"\bşler\b", 
    r"\bşrket", r"\bkendme\b", r"\bsnema", r"\btelevzyon", 
    r"\beğtm", r"\bblg\b", r"\blg\b", r"öğrencsym", r"etmekteym", 
    r"teknklerm", r"çekmler", r"gelr\b", r"\bbr\b", r"\bçn\b"
]
dropped_pattern = "|".join(dropped_i_words)
dropped_matches = re.findall(dropped_pattern, t)
print(f"\nDropped pattern: {dropped_pattern}")
print(f"Matches found: {dropped_matches}")
