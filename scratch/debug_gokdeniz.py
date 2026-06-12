import sys
from pathlib import Path
import json

# Add cv-parser-script to sys.path
sys.path.append(str(Path(__file__).parent.parent / "cv-parser-script"))
from cv_parser8 import extract_title_and_experience, process_cv

# Find Gökdeniz's PDF
pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gokdeniz can.pdf")
print("PDF exists:", pdf_path.exists())

# Process the CV and print its output
res = process_cv(pdf_path)
print("Title:", res.get("sections", {}).get("title"))
print("Years of Experience:", res.get("sections", {}).get("years_of_experience"))
print("Experience Text:\n", res.get("sections", {}).get("experience"))
