import sys
from pathlib import Path

# Add cv-parser-script to sys.path
sys.path.append(str(Path(__file__).parent.parent / "cv-parser-script"))
from cv_parser8 import process_cv

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gokdeniz can.pdf")
res = process_cv(pdf_path)

print("Contact email:", res.get("contact", {}).get("email"))
