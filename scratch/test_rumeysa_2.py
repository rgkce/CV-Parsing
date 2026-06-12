import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\rumeysa gokce 2.pdf")
record = cv_parser8.process_cv(pdf_path)

print("\n=== FINAL SKILLS ===")
print(repr(record.get("sections", {}).get("skills", "")))

print("\n=== FINAL ALL ===")
for sec, content in record.get("sections", {}).items():
    print(f"[{sec}]: {repr(content)}")

# Print raw layout output to see what the layout parser saw
with __import__("pdfplumber").open(pdf_path) as pdf:
    words = pdf.pages[0].extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False, use_text_flow=True)
    text = cv_parser8._extract_two_column(pdf.pages[0], words)
    print("\n--- RAW TEXT ---")
    print(text)
