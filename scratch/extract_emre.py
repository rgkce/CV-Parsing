import sys
sys.path.append("cv-parser-script")
from cv_parser8 import extract_text_pdf

pdf_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\emre celik.pdf"
raw_text, cleaned_text = extract_text_pdf(pdf_path)
print("=== RAW TEXT ===")
print(raw_text)
print("\n=== CLEANED TEXT ===")
print(cleaned_text)
