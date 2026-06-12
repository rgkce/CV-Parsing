import sys
import re
sys.path.append("cv-parser-script")
from cv_parser8 import extract_text_pdf, repair_broken_emails, normalize_text, extract_contact_info

pdf_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\emre celik.pdf"
raw_text, cleaned_text = extract_text_pdf(pdf_path)

print("Step 2a: Raw text length:", len(raw_text))

raw_text_repaired = repair_broken_emails(raw_text, debug=True)
print("Step 2b: Repaired raw text length:", len(raw_text_repaired))

raw_text_norm = normalize_text(raw_text_repaired)
print("Step 2c: Normalized raw text length:", len(raw_text_norm))

print("\n--- Running extract_contact_info on raw_text_norm ---")
contact = extract_contact_info(raw_text_norm)
print("Contact extracted:", contact)
