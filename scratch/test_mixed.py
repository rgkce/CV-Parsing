import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8
from pathlib import Path
import re

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\rumeysa gokce 2.pdf")
with __import__("pdfplumber").open(pdf_path) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
        
mixed = re.findall(r"[a-z][A-Z][a-z]", text)
print("Mixed matches:", mixed)
