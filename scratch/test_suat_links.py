import sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import pdfplumber

pdf_path = r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\suat bilgay.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for p in pdf.pages:
        print(f"Page {p.page_number} hyperlinks:")
        for h in p.hyperlinks:
            print("  ", h.get("uri"))
