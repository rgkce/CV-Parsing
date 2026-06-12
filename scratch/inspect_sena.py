import pdfplumber

file_path = r"C:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\sena nur morbel.pdf"
with pdfplumber.open(file_path) as pdf:
    for i, page in enumerate(pdf.pages):
        print(f"--- PAGE {i+1} ---")
        text = page.extract_text()
        print(repr(text))
