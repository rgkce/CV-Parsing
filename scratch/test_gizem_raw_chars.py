import pdfplumber

file_path = r"C:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gizem kilinc.pdf"
with pdfplumber.open(file_path) as pdf:
    text = pdf.pages[0].extract_text()
    first_line = text.split("\n")[0]
    print("First line:", repr(first_line))
    print("Codepoints:", [ord(c) for c in first_line])
    
    second_line = text.split("\n")[1]
    print("Second line:", repr(second_line))
    print("Codepoints:", [ord(c) for c in second_line])
