import pdfplumber

file_path = r"C:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gizem kilinc.pdf"
with pdfplumber.open(file_path) as pdf:
    page = pdf.pages[0]
    chars = page.chars
    for idx, char in enumerate(chars[:150]):
        text = char.get("text", "")
        # print text, its unicode code point, and details
        code_points = [ord(c) for c in text]
        print(f"Char #{idx}: text={repr(text)} codepoints={code_points} fontname={char.get('fontname')} size={char.get('size'):.2f}")
