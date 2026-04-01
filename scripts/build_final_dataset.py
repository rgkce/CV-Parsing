import json
import os

# 📁 input dosyaları
PDF_DATA = r"C:\Users\rumeysagokce\Desktop\cv_parser_project\scripts\output\pdf_parsed_clean.json"
DOCX_DATA = r"C:\Users\rumeysagokce\Desktop\cv_parser_project\scripts\output\docx_parsed_clean.json"
OCR_DATA = (
    r"C:\Users\rumeysagokce\Desktop\cv_parser_project\scripts\output\ocr_dataset.json"
)

# 📁 output
FINAL_OUTPUT = (
    r"C:\Users\rumeysagokce\Desktop\cv_parser_project\scripts\output\final_dataset.json"
)


# 📥 load fonksiyonu
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


pdf_data = load_json(PDF_DATA)
docx_data = load_json(DOCX_DATA)
ocr_data = load_json(OCR_DATA)


# 🧠 kalite hesaplama
def score_text(text):
    if not text:
        return 0
    length_score = len(text)
    word_score = len(text.split()) * 2
    return length_score + word_score


# 🔄 indexleme (file_name bazlı)
def index_by_filename(data):
    index = {}
    for item in data:
        file_name = os.path.basename(item.get("file_path", item.get("file_name", "")))
        index[file_name] = item
    return index


pdf_index = index_by_filename(pdf_data)
docx_index = index_by_filename(docx_data)
ocr_index = index_by_filename(ocr_data)


# 📂 tüm dosyaları topla
all_files = set(pdf_index.keys()) | set(docx_index.keys()) | set(ocr_index.keys())

final_dataset = []

# 🚀 seçim algoritması
for i, file_name in enumerate(all_files):
    candidates = []

    if file_name in pdf_index:
        candidates.append(("pdfplumber", pdf_index[file_name]))

    if file_name in docx_index:
        candidates.append(("docx", docx_index[file_name]))

    if file_name in ocr_index:
        candidates.append(("ocr", ocr_index[file_name]))

    best = None
    best_score = -1
    best_source = None

    for source, item in candidates:
        text = item.get("raw_text", "")
        score = score_text(text)

        if score > best_score:
            best_score = score
            best = item
            best_source = source

    final_dataset.append(
        {
            "resume_id": str(i + 1),
            "file_path": best.get("file_path", ""),
            "raw_text": best.get("raw_text", ""),
            "source_format": best_source,
        }
    )


# 💾 kaydet
with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

print("Final dataset hazır:", FINAL_OUTPUT)
