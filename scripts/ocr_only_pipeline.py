import os
import json
import re
from tqdm import tqdm
from pdf2image import convert_from_path
import pytesseract

# 🔴 Tesseract path (Windows için zorunlu)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 📁 Klasör yolları
PDF_FOLDER = r"C:\Users\rumeysagokce\Desktop\cv_parser_project\data\PDF"
OUTPUT_FILE = (
    r"C:\Users\rumeysagokce\Desktop\cv_parser_project\scripts\output\ocr_dataset.json"
)


# 🧹 Text temizleme
def clean_text(text):
    text = re.sub(r"[^\w\s]", " ", text)  # noktalama kaldır
    text = re.sub(r"\s+", " ", text)  # fazla boşlukları sil
    text = text.lower()  # küçük harf
    return text.strip()


# 🔍 OCR fonksiyonu
def ocr_pdf(file_path):
    text = ""

    try:
        images = convert_from_path(file_path)

        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, lang="tur+eng")
            text += page_text + " "

    except Exception as e:
        print(f"OCR hata: {file_path} → {e}")

    return text


# ⚠️ Basit kalite kontrol
def check_quality(text):
    if len(text) < 100:
        return "low_text"
    if len(text.split()) < 30:
        return "low_word_count"
    return "ok"


# 🚀 Ana pipeline
dataset = []

files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

for i, file in enumerate(tqdm(files)):
    file_path = os.path.join(PDF_FOLDER, file)

    # OCR ile oku
    raw_text = ocr_pdf(file_path)

    # Temizle
    cleaned_text = clean_text(raw_text)

    # Kalite kontrol
    quality = check_quality(cleaned_text)

    dataset.append(
        {
            "resume_id": str(i + 1),
            "file_path": file_path,
            "raw_text": cleaned_text,
            "source_format": "ocr_only",
            "quality_flag": quality,
        }
    )

# 💾 Kaydet
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("OCR dataset oluşturuldu:", OUTPUT_FILE)
