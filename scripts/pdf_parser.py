import pdfplumber
import os
import json
from tqdm import tqdm
import re

# --------------------------------------
# Proje klasörünü baz alarak yollar
# --------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))  # script'in bulunduğu klasör
pdf_folder = os.path.join(base_dir, "..", "data", "PDF")  # PDF dosyaları burada olmalı
output_folder = os.path.join(base_dir, "output")  # JSON çıktısı bu klasöre
output_file = os.path.join(output_folder, "pdf_parsed_clean.json")

# Çıktı klasörü yoksa oluştur
os.makedirs(output_folder, exist_ok=True)


# --------------------------------------
# Metin temizleme fonksiyonu
# --------------------------------------
def clean_text(text):
    """
    Texti temizler:
    - Noktalama işaretleri kaldır
    - Özel karakterleri kaldır
    - Tüm metni küçük harfe çevirir
    """
    text = re.sub(r"[^\w\s]", " ", text)  # sadece harf ve boşluk bırak
    text = re.sub(r"\s+", " ", text)  # fazla boşlukları tek boşluk yap
    text = text.lower()  # küçük harfe çevir
    return text.strip()  # baş ve sondaki boşlukları temizle


# --------------------------------------
# PDF'leri oku ve temizle
# --------------------------------------
results = []

for i, file in enumerate(tqdm(os.listdir(pdf_folder))):
    if file.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file)
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception as e:
            print(f"Hata: {file} → {e}")
            continue

        cleaned_text = clean_text(text)

        results.append(
            {"resume_id": str(i + 1), "file_name": file, "raw_text": cleaned_text}
        )

# --------------------------------------
# JSON çıktısını klasöre yaz
# --------------------------------------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"PDF parsing ve temizleme tamamlandı. Çıktı dosyası: {output_file}")
