import os
import json
import re
from tqdm import tqdm
from docx import Document

# --------------------------------------
# Proje klasörünü baz alarak yollar
# --------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
docx_folder = os.path.abspath(
    os.path.join(base_dir, "..", "data", "DOCX")
)  # DOCX dosyaları burada
output_folder = os.path.join(base_dir, "output")
output_file = os.path.join(output_folder, "docx_parsed_clean.json")

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
# DOCX dosyalarını oku ve temizle
# --------------------------------------
results = []

for i, file in enumerate(tqdm(os.listdir(docx_folder))):
    if file.endswith(".docx"):
        file_path = os.path.join(docx_folder, file)
        text = ""
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text:
                    text += para.text + " "
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

print(f"DOCX parsing ve temizleme tamamlandı. Çıktı dosyası: {output_file}")
