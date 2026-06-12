import json, sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8

raw_text = """İŞ GEÇMİŞİ VE DENEYİMLER
KAMERAMAN
Akademx TV - Isparta - Aralık 2023 - Günümüz
Haber ve Program Kameramanlığı Yaptığım Projeler:
1.Ulusal PİBEX Fkr Maratonu
Lve From Fest Isparta
2023-2024 SDÜ Mezunyet Tören
Selfy Fest Isparta
TEKNOFEST Türkye Drone Şampyonası 2024 Isparta
TEKNOFEST Akdenz 2024
TEKNOFEST Adana 2024
Drone Operatörlüğü Yaptığım Projeler:
Lve From Fest Isparta
2023-2024 SDÜ Mezunyet Tören
selfy Fest Isparta
TEKNİK SORUMLU
Akademx TV - Isparta - Mart 2023 - Günümüz
İşe başladığım tarhten tbaren kendm şrketmde çalışmalarım
ve düzenmle göstererek bu ünvana yükselmey hak ettm.
LİNK ÜZERİNDEN ÇEKİMLERİME ULAŞABİLİRSİNİZ:
ÇEKİMLERİN LİSTESİ
FOTOĞRAFÇILIK
Elfefe Fotoğrafçılık - Isparta - Aralık 2023 - Günümüz
Burada kendme ekstra gelr olması ve kendm daha hızlı
gelştrmek çn çalışıyorum. Düğün çekmler, klp çekmler ve
fotoğrafçılık şler yapıyorum.
REFERANSLAR
Mert KARACAN
Akademx TV Koordnatörü
+90 505 701 11 13"""

try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

sections = cv_parser8.extract_sections(raw_text.lower(), debug=True)
print("\nEXTRACTED SECTIONS:")
for k, v in sections.items():
    if isinstance(v, str) and v.strip():
        print(f"[{k}]: {repr(v)}")
