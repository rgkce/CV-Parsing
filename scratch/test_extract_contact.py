import sys
import os
sys.path.insert(0, os.path.abspath("."))
from cv_parser8 import extract_contact_info, _RE_LINKEDIN
import json

sena_raw = """SENA NUR MORBEL
Fkr geltrmenn zhn, dolaysyla beden srekl aktf tuttuunun blncnde
olarak hem  hem de breysel hayatta yaratc olmann neml olduunu
dnyorum.
Aratrmay seven, yenlklere ak, abuk renen ve rendn aktarmaktan
zevk duyan br kle sahbm.
Takm ersnde teknk ve sosyal anlamda kolay uyum salayacam
MEKATRONK
dnyorum.
MHENDS
ETM GEM
letm Blgler
AFYON KOCATEPE NVERSTES
MEKATRONK MHENDSL | 2018-2022
ZEL FNAL LSES
2018 MEZUNU
+90 533 470 7542
ALIMA GEM
OPTMUM MEKATRONK - STAJYER TASARIM MHENDS
TEMMUZ 2021 - EYLL 2021
3 BOYUTLU MODELLEME, MALAT YNTEMLER, OK BLEENL
Mansa, Turkey MAKNE VE MEKANZMALARIN TASARIMI, TEKNK RESMLERNN
IKARILMASI, HAREKETL MONTAJ KONULARINDA ALITIM.
AYDINOLU MAKNE - STAJYER MEKATRONK MHENDSL
UBAT 2021 - MART 2021
senamorbel42@gmal.com 2 VE 3 BOYUTLU TASARIM, MAKNE RETM, TORNA, FREZE VE CNC
LE TALALI MALAT KONULARINDA ALIMALAR YAPTIM.
s.morbel@mekatronk.org.tr
BECERLER
SOLIDWORKS
lnkedn.com/n/senamorbel
CATIA
AUTOCAD
ANSYS MECHANICAL
B C PROGRAMLAMA
PYTHON
Dller
PROTEUS
Trke
nglzce DENEYMLER
Almanca
MEKATRONK MHENDSLER DERNE
ETM KOMSYONU YES | TEMMUZ 2020 - HALEN
lg Alanlar
AFYON KOCATEPE NVERSTES MEKATRONK
MHENDSL KULB
YNETM KURULU YES | EKM 2019 - HALEN"""

giz_raw = """GİZEM KILINÇ
Endüstri Mühendisi
Tel: 0531 271 03 27
E-mail: gzmkilinncc@gmail.com
LinkedIn: www.linkedin.com/in/gizemkılınç
HAKKIMDA
Eskişehir Osmangazi Üniversitesi Endüstri Mühendisliği 4. sınıf öğrencisiyim. İmalat ve yönetim stajlarıyla makine imalat
süreçleri, veri yönetimi, süreç iyileştirme ve SolidWorks tabanlı teknik çizim konularında deneyim kazandım. Şu anda Benli
Geri Dönüşüm ’de Arena üzerinden süreç geliştirme ve RACI matrisi oluşturma üzerine bitirme projesi yürütmekteyim. Üretim
ve yönetim süreçlerine bütüncül bakabilen, analitik yönü güçlü bir mühendis adayıyım."""

print("SENA NUR CONTACT:")
print(json.dumps(extract_contact_info(sena_raw), indent=2))
print("GİZEM CONTACT:")
print(json.dumps(extract_contact_info(giz_raw), indent=2))
