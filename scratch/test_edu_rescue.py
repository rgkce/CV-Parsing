import re

summary = """eskişehir osmangazi üniversitesi endüstri mühendisliği 4. sinif öğrencisiyim. imalat ve yönetim stajlariyla makine imalat
süreçleri, veri yönetimi, süreç iyileştirme ve solidworks tabanli teknik çizim konularında deneyim kazandım. şu anda benli
geri dönüşüm 'de arena üzerinden süreç geliştirme ve racı matrisi oluşturma üzerine bitirme projesi yürütmekteyim. üretim
ve yönetim süreçlerine bütüncül bakabilen, analitik yönü güçlü bir mühendis adayiyim."""

_edu_lines = []
for line in summary.split("\n"):
    sentences = re.split(r'(?<!\d)\.\s+', line)
    for sentence in sentences:
        s_clean = sentence.strip().rstrip('.')
        if not s_clean:
            continue
        if any(kw in s_clean.lower() for kw in ["üniversite", "universite", "okumaktayım", "okumaktayim", "öğrenci", "ogrenci", "öğrenim"]):
            _edu_lines.append(s_clean + '.')

print("Rescued education:")
print("\n".join(_edu_lines))
