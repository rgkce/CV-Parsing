# CV Parser ve Akıllı Aday Değerlendirme Sistemi - Sunum Rehberi

Bu belge, projeyi hocanıza sunarken adım adım sistemi nasıl anlatabileceğiniz konusunda size rehberlik etmesi için hazırlanmıştır. Sistemin 4 ana aşamasını (Milestone) sırasıyla anlatabilir ve aşağıda yer alan örnek çıktıları göstererek projenin ne kadar başarılı çalıştığını kanıtlayabilirsiniz.

---

## 📍 1. Giriş ve Proje Amacı (Kısa Özet)
"Hocam, bu proje klasik kelime bazlı CV tarama sistemlerinin çok ötesine geçen, uçtan uca otonom bir İnsan Kaynakları (İK) asistanıdır. Sistem, herhangi bir formattaki (ister tek sütun, ister tablo, ister tarayıcıdan geçmiş fotoğraf) PDF CV'sini alıyor, anlamsal olarak analiz ediyor ve verdiğimiz iş tanımına (Job Description - JD) göre en uygun adayları sıralayarak yapay zeka destekli bir rapor sunuyor."

---

## 📍 2. Milestone 1 & 2: CV İşleme (PDF'ten JSON'a)

**Ne Anlatılmalı?**
"İlk aşamamız veri çıkarımı. Piyasada her adayın CV formatı farklıdır. Bu yüzden sistemimiz şu adımları izliyor:
1. **Layout (Düzen) Analizi:** Sistem sayfanın tek sütun mu yoksa iki sütun mu olduğunu anlıyor (Gap Analysis). Sütunları karıştırmadan doğru sırayla okuyor.
2. **OCR Fallback:** Eğer PDF taranmış bir resimse veya bozuk karakterlere sahipse, sistem bunu otomatik algılayıp `Tesseract OCR` devreye sokuyor. Görüntüden metin okuma yapıyor.
3. **Section Extraction:** Elde edilen karmaşık metni Eğitim, Deneyim, Beceriler gibi 10 farklı kategoriye ayırıyor. Bizim geliştirdiğimiz akıllı kural motoru (heuristic algorithm), tasarım boşluklarından dolayı 'üniversite' ile 'bölüm' farklı satırlarda kalsa bile metin örüntülerini analiz ederek bunları birleştirmeyi başarıyor. URL'leri ve kişisel websitelerini silmeden koruyor."

**Göstermeniz Gereken Çıktı Örneği:**
> *"Hocam, bu sürecin sonunda sistem binlerce farklı CV'yi standart bir JSON formatına dönüştürüyor. Örnek bir çıktı şu şekildedir:"*

data/PDF/ahmet berat bulduk.pdf
final_dataset.json

```json
{
  "name": "Ahmet Berat Bulduk",
  "contact": {
    "email": "beratbulduk6@gmail.com",
    "phone": "+90 553 332 4366",
    "linkedin": "https://www.linkedin.com/in/beratbulduk/",
    "website": ""
  },
  "sections": {
    "education": "2018 - 2022, inşaat mühendisliği, afyon kocatepe üniversitesi",
    "experience": "Temmuz 2022 - Şuanda\nAfyon Alin Yapı Denetim\nİnşaat Mühendisi\nŞantiye ve proje uygulamalarında denetmen olarak görev yapmaktayım...",
    "skills": "autocad sta4cad idecad sap2000"
  },
  "profile_photo": true,
  "language": "tr",
  "source_format": "ocr"
}
```

---

## 📍 3. Milestone 3: Semantik (Anlamsal) Arama Motoru

**Ne Anlatılmalı?**
"JSON formatına getirdiğimiz bu verileri sadece kelime eşleştirerek aramıyoruz. Sistemimiz **Hibrit Arama (Hybrid Search)** altyapısına sahip.
1. **Dense Search (Vektörel Arama):** CV'deki yetenek ve deneyimleri `SentenceTransformers` ile vektör uzayına (sayısal verilere) çevirip `FAISS` veritabanına kaydediyoruz. Bu sayede 'Veritabanı Yönetimi' arattığımızda, sistem içinde 'SQL' veya 'PostgreSQL' geçen adayları da bulabiliyor (Anlamsal eşleşme).
2. **Sparse Search (BM25):** Vektörel aramanın zayıf kaldığı özel ürün isimlerinde vb. nokta atışı kelime eşleştirmesi yapıyor. İkisi birleşince mükemmel bir arama motoru ortaya çıkıyor."

---

## 📍 4. Milestone 4: Aday Sıralama ve Yapay Zeka Raporu

**Ne Anlatılmalı?**
"Son aşamamız, sürecin meyvesini yediğimiz yerdir. Sisteme İngilizce veya Türkçe bir İş Tanımı (Job Description) veriyoruz.
1. **JD Parser:** Verdiğimiz iş tanımındaki kaç yıl deneyim istendiğini, hangi becerilerin zorunlu olduğunu otomatik ayıklıyor.
2. **Skorlama:** Tüm adayları bu iş tanımına göre tek tek tarayıp 100 üzerinden Matematiksel bir puan (Ağırlıklı Skor) veriyor.
3. **LLM Explainer:** En iyi adayları listeliyor ve işe alım uzmanı için adayın neden iyi bir eşleşme olduğunu veya hangi şartları sağlamadığını LLM (Gemini vb.) kullanarak özetliyor."

**Göstermeniz Gereken Çıktı Örneği:**
> *"Hocam, örneğin sisteme 'En az 2 yıl deneyimli Python, Django ve Docker bilen Yazılım Mühendisi arıyoruz' şeklinde bir ilan girdik. Sistemin tüm adaylar arasından süzüp bize verdiği Nihai Rapor şu şekildedir:"*

ranking_outputs/JD-71b331bd_report.txt
ranking_outputs/JD-71b331bd_results.json

```text
================================================================================
  CANDIDATE RANKING REPORT (ADAY SIRALAMA RAPORU)
================================================================================
  Job ID     : JD-71b331bd
  Candidates : 32 (Tüm Veritabanı)

  Job Description (İş Tanımı):
  We are looking for a Software Engineer with experience in Python, Django, Docker, and PostgreSQL. At least 2 years of experience required.

---------------------------------------------------------------------------------------------------------
  Rank  Candidate Name           Skills   Exp      Edu      Soft     Total Score   
---------------------------------------------------------------------------------------------------------
  1     Yuksel Cosgun Backend     75.2    70.8    60.0    60.0    69.9 / 100
  2     Muhammed Fatih Ulasli     74.2    71.7    60.0    60.0    69.8 / 100
  3     Rumeysa Dilan Gokce       74.2    70.1    60.0    60.0    69.2 / 100
... (Diğer adaylar)
---------------------------------------------------------------------------------------------------------

  CANDIDATE #1 — Yuksel Cosgun Backend
  Final Score: 69.9/100
  Recommendation: Moderate Match (Uygun Eşleşme)

  [+] Strengths (Güçlü Yönler):
    - Güçlü teknik beceri uyumu (Skor: 75)
    - İstenen tecrübe süresi ile uyumlu deneyimler bulundu (Skor: 71)

  [!] Missing Requirements (Eksik Beklentiler):
    - Django yetkinliği özgeçmişte bulunamadı.

  ----------------------------------------

  CANDIDATE #2 — Muhammed Fatih Ulasli
  Final Score: 69.8/100
  Recommendation: Moderate Match (Uygun Eşleşme)

  [+] Strengths (Güçlü Yönler):
    - Güçlü teknik beceri uyumu (Skor: 74)
    - İstenen alanlarda (Python vb.) tecrübesi bulunuyor.

  [!] Missing Requirements (Eksik Beklentiler):
    - Docker tecrübesine rastlanmadı.
================================================================================
```

---

### Kapanış Cümlesi Önerisi:
"Sonuç olarak hocam, bu proje PDF'in içindeki ham, karmaşık metinden başlayıp; otonom olarak aday puanlamaya, eksik becerilerin tespitine kadar giden tam teşekküllü bir AI ürünü haline gelmiştir."
