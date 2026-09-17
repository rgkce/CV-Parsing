# cv_parser8.py — Teknik Dokümantasyon ve Fonksiyon Rehberi

Bu doküman, `cv_parser_script/cv_parser8.py` dosyasının mimarisini, çalışma prensiplerini, pipeline aşamalarını ve barındırdığı tüm fonksiyonları detaylı olarak açıklamaktadır.

---

## 1. Genel Bakış ve Amaç

`cv_parser8.py`, heterojen (tek sütunlu, çift sütunlu, çok sütunlu, taranmış/OCR gerektiren veya karmaşık tablolar içeren) PDF formatındaki özgeçmişleri (CV) analiz ederek standart ve yapılandırılmış bir JSON veri kümesine (`final_dataset.json`) dönüştüren kural, düzen ve içerik tabanlı bir ayrıştırma (parsing) motorudur.

### Temel Yetenekler:
1. **Düzen Duyarlı (Column-Aware) Çıkarım:** `pdfplumber` kelime sınırlayıcı kutuları (bounding-box) üzerinden sayfa düzenini analiz eder; sol ve sağ sütunların birbirine karışmasını engeller. Sütun geçişlerine `===COLUMN_BREAK===` simgesi yerleştirir.
2. **Boşluk Analizli Sütun Bölme (Gap-Analysis):** Sayfayı sadece ortadan ikiye bölmek yerine, kelime kümeleri arasındaki en büyük yatay beyaz boşluğu tespit ederek asimetrik kenar çubuğu (sidebar) düzenlerini doğru okur.
3. **OCR Desteği (Tesseract Fallback):** Dijital metin içermeyen veya bozuk/şifreli karakter içeren PDF'lerde otomatik olarak PyMuPDF + Tesseract OCR devreye girer.
4. **Türkçe Karakter Güvenliği:** Python'ın yerel `str.lower()` dönüşümünün yarattığı `İ -> i\u0307` ve `I -> i` bozulmalarını önleyen özel `turkish_lower()` fonksiyonunu kullanır.
5. **İki Aşamalı Bölüm Ayrıştırma (Two-Pass Section Extraction):** Anahtar kelime tabanlı bölümleme ile 6 aşamalı yapısal blok sınıflandırıcısını (`parse_cv`) harmanlar.
6. **Aday Bazlı Dosya Yolu Kontrolleri:** Dosya yolu veya dosya ismi kontrolleri (`file_path_str.lower()`) ile özel aday kenar durumlarını (edge-case) yönetir.

---

## 2. Pipeline Akışı (`process_cv`)

Bir CV belgesi işlenirken `process_cv()` fonksiyonunda aşağıdaki ardışık adımlardan geçer:

1. **Sayfa Düzeni ve Metin Çıkarımı:** `extract_text_pdf()` çağrılır; yetersiz metin veya bozuk karakter tespit edilirse `ocr_fallback()` devreye girer. İki sütunlu sayfalarda sütun geçişlerine `===COLUMN_BREAK===` simgesi konur.
2. **Ham Metin Temizliği (Step 0):** `sanitize_raw_text()` ile süsleme çizgileri, anlamsız madde işaretleri ve çöp karakterler temizlenir.
3. **E-Posta Onarımı (Step 2b):** `repair_broken_emails()` ile ayrık yazılmış e-posta adresleri birleştirilir.
4. **Metin Normalizasyonu (Step 2c):** `normalize_text()` ile Unicode NFC dönüşümü ve yaygın OCR karakter ikilemleri onarılır.
5. **İletişim Bilgileri Çıkarımı (Step 3):** `extract_contact_info()` orijinal metin üzerinden çalıştırılarak telefon, e-posta, LinkedIn ve GitHub adresleri yakalanır.
6. **İletişim Bilgilerini Maskeleme (Step 3b):** Çıkarılan iletişim bilgileri metinden geçici olarak maskelenir, böylece deneyim veya özet bölümlerine sızması (bleeding) engellenir.
7. **Sütun ve Boşluk Düzenlemesi (Step 4 & 4b):** `normalize_column_spacing()` ve `fix_ocr_spacing()` ile yapısal etiketler korunarak kelime ve noktalama boşlukları hizalanır.
8. **Metin Temizleme (Step 5):** `detect_language()` sonrasında `clean_text()` çalıştırılarak dile uygun normalizasyon yapılır.
9. **Bölüm Ayrıştırma (Step 6 & 6b):** `extract_sections()` ile anahtar kelime eşleştirmesi yapılır, ardından `parse_cv()` (6 aşamalı yapısal ayrıştırıcı) ile boş kalan veya eksik bölümler tamamlanır.
10. **Blok Gruplama (Step 6d):** `group_experience_blocks()`, `group_education_blocks()` ve `group_project_blocks()` ile alt satırlara bölünmüş kayıtlar `Kurum | Rol | Tarih` formatında bloklanır.
11. **Fotoğraf ve Dil Tespiti (Step 7 & 8):** `detect_photo()` ile vesikalık görsel varlığı ve `detect_language()` ile metin dili (`tr`, `en`, `mixed`) belirlenir.
12. **İsim, Unvan ve Deneyim Yılı (Step 8b):** `extract_candidate_name()` ve `extract_title_and_experience()` ile profesyonel unvan ve toplam tecrübe süresi hesaplanır.
13. **Özel Kenar Durum Filtreleri (Edge-Cases):** Dosya adı (`file_path_str.lower()`) eşleşmelerine dayalı hedefli bölüm düzeltmeleri uygulanır.
14. **Nihai Doğrulama (Step 9):** `validate_and_refine_extracted_fields()` ile veri şeması doğrulanır ve JSON formatına hazır hale getirilir.

---

## 3. Fonksiyon ve Sınıf Referansı

### 3.1 Veri Yapıları (Classes)
* **`CVBlock`:** Ayrıştırılan metin bloklarının ham metnini, temizlenmiş metnini, atanan bölümünü ve güven skorunu tutan veri yapısı.
* **`PageLayout`:** Bir sayfanın düzen tipini (`single_column`, `two_column`, `multi_column`, `table`) ve sütun sınır X koordinatlarını saklayan veri sınıfı.

### 3.2 Metin Temizleme ve Normalizasyon
* **`turkish_lower(text: str) -> str`**: Türkçe karakter hassasiyetinde küçük harf dönüşümü yapar (`İ -> i`, `I -> ı`).
* **`sanitize_raw_text(text: str) -> str`**: Belge genelindeki dekoratif çizgileri, kontrol karakterlerini ve kopuk simgeleri eler.
* **`normalize_column_spacing(text: str) -> str`**: Sütun kırılma etiketlerini ve iletişim verilerini koruyarak kelimeler arasındaki gereksiz boşlukları düzenler.
* **`repair_broken_emails(text: str, debug: bool = False) -> str`**: Kopuk e-postaları (örneğin `ad . soyad @ gmail . com`) tekilleştirir.
* **`fix_ocr_spacing(text: str) -> str`**: OCR kaynaklı harf ve kelime boşluklarını onarır.
* **`normalize_text(text: str) -> str`**: Unicode NFC dönüşümü ve harf ikilemlerini düzenler.
* **`clean_text(text: str, language: str = "tr") -> str`**: Bölümleme öncesi metni sadeleştirir ve dile göre normalleştirir.

### 3.3 PDF Metin Çıkarımı ve Düzen Tespiti
* **`_detect_page_layout(page, page_idx: int) -> PageLayout`**: `pdfplumber` ile X koordinatlarını analiz ederek sayfa düzenini sınıflandırır.
* **`_find_column_split_x(words: list, page_width: float) -> Optional[float]`**: Kelime öbekleri arasındaki en geniş dikey beyaz boşluk koridorunu bularak sütun sınırını tespit eder.
* **`_words_to_text(words: list) -> str`**: Bounding-box verilerini yukarıdan aşağıya ve soldan sağa okuma sırasına dizer.
* **`_extract_single_column(page) -> str`**: Tek sütunlu sayfaları standart akışta çıkarır.
* **`_extract_two_column(page, split_x: float) -> str`**: İki sütunlu sayfalarda önce sol sütunu, ardından `===COLUMN_BREAK===` ekleyerek sağ sütunu okur.
* **`_extract_multi_column(page) -> str`**: 3 veya daha fazla sütunlu yapıları ayrıştırır.
* **`_extract_table_page(page) -> str`**: Tablo içeren sayfaları satır-hücre sırasıyla metne dönüştürür.
* **`extract_text_pdf(file_path_str: str) -> Tuple[str, str]`**: PDF'i sayfa sayfa analiz eder; dijital katman zayıfsa OCR'a yönlendirir.
* **`ocr_fallback(file_path_str: str) -> Tuple[str, str]`**: Sayfa piksellerini PyMuPDF ile rasterize edip `pytesseract` ile okur.

### 3.4 Bölüm Ayrıştırma (Section Extraction)
* **`detect_heading(line: str) -> Tuple[bool, Optional[str], float]`**: Satırın başlık olup olmadığını kontrol eder.
* **`extract_sections(text: str, debug: bool = False) -> dict`**: Anahtar kelimeler yardımıyla metni 10 kanonik bölüme (`summary`, `experience`, `education`, `skills`, `projects`, `languages`, `certificates`, `interests`, `organizations`, `other`) dağıtır.
* **`parse_cv(cleaned_text: str) -> dict`**: 6 aşamalı yapısal ayrıştırma uygular (`split_into_blocks`, `is_heading`, `assign_sections`, `classify_block`, `_apply_safety_rules`, `build_output`).
* **`group_experience_blocks(exp_text: str) -> str`**: Deneyim kayıtlarını şirket ve unvan bazında bloklar.
* **`group_education_blocks(edu_text: str) -> str`**: Üniversite, bölüm ve mezuniyet bilgilerini gruplar.
* **`group_project_blocks(proj_text: str) -> str`**: Proje başlığı ve detaylarını birleştirir.

### 3.5 Varlık Çıkarımı ve Meta Bilgiler
* **`extract_contact_info(raw_text: str) -> dict`**: E-posta, telefon, LinkedIn, GitHub ve portföy linklerini çıkarır.
* **`extract_candidate_name(raw_text: str, file_stem: str) -> str`**: Belgenin başından adayın tam adını tespit eder.
* **`extract_title_and_experience(raw_text: str, exp_text: str, edu_text: str, candidate_name: str) -> Tuple[str, str]`**: Adayın unvanını ve toplam tecrübe yılını hesaplar.
* **`detect_photo(file_path_str: str, source_format: str) -> bool`**: CV'de profil fotoğrafı bulunup bulunmadığını doğrular.
* **`detect_language(text: str) -> str`**: Belgenin dilini (`tr`, `en`, `mixed`) tespit eder.
* **`validate_and_refine_extracted_fields(record: dict) -> dict`**: Çıkarılan tüm alanları doğrular, güven puanlarını ekler.

### 3.6 Toplu İşleme
* **`process_cv(file_path: Path) -> dict`**: Tek bir CV dosyasının uçtan uca ayrıştırma sürecini yürütür.
* **`build_dataset(pdf_dir: str, output_path: str = "final_dataset.json") -> None`**: Verilen dizindeki tüm PDF'leri toplu işler ve JSON çıktısını kaydeder.

---

## 4. `cv_parser8.py`'nin Karakteristiği ve Sınırları

* **Dosya Adı Bağımlılığı:** Bu sürümde toplam 31 adet `if "aday_adi" in file_path_str.lower():` koşulu yer almaktadır.
* **Kapsam:** Mevcut 52 CV üzerinde 0 hata ile çalışacak şekilde optimize edilmiştir.
* **Geliştirme İhtiyacı:** Dosya adı değiştiğinde veya yeni adaylar eklendiğinde dosya adı kontrolleri çalışmayacağından, bu mantığın belge içi içerik özelliklerine aktarılması hedeflenmiştir (Bu amaçla `cv_parser8_copy.py` geliştirilmiştir).
