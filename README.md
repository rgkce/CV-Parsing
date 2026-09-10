"# CV-Parsing" 

## Çalıştırmak için lazım olan komutlar:
- pip install -r requirements.txt  (ilk kez çalıştırırken)
Alternatifler:
  - python cv_parser_script/cv_parser8.py --pdf-dir "c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF"
  - python cv_parser_script/cv_parser8.py --pdf-dir "data/PDF"

- python cv_parser_script/cv_parser8.py (final.dataset.json oluşturmak için)
- python -m semantic_search.run_pipeline (vektörleştirme ve indeks oluşturmak için)
- python -m semantic_search.run_query --query "Python developer with ML experience" (sadece semantik arama yapmak için)
- python -m candidate_ranker.run_ranking --jd "Python developer with machine learning experience"
  python -m candidate_ranker.run_ranking --jd "Python developer with machine learning
  experience"
  (Aday sıralama ve Akıllı değerlendirme raporu almak için)
  Farklı seçeneklerle çalıştırmak için:
  - python -m candidate_ranker.run_ranking --jd "Data Scientist" --top-k 10 (toplam getirilecek aday sayısı için)
  - python -m candidate_ranker.run_ranking --jd "Data Scientist" --skip-llm (LLM açıklama kısmını atlamak için)
- python -m candidate_ranker.run_ranking --job 2 --top-k 5 --skip-llm (iş ilanı numarası yazarak sonuç almak için)

## Gemini API aktif etmek için:
- $env:GOOGLE_API_KEY="API_KEYINIZI_BURAYA_YAZIN" (Windows powershell)
- set GOOGLE_API_KEY=API_KEYINIZI_BURAYA_YAZIN (Windows CMD)"# CV-Parser-Project" 

