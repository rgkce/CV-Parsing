import json, sys, io, re, unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Simulate the clean_and_resolve_title function to debug it
DEPT_TO_TITLE = {
    "bilgisayar mühendis": "Bilgisayar Mühendisi",
    "bilgisayar muhendis": "Bilgisayar Mühendisi",
    "endüstri mühendis": "Endüstri Mühendisi",
    "endustri muhendis": "Endüstri Mühendisi",
    "endiistri miihendis": "Endüstri Mühendisi",
    "endiistri muhendis": "Endüstri Mühendisi",
    "veteriner": "Veteriner Hekim",
    "lojistik": "Lojistik Uzmanı",
    "python": "X",
    "java": "Computer Engineer",
}

data = json.load(open("final_dataset.json", "r", encoding="utf-8"))

# Simulate what clean_and_resolve_title receives
for idx in [5, 7, 9, 12]:  # ayse soydal, aziz ekren, beyza aktas, burcu kuzucu
    c = data[idx]
    name = c.get("file_path", "?").rsplit("\\", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    title = c.get("sections", {}).get("title", "?")
    raw_text = c.get("raw_text", "")
    
    # Simulate the function
    title_clean = title.strip().rstrip(".:;,-|/ ")
    title_norm = unicodedata.normalize("NFC", title_clean)
    title_lower = title_norm.lower()
    
    # has_student_keyword check
    has_student = bool(re.search(
        r'(?:öğrenci(?:si(?:yim)?)?|ogrenci(?:si(?:yim)?)?|student|mezunu?)\b', title_lower
    ))
    
    # search raw text
    search_src = raw_text.lower()
    found_kw = None
    for kw, mapped in DEPT_TO_TITLE.items():
        if kw in search_src:
            found_kw = (kw, mapped)
            break
    
    print(f"=== #{idx+1}: {name} ===")
    print(f"  Input title: '{title}'")
    print(f"  title_lower: '{title_lower}'")
    print(f"  has_student_keyword: {has_student}")
    print(f"  Found dept in raw: {found_kw}")
    print(f"  Raw snippet: {raw_text[:200]!r}")
    print()
