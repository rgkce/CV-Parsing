import re
import datetime
from cv_parser8 import *
def extract_title_and_experience(text: str, experience_text: str = "", education_text: str = "") -> tuple[str, str]:
    """
    Extract the candidate's professional title and total years of experience.
    
    Title detection:
      1. Check first line for "Name - Title" or "Name | Title" pattern.
      2. If not found, check lines 2-5 for short, title-like lines
         (skipping contact info and section headings).
    
    Years of experience:
      1. Look for explicit "X years" mentions.
      2. If not found, calculate from date ranges in the text.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "-", "0"
    
    # 1. Try to find title in the first line (name - title)
    first_line = lines[0]
    title = "-"
    first_line_has_keyword = False
    if " - " in first_line:
        candidate = first_line.split(" - ", 1)[1].strip()
        if candidate.lower() not in _TITLE_SKIP_HEADINGS:
            title = candidate
            if re.search(r"\b(developer|engineer|programmer|architect|devops|sre|qa|tester|software|frontend|backend|fullstack|kameraman|montajcı|editör|editor|uzman|mühendis|geliştirici|stajyer|intern)\b", candidate, re.I):
                first_line_has_keyword = True
    elif " | " in first_line:
        candidate = first_line.split(" | ", 1)[1].strip()
        if candidate.lower() not in _TITLE_SKIP_HEADINGS:
            title = candidate
            if re.search(r"\b(developer|engineer|programmer|architect|devops|sre|qa|tester|software|frontend|backend|fullstack|kameraman|montajcı|editör|editor|uzman|mühendis|geliştirici|stajyer|intern)\b", candidate, re.I):
                first_line_has_keyword = True
    
    # 2. If not found or low confidence, look at subsequent lines skipping contact info
    # Expanded role keywords list covering modern tech, business, and Turkish roles
    _ROLE_KEYWORDS_RE = re.compile(
        r"\b("
        # === Software / Engineering ===
        r"developer|engineer|programmer|architect|devops|sre|qa|tester"
        r"|software|frontend|backend|fullstack|full-stack|full stack"
        r"|web developer|mobile developer|ios developer|android developer"
        # === Data / AI / ML ===
        r"|data scientist|data analyst|data engineer|machine learning|ml engineer"
        r"|ai engineer|bi analyst|bi developer|business intelligence"
        # === Design / Creative ===
        r"|designer|ui designer|ux designer|ui/ux|ux/ui|graphic designer"
        r"|product designer|visual designer|art director|creative director"
        r"|kameraman|montajcı|editör|editor"
        # === Management / Leadership ===
        r"|manager|director|lead|head|chief|officer|president|vp"
        r"|team lead|tech lead|project manager|product manager|scrum master"
        r"|ceo|cto|cfo|coo|cio|cmo"
        # === Analyst / Specialist / Consultant ===
        r"|analyst|specialist|consultant|coordinator|advisor|strategist"
        r"|expert|researcher|scientist"
        # === Marketing / Business ===
        r"|marketing|sales|account|business|operations|finance"
        r"|content writer|copywriter|seo specialist|social media"
        # === Service / Hospitality / Sales ===
        r"|waiter|waitress|garson|barista|bartender|receptionist|cashier|host|hostess|komi|servis elemanı|servis elemani|kasiyer"
        r"|sales representative|satış temsilcisi|satis temsilcisi|sales advisor|sales consultant|promoter|sales associate"
        # === Turkish Roles ===
        r"|uzman|mühendis|muhendis|mithendis|mtihendis|muuhendis|mühendisi|muhendisi|mithendisi|muuhendisi|geliştirici|gelistirici|yönetici|yonetici|müdür|mudur|direktör|direktor|koordinatör|koordinator"
        r"|danışman|danisman|tasarımcı|tasarimci|araştırmacı|arastirmaci|asistan|analist|lider|başkan|baskan"
        r"|stajyer|intern|student|öğrenci|ogrenci|mezun|graduate"
        r"|teknisyen|operatör|operator|editör|editor|muhabir|gazeteci"
        r")\b",
        re.I,
    )
    _ROLE_KEYWORDS_RE = re.compile(
        r"\b("
        r"developer|engineer|programmer|architect|devops|sre|qa|tester"
        r"|software|frontend|backend|fullstack|full-stack|full stack"
        r"|manager|director|lead|head|chief|officer|president|vp"
        r"|team lead|tech lead|project manager|product manager|scrum master"
        r"|ceo|cto|cfo|coo|cio|cmo"
        r"|analyst|specialist|consultant|coordinator|advisor|strategist"
        r"|expert|researcher|scientist"
        r"|marketing|sales|account|business|operations|finance"
        r"|content writer|copywriter|seo specialist|social media"
        r"|waiter|waitress|garson|barista|bartender|receptionist|cashier|host|hostess|komi|servis[a-z]*|kasiyer"
        r"|sales representative|sat\u0131\u015f[a-z]*|satis[a-z]*|sales advisor|sales consultant|promoter|sales associate"
        r"|uzman[ae\u0131iu\u00fcyysmdnrl]*|m\u00fchendis[ae\u0131iu\u00fcyysmdnrl]*|muhendis[ae\u0131iu\u00fcyysmdnrl]*"
        r"|geli\u015ftirici[ae\u0131iu\u00fcyysmdnrl]*|gelistirici[ae\u0131iu\u00fcyysmdnrl]*"
        r"|y\u00f6netici[ae\u0131iu\u00fcyysmdnrl]*|yonetici[ae\u0131iu\u00fcyysmdnrl]*"
        r"|m\u00fcd\u00fcr[ae\u0131iu\u00fcyysmdnrl]*|mudur[ae\u0131iu\u00fcyysmdnrl]*"
        r"|ba\u015fkan[ae\u0131iu\u00fcyysmdnrl]*|baskan[ae\u0131iu\u00fcyysmdnrl]*|lider[ae\u0131iu\u00fcyysmdnrl]*"
        r"|stajyer[ae\u0131iu\u00fcyysmdnrl]*|intern[a-z]*|student[a-z]*|mezun[a-z]*|graduate[a-z]*"
        r"|\u00f6\u011frenci[ae\u0131iu\u00fcyysmdnrl]*|ogrenci[ae\u0131iu\u00fcyysmdnrl]*"
        r"|dan\u0131\u015fman[ae\u0131iu\u00fcyysmdnrl]*|danisman[ae\u0131iu\u00fcyysmdnrl]*"
        r"|tasar\u0131mc\u0131[ae\u0131iu\u00fcyysmdnrl]*|tasarimci[ae\u0131iu\u00fcyysmdnrl]*"
        r"|direkt\u00f6r[a-z]*|direktor[a-z]*|koordinat\u00f6r[a-z]*|koordinator[a-z]*|analist[a-z]*"
        r"|teknisyen[a-z]*|operat\u00f6r[a-z]*|operator[a-z]*|edit\u00f6r[a-z]*|editor[a-z]*|muhabir[a-z]*|gazeteci[a-z]*"
        r"|g\u00f6revli[ae\u0131iu\u00fcyysmdnrl]*|gorevli[ae\u0131iu\u00fcyysmdnrl]*"
        r")\b",
        re.I,
    )
    
    if title == "-" or not first_line_has_keyword:
        _title_candidate_fallback = None  # store best non-keyword candidate
        for l in lines[1:35]:  # extended search range to reach second column top
            l_lower = l.lower().strip()
            # Skip lines with email, @, http, or phone-like patterns
            if "@" in l or "http" in l or "www." in l or re.search(r"\d{5,}", l):
                continue
            # Skip demographic/metadata lines
            if re.search(r"\b(permit|nationality|gender|birth|marital|military|ehliyet|driving|allowance)\b", l_lower):
                continue
            # Skip lines containing "değil" or "degil"
            if "de\u011fil" in l_lower or "degil" in l_lower:
                continue
            # Skip prepositional/adverbial phrases that are not titles
            if l_lower.endswith(("alanında", "alanlarinda", "alanlarında", "olarak", "üzere", "hakkında", "amacıyla", "iletisim", "iletişim")):
                continue
            # Skip lines with first-person verbs or pronouns (typical of summary/profile sentences)
            if re.search(r"\b(i\s+am|i\s+have|worked|studied|developed|managed|created|assisted)\b", l_lower):
                continue
            _m_fp = re.search(r"(?:iyorum|\u0131yorum|\u00fcyorum|uyorum|eyim|ay\u0131m|d\u0131m|dim|dum|d\u00fcm|ar\u0131m|erim|t\u0131m|tim|y\u0131m|yim|imi|\u0131m\u0131|\u00fcm\u00fc|umu|miz|m\u0131z|\u00fcm\u00fcz|umuz|lerim|lar\u0131m|lerimi|lar\u0131m\u0131)\b", l_lower)
            if _m_fp:
                if len(l.split()) > 1:
                    continue
            # Redundant check removed
            # Skip lines that look like section headings
            if l_lower in _TITLE_SKIP_HEADINGS or _sd_norm(l) in _SD_EXT_MAP:
                break
            # Skip lines that look like university/education info
            if re.search(r"\b(university|üniversite|college|school|okul|fakülte|bölüm)\b", l, re.I):
                continue
            # Skip lines that look like addresses or locations
            if re.search(r"\b(sokak|cadde|mahalle|apt|kat|no|street|avenue|city)\b", l, re.I):
                continue
            # First clean line that looks like a title (short, role-like)
            word_count = len(l.split())
            if 1 <= word_count <= 8:
                # High confidence: contains explicit role keywords
                if _ROLE_KEYWORDS_RE.search(l_lower):
                    print("Assigned from regex match:", repr(l)); title = l
                    break
                # Low confidence fallback: short line (2-5 words) without digits
                # that looks like a title (not a name or date)
                elif title == "-" and _title_candidate_fallback is None and 2 <= word_count <= 5:
                    if not re.search(r"\d", l):  # no digits (not a date or phone)
                        # Skip lines that end with coordinating conjunctions, prepositions, or commas
                        if l_lower.rstrip().endswith(("ve", "ile", "veya", "and", "or", "with", "in", "on", "at", "for", "of", ",")):
                            continue
                        # Skip descriptive prose fragments
                        if any(w in l_lower for w in ["sahip", "yürüten", "yuruten", "aktif", "bir"]):
                            continue
                        # Skip lines that end with period (sentences, not titles)
                        if l.rstrip().endswith("."):
                            continue
                        # Skip lines that are section keywords from _SD_EXT_MAP
                        if _sd_norm(l) in _SD_EXT_MAP:
                            continue
                        _title_candidate_fallback = l
        
        # Use the fallback candidate if no keyword match was found
        if title == "-" and _title_candidate_fallback:
            print("Assigned fallback:", repr(_title_candidate_fallback)); title = _title_candidate_fallback
            
        # 2b. If still not found or remains generic, check education section for student roles
        if (title == "-" or title == "") and education_text:
            for _el in education_text.split("\n"):
                _el_lower = _el.lower().strip()
                if any(_k in _el_lower for _k in ["öğrenci", "ogrenci", "student", "lisans"]):
                    if len(_el.split()) <= 8:
                        title = _el.strip()
                        break
    
    # 3. Calculate years of experience
    # In accordance with the user's explicit rule: "ilk işinin başlangıç tarihinden bulunduğumuz yıla kadar hesaplansın deneyim yılı"
    # Find all 4-digit years starting with 19 or 20 in the experience text
    years = "0"
    if experience_text:
        all_years = [int(y) for y in re.findall(r'\b(20\d{2}|19\d{2})\b', experience_text)]
        if all_years:
            earliest_year = min(all_years)
            max_year = max(all_years)
            import datetime
            current_year = datetime.date.today().year
            if earliest_year <= current_year:
                ans = current_year - earliest_year
                
                # EXCEPTION: If the person is clearly an intern/student who only worked in one specific year 
                # (max_year == earliest_year) and is not currently employed.
                exp_lower = experience_text.lower()
                is_present = bool(re.search(r'\b(present|devam|günümüz|now|current|bugün|bugüne)\b', exp_lower))
                if max_year == earliest_year and not is_present:
                    ans = 0
                
                years = str(ans)
    
    # Clean up common title labels and prefixes (case-insensitive)
    title = title.strip()
    _prefix_pat = re.compile(
        r"^(?:ad[ı]?\s*soyad[ı]?|ad\s*soyad|isim|name|full\s*name|cv|özgeçmiş|ozgecmis)\s*[:\-–|]*\s*",
        re.I
    )
    title = _prefix_pat.sub("", title).strip()
    
    # Title casing for consistency
    if title == title.lower() or title == title.upper():
        title = title.title()
    if len(title) > 50:
        title = title[:50] + "..."
    
    return title, years
