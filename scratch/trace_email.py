import sys
from pathlib import Path
import re

# Add cv-parser-script to sys.path
sys.path.append(str(Path(__file__).parent.parent / "cv-parser-script"))
from cv_parser8 import extract_contact_info, process_cv, ocr_fallback

pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gokdeniz can.pdf")
raw_text, format_str = ocr_fallback(pdf_path)

# Let's trace extract_contact_info line-by-line
text = raw_text
contact_search_text = text

_lines_raw = contact_search_text.splitlines()
for _idx, _l in enumerate(_lines_raw):
    _l_clean = _l.strip()
    _l_norm = _l_clean.lower().replace("lınkedın", "linkedin").replace("httos", "https")
    if "linkedin.com/in/" in _l_norm or "lınkedın.com/in/" in _l_norm:
        _match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_\-%.\u00C0-\u024F\-]+)?', _l_clean, re.I)
        if _match:
            _slug = _match.group(1) or ""
            _needs_next = (_slug.endswith("-") or _slug.endswith("/") or len(_slug) <= 3)
            if _needs_next:
                for _next_idx in range(_idx + 1, min(_idx + 2, len(_lines_raw))):
                    _next_l = _lines_raw[_next_idx].strip()
                    if not _next_l:
                        continue
                    if "@" in _next_l or re.search(r"\d{7,}", _next_l) or any(_h in _next_l.lower() for _h in ["e-posta", "telefon", "deneyim", "egitim", "profil", "skills", "experience", "hakkında", "hakkimda", "summary", "about", "özet", "ozet", "kişisel", "kisisel", "personal", "contact", "iletisim", "iletişim", "diller", "beceriler", "yabancı", "dil", "referans"]):
                        break
                    _next_parts = _next_l.split()
                    if _next_parts:
                        _first_token = _next_parts[0]
                        if re.match(r'^[a-zA-Z0-9_\-%.\u00C0-\u024F\-/]+$', _first_token):
                            _slug = _slug + _first_token
            _combined_url = f"https://www.linkedin.com/in/{_slug}"
            contact_search_text = contact_search_text.replace(_l, _combined_url)
            break

ref_match = re.search(r'\n\s*(referanslar|references)\s*[:]?\s*\n', contact_search_text, re.IGNORECASE)
if ref_match and ref_match.start() > 300:
    contact_search_text = contact_search_text[:ref_match.start()]

from cv_parser8 import _RE_EMAIL

_has_valid_email_already = _RE_EMAIL.search(contact_search_text)
print("Trace: _has_valid_email_already:", bool(_has_valid_email_already))
if _has_valid_email_already:
    email_search_text = contact_search_text
else:
    email_search_text = re.sub(r"([A-Za-z0-9._%+\-])\s+@\s+([A-Za-z0-9])", r"\1@\2", contact_search_text)

email_search_text = re.sub(
    r"([A-Za-z0-9])\s*\.\s*(com|net|org|edu|gov|info|online|site|link|app|dev|me|io|co|tr|in|biz|[a-z]{2})(?=\s|$|[,;\)])",
    r"\1.\2",
    email_search_text,
    flags=re.I
)

print("Trace: email_search_text matching:")
email_match = _RE_EMAIL.search(email_search_text)
if email_match:
    email_addr = email_match.group(0).strip()
    print("  Found email_addr in search text:", repr(email_addr))
    
    _tld_trunc = re.search(r'\.(com|net|org|edu|gov|io|me|co\.uk|co\.in|co\.jp|co\.kr|info|biz|tr|app|dev)', email_addr, re.I)
    if _tld_trunc:
        _end_pos = _tld_trunc.end()
        _trailing = email_addr[_end_pos:]
        if _trailing and not re.match(r'^(\.[a-z]{2})?$', _trailing, re.I):
            email_addr = email_addr[:_end_pos]
            print("  After TLD trunc:", repr(email_addr))
            
    _common_tld_extensions = {
        ".co": [".com", ".co.uk", ".co.in", ".co.jp", ".co.kr"],
        ".ne": [".net"],
        ".or": [".org"],
        ".ed": [".edu"],
        ".go": [".gov"],
    }
    for _short_tld, _full_tlds in _common_tld_extensions.items():
        if email_addr.endswith(_short_tld):
            for _full_tld in _full_tlds:
                _candidate = email_addr[: -len(_short_tld)] + _full_tld
                _text_no_space = text.lower().replace(" ", "")
                _search_no_space = email_search_text.lower().replace(" ", "")
                if _candidate.lower() in _text_no_space or _candidate.lower() in _search_no_space:
                    email_addr = _candidate
                    print("  After short TLD expansion:", repr(email_addr))
                    break
            break
            
    if "@" in email_addr:
        email_addr = email_addr.replace("|", "l")
        email_addr = email_addr.lower().translate(str.maketrans("ışğüçöı", "isgucoi"))
        print("  After char translit:", repr(email_addr))
        
        local_part, domain_part = email_addr.split("@", 1)
        domain_part_lower = domain_part.lower()
        if domain_part_lower in ["gmal.com", "qmail.com", "gqmail.com", "gmial.com", "gamil.com", "gmaıl.com", "gma1l.com", "gmai1.com"]:
            domain_part = "gmail.com"
        elif domain_part_lower in ["hotmal.com", "hotmai1.com", "hotmaıl.com"]:
            domain_part = "hotmail.com"
            
        if "cosqun" in local_part:
            local_part = local_part.replace("cosqun", "cosgun")
            
        print("  local_part before noise removal:", repr(local_part))
        local_part = re.sub(r'^[^a-zA-Z0-9]+', '', local_part)
        print("  local_part after non-alphanum removal:", repr(local_part))
        while True:
            stripped_any = False
            for noise_pref in ["mm", "sj", "lo", "39"]:
                if local_part.lower().startswith(noise_pref) and len(local_part) > len(noise_pref):
                    local_part = local_part[len(noise_pref):]
                    local_part = re.sub(r'^[^a-zA-Z0-9]+', '', local_part)
                    stripped_any = True
            if not stripped_any:
                break
        print("  local_part after noise loop:", repr(local_part))
        
        # Wait, is there any other place in the script that replaces things?
        email_addr = f"{local_part}@{domain_part}"
        print("  Final email_addr in trace:", repr(email_addr))
else:
    print("  No email found in trace")
