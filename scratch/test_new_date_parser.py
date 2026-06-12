import re

_MONTH_MAP = {
    "ocak": 1, "january": 1, "jan": 1,
    "şubat": 2, "subat": 2, "february": 2, "feb": 2,
    "mart": 3, "march": 3, "mar": 3,
    "nisan": 4, "april": 4, "apr": 4,
    "mayıs": 5, "mayis": 5, "may": 5,
    "haziran": 6, "june": 6, "jun": 6,
    "temmuz": 7, "july": 7, "jul": 7,
    "ağustos": 8, "agustos": 8, "august": 8, "aug": 8,
    "eylül": 9, "eylul": 9, "september": 9, "sep": 9,
    "eylil": 9,
    "ekim": 10, "october": 10, "oct": 10,
    "kasım": 11, "kasim": 11, "november": 11, "nov": 11,
    "aralık": 12, "aralik": 12, "december": 12, "dec": 12
}

def parse_start_date(block: str) -> tuple[int, int] | None:
    # 1. Split block into targets (e.g. parenthesis contents first, then the whole block)
    paren_matches = re.findall(r'\(([^)]+)\)', block)
    search_targets = paren_matches + [block]
    
    for target in search_targets:
        target_lower = target.lower().strip()
        # Split target by hyphen/to
        parts = re.split(r'[-–]|to', target_lower)
        
        # Loop through all parts to find the first part that contains a date
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            found_date = False
            year_val = 0
            month_val = 1
            
            # A. Check for Month and Year (e.g. 'Ocak 2021' or '2021 Ocak')
            m_month_year = re.search(r'\b(' + '|'.join(_MONTH_MAP.keys()) + r')\b.*?(\b(?:19|20)\d{2}\b)', part)
            if m_month_year:
                month_str = m_month_year.group(1)
                year_str = m_month_year.group(2)
                return int(year_str), _MONTH_MAP[month_str]
                
            m_year_month = re.search(r'(\b(?:19|20)\d{2}\b).*?\b(' + '|'.join(_MONTH_MAP.keys()) + r')\b', part)
            if m_year_month:
                year_str = m_year_month.group(1)
                month_str = m_year_month.group(2)
                return int(year_str), _MONTH_MAP[month_str]
                
            # B. Check for 3-part numeric date (DD/MM/YYYY or DD.MM.YYYY)
            m_3part = re.search(r'\b(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)?\d{2,4})\b', part)
            if m_3part:
                day_val = int(m_3part.group(1))
                month_val = int(m_3part.group(2))
                year_str = m_3part.group(3)
                if len(year_str) == 2:
                    year_val = 2000 + int(year_str)
                else:
                    year_val = int(year_str)
                if 1 <= month_val <= 12 and 1900 <= year_val <= 2030:
                    return year_val, month_val
                    
            # C. Check for 2-part numeric date (MM/YYYY)
            m_2part = re.search(r'\b(\d{1,2})[./-]((?:19|20)?\d{2,4})\b', part)
            if m_2part:
                month_val = int(m_2part.group(1))
                year_str = m_2part.group(2)
                if len(year_str) == 2:
                    year_val = 2000 + int(year_str)
                else:
                    year_val = int(year_str)
                if 1 <= month_val <= 12 and 1900 <= year_val <= 2030:
                    return year_val, month_val
                    
            # D. Check for just Year (YYYY)
            m_year = re.search(r'\b((?:19|20)\d{2})\b', part)
            if m_year:
                year_val = int(m_year.group(1))
                return year_val, 1
                
    return None

# Test on our troublesome blocks
test_cases = [
    "01/06/2021 - 01/09/2021",
    "01/06/2022 - 01/09/2022",
    "2.10.2024-11.02.2025",
    "01/07/2022-mevcut durum",
    "freelance mobile developer-2021-present",
    "Tebessüm Anaokulu (Mayıs 2025 - Ağustos 2025)",
]

print("Testing date extraction:")
for tc in test_cases:
    res = parse_start_date(tc)
    print(f"  {tc!r:50} -> {res}")
