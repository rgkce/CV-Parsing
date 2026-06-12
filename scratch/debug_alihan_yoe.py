import re
import sys
sys.path.append('cv-parser-script')
from cv_parser8 import turkish_lower

experience_text = """mesudun yeri adana, Türkiye
waiter
01/06/2021 - 01/09/2021 | la blanche hotel mugla/bodrum, Türkiye
01/06/2022 - 01/09/2022 | baia hotel muğla/bodrum, Türkiye
01/04/2023 - 01/10/2023 | mett hotel mugla/bodrum, Türkiye waiter
01/05/2024 - 01/10/2024 | dogus group/dmaris bay mugla/marmaris, Türkiye
01/06/2025 - 23/10/2025 | hobbies and interests hiking im into outdoor sports. cycling"""

blocks = []
for line in experience_text.splitlines():
    for part in line.split("|"):
        part_clean = part.strip()
        if part_clean:
            blocks.append(part_clean)
    
start_dates = []
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

for i, block in enumerate(blocks):
    block_lower = block.lower()
    block_lower_tr = turkish_lower(block)
    
    is_intern = False
    if re.search(r'\b(staj|stajyer[a-z]*|staj[ıi][a-z]*|intern|interns|internship|trainee|trainees)\b', block_lower) or \
       re.search(r'\b(staj|stajyer[a-z]*|staj[ıi][a-z]*|intern|interns|internship|trainee|trainees)\b', block_lower_tr):
        is_intern = True
        
    if is_intern:
        continue
        
    if any(h in block_lower for h in ["lise", "lisesi", "high school"]):
        continue
        
    if any(v in block_lower for v in ["topluluk", "topluluğu", "kulüp", "kulübü", "society", "association", "gönüllü", "gonullu", "dernek", "derneği"]):
        continue
        
    paren_matches = re.findall(r'\(([^)]+)\)', block)
    search_targets = paren_matches + [block]
    
    found_date = False
    for target in search_targets:
        target_lower = turkish_lower(target)
        parts = re.split(r'[-–]|to', target_lower)
        
        if parts:
            part = parts[0].strip()
            if not part:
                continue
            
            m_month_year = re.search(r'\b(' + '|'.join(_MONTH_MAP.keys()) + r')\b.*?(\b(?:19|20)\d{2}\b)', part)
            if m_month_year:
                month_str = m_month_year.group(1)
                year_str = m_month_year.group(2)
                start_dates.append((int(year_str), _MONTH_MAP[month_str], f"1: {block[:30]}"))
                found_date = True
                
            m_year_month = re.search(r'(\b(?:19|20)\d{2}\b).*?\b(' + '|'.join(_MONTH_MAP.keys()) + r')\b', part)
            if m_year_month and not found_date:
                year_str = m_year_month.group(1)
                month_str = m_year_month.group(2)
                start_dates.append((int(year_str), _MONTH_MAP[month_str], f"2: {block[:30]}"))
                found_date = True
                
            if not found_date:
                m_numeric = re.search(r'\b(\d{1,2})[./-](?:19|20)?(\d{2,4})\b', part)
                if m_numeric:
                    month_val = int(m_numeric.group(1))
                    year_str = m_numeric.group(2)
                    if len(year_str) == 2:
                        year_val = 2000 + int(year_str)
                    else:
                        year_val = int(year_str)
                    if 1 <= month_val <= 12 and 1900 <= year_val <= 2030:
                        start_dates.append((year_val, month_val, f"3: {block[:30]}"))
                        found_date = True
                        
            if not found_date:
                m_year = re.search(r'\b((?:19|20)\d{2})\b', part)
                if m_year:
                    year_val = int(m_year.group(1))
                    start_dates.append((year_val, 1, f"4: {block[:30]}"))
                    found_date = True
        if found_date:
            break

print("Extracted start dates:")
for d in start_dates:
    print(d)
