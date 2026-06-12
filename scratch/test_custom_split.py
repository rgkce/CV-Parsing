import re
from pathlib import Path

def name_aware_split(raw_text: str, file_path_stem: str) -> str:
    _fname_parts = re.findall(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}', file_path_stem.lower())
    for _p in _fname_parts:
        # We want to match letters before the name part _p
        pattern = f'([a-zA-ZçğıöşüÇĞİÖŞÜ]+)({_p}[a-zA-Z0-9._%+\\-]*@)'
        
        def replace_fn(match: re.Match) -> str:
            prefix = match.group(1)
            rest = match.group(2)
            prefix_lower = prefix.lower()
            
            # Check if prefix contains any of the OTHER name parts
            other_parts = [part for part in _fname_parts if part != _p]
            if any(part in prefix_lower for part in other_parts):
                # Do NOT split, it's likely a combination of the candidate's names
                return match.group(0)
            else:
                # Split off the prefix
                return prefix + " " + rest
                
        raw_text = re.sub(pattern, replace_fn, raw_text, flags=re.I)
    return raw_text

# Test cases
tests = [
    ("gokdenizcanofficial@gmail.com", "gokdeniz can", "gokdenizcanofficial@gmail.com"),
    ("iletisimbeyzaaktas2003@gmail.com", "beyza aktas", "iletisim beyzaaktas2003@gmail.com"),
    ("aysesoydal@gmail.com", "ayse soydal", "aysesoydal@gmail.com"),
    ("epostaayse.soydal@gmail.com", "ayse soydal", "eposta ayse.soydal@gmail.com"),
    ("bilgisayarmuhendisligibeyzaaktas2003@gmail.com", "beyza aktas", "bilgisayarmuhendisligibeyzaaktas2003@gmail.com"), # wait, "muhendisligi" doesn't contain "aktas", so this splits? Yes, if it's prepended by other text.
]

for raw, stem, expected in tests:
    result = name_aware_split(raw, stem)
    status = "SUCCESS" if result == expected else "FAIL"
    print(f"[{status}] Stem: {stem!r} | In: {raw!r} | Out: {result!r} (Expected: {expected!r})")
