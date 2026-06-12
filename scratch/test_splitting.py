import re
from pathlib import Path

file_path = Path("gokdeniz can.pdf")
raw_text = "gokdenizcanofficial@gmail.com"

print("Original raw_text:", raw_text)

_fname_parts = re.findall(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}', file_path.stem.lower())
print("Fname parts:", _fname_parts)

for _p in _fname_parts:
    # Split "wordbeyza@..." into "word beyza@..."
    raw_text = re.sub(f'([a-zA-ZçğıöşüÇĞİÖŞÜ])({_p}[a-zA-Z0-9._%+\\-]*@)', r'\1 \2', raw_text, flags=re.I)
    print(f"After part {_p!r}:", raw_text)
