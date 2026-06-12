import sys
sys.stdout.reconfigure(encoding='utf-8')
import re

title = "Baskani Olarak öğrenci-Sektör"
first_person_verbs = [
    "oldum", "yaptım", "yaptim", "çalıştım", "calistim", "ettim", "aldım", "aldim", "okudum", 
    "mezun", "öğrencisiyim", "ogrencisiyim", "adayıyım", "adayiyim", "hedefim", "hedef", "aranıyor", "araniyor"
]

is_noisy = (
    title == "-" or not title or
    any(v in title.lower() for v in first_person_verbs) or
    len(title.split()) > 5 or
    "-" in title or "/" in title or "|" in title or "," in title or "." in title
)

print("Title:", repr(title))
print("is_noisy:", is_noisy)
print("len of split:", len(title.split()))
print("hyphen in title:", "-" in title)
print("Characters in title:")
for c in title:
    print(f"  {repr(c)}: {ord(c)}")
