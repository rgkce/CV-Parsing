import re
title1 = "MÜHENDİSİ"
title2 = "2018 MEZUNU"

l1 = title1.lower()
l2 = title2.lower()

ROLE_KEYWORDS_RE = re.compile(
    r"\b("
    r"mühendis[aeıiuüüşöçyysmdnrl]*"
    r"|mezun[a-z]*"
    r")\b",
    re.I
)

print(f"l1: {repr(l1)}")
print(f"l2: {repr(l2)}")
print("Match 1:", ROLE_KEYWORDS_RE.search(l1))
print("Match 2:", ROLE_KEYWORDS_RE.search(l2))
