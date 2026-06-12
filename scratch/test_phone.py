import re

text = """
Poy
t wm -
eS, |
q o
ILETISIM BILGILERI
@ +90 507 385 07 61
Mm ee.clk61@gmail.com
Q CANKAYA/ANKARA
FR www.linkedin.com/in/emre-celikOO
 https://github.com/eeclk
"""

_RE_PHONE_CONTACT = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?){1,4}\d{2,4}(?!\d)",
)

matches = _RE_PHONE_CONTACT.findall(text)
print("Matches:", matches)
for m in matches:
    digits = re.sub(r"\D", "", m)
    print(f"Match: {m!r}, digits: {digits}")
    raw_stripped = m.strip()
    if re.search(r"^\(?(?:19|20)\d{2}\s*[-–\s]", raw_stripped):
        print("Rejected by rule 1")
        continue
    if re.search(r"[-–\s]\s*(?:19|20)\d{2}\)?$", raw_stripped):
        print("Rejected by rule 2")
        continue
    if re.match(r"^\(?(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\)?$", raw_stripped):
        print("Rejected by rule 3")
        continue
    if re.search(r"\d{1,2}/\d{4}", raw_stripped):
        print("Rejected by rule 4")
        continue
    print("Accepted!")
