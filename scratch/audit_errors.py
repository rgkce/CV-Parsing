import json
import re

with open("cv-parser-script/final_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total parsed records: {len(data)}")

# 1. Check for contact details/labels/headings in 'other' section
contact_patterns = [
    r"\btelefon\b", r"\bphone\b", r"\be-?posta\b", r"\bemail\b", r"\bmail\b",
    r"\blinkedin\b", r"\bgithub\b", r"\biletişim\b", r"\biletisim\b", r"\badres\b",
    r"\baddress\b", r"\bsosyal\b"
]
contact_regex = re.compile("|".join(contact_patterns), re.IGNORECASE)

print("\n--- AUDIT: Contact details / headings leaking into 'other' section ---")
found_contact_leak = False
for idx, r in enumerate(data):
    other = r.get("sections", {}).get("other", "")
    if not other:
        continue
    leaks = []
    for line in other.splitlines():
        if contact_regex.search(line):
            leaks.append(line.strip())
    if leaks:
        print(f"Candidate: {r.get('file_path')} ({idx})")
        print(f"  Leaks: {leaks[:10]}")
        found_contact_leak = True

# 2. Check for certificates/education headings or terms in 'other' section
cert_patterns = [
    r"\bkurs\b", r"\bseminer\b", r"\bbelge\b", r"\beğitim\b", r"\bcertificate\b", r"\bcourse\b"
]
cert_regex = re.compile("|".join(cert_patterns), re.IGNORECASE)

print("\n--- AUDIT: Certificate/Course headings/content in 'other' section ---")
found_cert_leak = False
for idx, r in enumerate(data):
    other = r.get("sections", {}).get("other", "")
    if not other:
        continue
    leaks = []
    for line in other.splitlines():
        # Look for headers especially
        if line.startswith("---") and cert_regex.search(line):
            leaks.append(line.strip())
        elif cert_regex.search(line) and len(line) < 50:
            # Short lines that look like headers or category labels
            leaks.append(line.strip())
    if leaks:
        print(f"Candidate: {r.get('file_path')} ({idx})")
        print(f"  Cert terms in 'other': {leaks[:10]}")
        found_cert_leak = True

# 3. Check for email format/typo issues
print("\n--- AUDIT: Candidate contact details overview ---")
for idx, r in enumerate(data):
    contact = r.get("contact", {})
    email = contact.get("email", "")
    phone = contact.get("phone", "")
    # Check for suspicious emails or phones
    suspicious = []
    if email and not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        suspicious.append(f"Invalid format: {email}")
    if email and any(domain in email for domain in ["gmal", "qmail", "gqmail", "hotmal", "gamil", "gmial"]):
        suspicious.append(f"Possible domain typo: {email}")
    if email and any(char in email.split("@")[0] for char in ["✉", "☎", "📞"]):
        suspicious.append(f"Icon in local part: {email}")
    if phone and re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b", phone):
        suspicious.append(f"Birth date in phone: {phone}")
    
    if suspicious:
        print(f"Candidate: {r.get('file_path')} ({idx})")
        for s in suspicious:
            print(f"  Suspicious: {s}")

# 4. Check for experience years of 0 vs their first job dates
print("\n--- AUDIT: Experience Years and Experience Section ---")
for idx, r in enumerate(data):
    exp_years = r.get("sections", {}).get("years_of_experience")
    exp_text = r.get("sections", {}).get("experience", "")
    if exp_years is not None:
        try:
            years = int(exp_years)
        except ValueError:
            years = -1
        # Check if experience text has dates but years is 0
        if years == 0 and exp_text and any(yr in exp_text for yr in ["201", "202", "200"]):
            # Let's print to check if this is an internship or a real job
            print(f"Candidate: {r.get('file_path')} ({idx}) - Years of Exp: {exp_years}")
            print(f"  Experience text snapshot: {exp_text[:200].replace(chr(10), ' | ')}")
