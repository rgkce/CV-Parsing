import json
import sys

# Reconfigure stdout to use utf-8 if possible
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def print_safe(msg):
    try:
        # try printing normally
        print(msg)
    except UnicodeEncodeError:
        # print with replaced characters
        safe_msg = msg.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(safe_msg)

with open(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

# Build index by file_path stem
idx = {}
for d in data:
    fp = d.get("file_path", "")
    stem = fp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".pdf", "").lower()
    idx[stem] = d

errors = []

def get_field(candidate, field):
    if field in candidate.get("sections", {}):
        return candidate["sections"][field]
    if field in candidate.get("contact", {}):
        return candidate["contact"][field]
    return ""

def check(name_key, field, expected, label=""):
    for k, v in idx.items():
        if name_key.lower() in k:
            actual = get_field(v, field)
            if str(actual).strip().lower() != str(expected).strip().lower():
                errors.append(f"[FAIL] {k} -> {field}: expected '{expected}', got '{actual}' {label}")
            else:
                print_safe(f"  [OK] {k} -> {field} = '{actual}' {label}")
            return
    errors.append(f"[FAIL] Could not find candidate matching '{name_key}'")

print_safe("=" * 70)
print_safe("PHASE 9 VERIFICATION - Systemic Fixes (Safe Console Output)")
print_safe("=" * 70)

# 1. DATE/YoE CHECKS
print_safe("\n--- YoE Checks ---")
check("alihan tekin", "years_of_experience", "5", "(date parsing fix)")
check("sudenaz boyali", "years_of_experience", "2", "(date parsing fix)")
check("furkan karakurt", "years_of_experience", "4", "(date parsing fix)")
check("yuksel cosgun mobile", "years_of_experience", "5", "(date parsing fix)")

# 2. LANGUAGE SECTION CHECKS - verify languages are populated
print_safe("\n--- Language Section Checks ---")
for name_key in ["yagiz tokgoz", "ipek nur", "ilteris cansiz"]:
    found_cand = False
    for k, v in idx.items():
        if name_key.lower() in k:
            found_cand = True
            langs = get_field(v, "languages")
            certs = get_field(v, "certificates")
            if langs and len(str(langs).strip()) > 3:
                print_safe(f"  [OK] {k} -> languages populated ({len(str(langs))} chars)")
            else:
                errors.append(f"[FAIL] {k} -> languages is empty/too short: '{langs}'")
            # Check no language leakage in certificates
            lang_keywords = ["english", "turkish", "ingilizce", "türkçe", "turkce", "almanca", "german"]
            leaked = [lk for lk in lang_keywords if lk in str(certs).lower()]
            if leaked:
                errors.append(f"[WARN] {k} -> certificates may contain language leakage: {leaked}")
            else:
                print_safe(f"  [OK] {k} -> no language leakage in certificates")
            break
    if not found_cand:
        errors.append(f"[FAIL] Language check candidate not found matching '{name_key}'")

# 3. REFERENCES STRIPPING CHECK
print_safe("\n--- Reference Stripping Checks ---")
for k, v in idx.items():
    if "yuksel cosgun" in k:
        certs = str(get_field(v, "certificates"))
        other = str(get_field(v, "other"))
        full_text = certs + " " + other
        has_ref = "references" in full_text.lower() or "referanslar" in full_text.lower()
        status = "[WARN] contains references text" if has_ref else "[OK] clean"
        print_safe(f"  {status}: {k} -> certs({len(certs)}c) other({len(other)}c)")

# 4. REGRESSION CHECK
print_safe("\n--- Regression Check ---")
total = len(data)
print_safe(f"  Total candidates: {total}")

# Previously fixed candidates - regression check
print_safe("\n--- Previously Fixed Candidates Regression ---")
check("gokdeniz can", "email", "gokdenizcanofficial@gmail.com", "(Phase 6 fix)")
check("gul koc", "years_of_experience", "0", "(Phase 7 fix)")
check("gorkem tolu", "years_of_experience", "2", "(Phase 8 fix)")

# Print all YoE values for quick visual scan
print_safe("\n--- Full YoE Overview ---")
for k in sorted(idx.keys()):
    v = idx[k]
    title = get_field(v, "title")
    yoe = get_field(v, "years_of_experience")
    print_safe(f"  {k:45s} title={title:35s} yoe={yoe}")

print_safe("\n" + "=" * 70)
if errors:
    print_safe(f"RESULT: {len(errors)} ISSUES FOUND")
    for e in errors:
        print_safe(f"  {e}")
    sys.exit(1)
else:
    print_safe("RESULT: ALL CHECKS PASSED ✓")
    sys.exit(0)
