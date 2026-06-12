import sys
import os
sys.path.insert(0, os.path.abspath("."))
from cv_parser8 import sanitize_raw_text, repair_ocr_missing_i, normalize_text, clean_text, turkish_lower

original_text = "GİZEM KILINÇ\nEndüstri Mühendisi"
print("Original:", [ord(c) for c in original_text])

t = sanitize_raw_text(original_text)
print("After sanitize_raw_text:", [ord(c) for c in t])

t = repair_ocr_missing_i(t)
print("After repair_ocr_missing_i:", [ord(c) for c in t])

t = normalize_text(t)
print("After normalize_text:", [ord(c) for c in t])

# Let's see what clean_text(t) returns
t_clean = clean_text(t)
print("After clean_text:", [ord(c) for c in t_clean])
