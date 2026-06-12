import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import re
from pathlib import Path
from cv_parser8 import extract_text_pdf

# Let's define the new robust LinkedIn extraction function
def test_extract_linkedin(raw_text: str) -> str:
    # Transliteration map for Turkish characters in URL slugs
    trans_map = str.maketrans("ışğüçöıİŞĞÜÇÖ", "isgucoiISGUCO")
    
    # 1. Clean systematic OCR typos for linkedin.com
    text_clean = re.sub(r'l\s*\s*n\s*k\s*e\s*d\s*\s*n', 'linkedin', raw_text, flags=re.I)
    text_clean = re.sub(r'l\s*ı\s*n\s*k\s*e\s*d\s*ı\s*n', 'linkedin', text_clean, flags=re.I)
    text_clean = re.sub(r'l\s*i\s*n\s*k\s*e\s*d\s*i\s*n', 'linkedin', text_clean, flags=re.I)
    text_clean = re.sub(r'l\x00nked\x00n', 'linkedin', text_clean, flags=re.I)
    
    # Try to find a standard linkedin.com/in/ pattern
    # We allow some non-ASCII letters in the initial match to capture Turkish characters, then we transliterate them
    match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_\-%.\u00C0-\u024F]+)', text_clean, re.I)
    if match:
        slug = match.group(1)
        # Transliterate Turkish characters to standard ASCII
        slug_ascii = slug.translate(trans_map)
        # Strip trailing punctuation/junk
        slug_ascii = slug_ascii.rstrip(".,;?!():\"'{}|-–")
        # Lowercase the slug
        slug_ascii = slug_ascii.lower()
        
        # Stop at any capitalized section keyword/label if merged (e.g. senamorbelDiller -> senamorbel)
        # Split on uppercase letters following lowercase (standard CamelCase/merged words)
        parts = re.split(r'(?=[A-Z])', slug_ascii) # Since it's lowercased, let's do it before lowercasing!
        
        # Let's do it on the original slug before lowercasing
        orig_slug = slug.translate(trans_map).rstrip(".,;?!():\"'{}|-–")
        # Split on CamelCase boundaries: e.g. "senamorbelDiller" -> "senamorbel", "Diller"
        # We find transitions from lowercase/digit to uppercase
        camel_split = re.split(r'(?<=[a-z0-9])(?=[A-Z])', orig_slug)
        if camel_split:
            cleaned_slug = camel_split[0]
        else:
            cleaned_slug = orig_slug
            
        # Additional safety check: if the slug ends with known noise words
        for noise in ["diller", "github", "tel", "email", "mail"]:
            if cleaned_slug.lower().endswith(noise) and len(cleaned_slug) > len(noise):
                cleaned_slug = cleaned_slug[:-len(noise)]
                
        cleaned_slug = cleaned_slug.lower().strip(".,;?!():\"'{}|-–")
        return f"https://www.linkedin.com/in/{cleaned_slug}"
    return ""

# Scan all PDF files in the data directory
pdf_dir = Path(r"C:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF")
pdfs = sorted(list(pdf_dir.glob("*.pdf")))

print(f"Scanning {len(pdfs)} PDFs for LinkedIn URLs:")
for p in pdfs:
    raw, _ = extract_text_pdf(str(p))
    extracted = test_extract_linkedin(raw)
    if extracted:
        print(f"  {p.name:<40} -> {extracted}")
