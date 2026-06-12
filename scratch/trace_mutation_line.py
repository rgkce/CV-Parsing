import sys
from pathlib import Path
import traceback

# Add cv-parser-script to sys.path
sys.path.append(str(Path(__file__).parent.parent / "cv-parser-script"))

# Define a custom dictionary class that logs changes to its keys
class LoggingDict(dict):
    def __setitem__(self, key, value):
        if key == "email":
            print(f"\n[LoggingDict] 'email' key being set to: {value!r}")
            print("Stack trace:")
            traceback.print_stack(limit=5)
        super().__setitem__(key, value)

# Let's monkeypatch cv_parser8 or intercept its contact dictionary
import cv_parser8

# Keep a reference to the original process_cv
orig_process_cv = cv_parser8.process_cv

def process_cv_monitored(file_path: Path) -> dict:
    # We will hook extract_contact_info to return our LoggingDict
    orig_extract = cv_parser8.extract_contact_info
    
    def extract_contact_info_monitored(text: str) -> dict[str, str]:
        res = orig_extract(text)
        # Return a LoggingDict initialized with res
        logging_res = LoggingDict(res)
        return logging_res
        
    cv_parser8.extract_contact_info = extract_contact_info_monitored
    
    try:
        return orig_process_cv(file_path)
    finally:
        cv_parser8.extract_contact_info = orig_extract

# Run on the gokdeniz can.pdf CV
pdf_path = Path(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\data\PDF\gokdeniz can.pdf")
res = process_cv_monitored(pdf_path)
print("\nFinal email value in returned record:", res.get("contact", {}).get("email"))
