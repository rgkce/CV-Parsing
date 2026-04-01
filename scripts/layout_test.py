import json


def detect_layout_issues(text):

    issues = []

    # Çok kısa text → extraction başarısız
    if len(text.split()) < 50:
        issues.append("low_text_extraction")

    # Çok uzun kesintisiz satır → layout bozulmuş olabilir
    if len(text) > 1000 and "\n" not in text:
        issues.append("no_line_breaks")

    # Aynı kelime tekrarları → column mixing olabilir
    words = text.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append("repetition_possible_layout_error")

    return issues


with open("output/pdf_parsed_clean.json", encoding="utf-8") as f:
    data = json.load(f)

log_results = []

for item in data:
    issues = detect_layout_issues(item["raw_text"])

    if issues:
        log_results.append({"file_name": item["file_name"], "issues": issues})

with open("output/layout_issues.json", "w", encoding="utf-8") as f:
    json.dump(log_results, f, indent=2)

print("Layout test tamamlandı")
