import json

with open(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script\final_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    fp = d["file_path"].lower()
    s = d["sections"]
    c = d["contact"]
    if "ilteris" in fp:
        print("ILTERIS languages:", repr(s["languages"]))
    if "yuksel cosgun mobile" in fp:
        print("YUKSEL MOBILE certificates:", repr(s["certificates"]))
    if "tolu" in fp:
        print("GORKEM: yoe=" + s["years_of_experience"] + ", title=" + s["title"])
    if "gokdeniz" in fp:
        print("GOKDENIZ: email=" + c["email"])
    if "gul koc" in fp:
        print("GUL KOC: yoe=" + s["years_of_experience"])
    if "alihan" in fp:
        print("ALIHAN: yoe=" + s["years_of_experience"])
    if "furkan karakurt" in fp:
        print("FURKAN: yoe=" + s["years_of_experience"])
    if "sudenaz" in fp:
        print("SUDENAZ: yoe=" + s["years_of_experience"])
