import json, sys
sys.path.append(r"c:\Users\rumeysagokce\Desktop\cv_parser_project - Kopya\cv-parser-script")
import cv_parser8

text = """Drone Operatörlüğü Yaptığım Projeler:
Lve From Fest Isparta
selfy Fest Isparta
TEKNİK SORUMLU
ÇEKİMLERİN LİSTESİ
FOTOĞRAFÇILIK
referanslar
"""
print("Testing _fallback_keyword_recovery on text:")
print(cv_parser8._fallback_keyword_recovery(text, ["projects", "skills"]))
