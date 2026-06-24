"""
config.py — Configuration for the Candidate Ranking System (Milestone 4)
=========================================================================

Scoring weights, LLM settings, output paths, and bilingual keyword
dictionaries for job-description parsing.
"""

from pathlib import Path

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "ranking_outputs"

# ─────────────────────────────────────────────
#  SCORING WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────

DEFAULT_SCORING_WEIGHTS = {
    "skills":     0.40,
    "experience": 0.35,
    "education":  0.15,
    "soft_skill": 0.10,
}

# ─────────────────────────────────────────────
#  LLM SETTINGS
# ─────────────────────────────────────────────

LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3          # Low temperature for consistent, factual output
LLM_MAX_TOKENS = 1024

# ─────────────────────────────────────────────
#  RETRIEVAL (how many candidates to rank)
# ─────────────────────────────────────────────

DEFAULT_TOP_K = 30
MIN_M3_RETRIEVAL_SCORE = 0.40  # Threshold to filter M3 results before passing to M4

# ─────────────────────────────────────────────
#  BILINGUAL KEYWORD DICTIONARIES
# ─────────────────────────────────────────────

# Technical skills — common terms found in Turkish and English JDs
TECHNICAL_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c",
    "ruby", "go", "golang", "rust", "swift", "kotlin", "scala", "r",
    "php", "perl", "matlab", "dart", "lua", "shell", "bash",
    # Web / Frontend
    "html", "css", "react", "reactjs", "react.js", "angular", "vue",
    "vuejs", "vue.js", "next.js", "nextjs", "svelte", "jquery",
    "bootstrap", "tailwind", "sass", "less", "webpack",
    # Backend / Frameworks
    "node.js", "nodejs", "express", "django", "flask", "fastapi",
    "spring", "spring boot", ".net", "asp.net", "laravel", "rails",
    # Data / ML / AI
    "machine learning", "deep learning", "makine öğrenmesi",
    "derin öğrenme", "yapay zeka", "artificial intelligence",
    "nlp", "natural language processing", "doğal dil işleme",
    "computer vision", "görüntü işleme", "tensorflow", "pytorch",
    "keras", "scikit-learn", "sklearn", "pandas", "numpy", "scipy",
    "opencv", "huggingface", "transformers", "llm",
    "data science", "veri bilimi", "data analysis", "veri analizi",
    "big data", "büyük veri", "hadoop", "spark", "kafka",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "k8s", "terraform", "ansible", "jenkins", "ci/cd", "devops",
    "linux", "git", "github", "gitlab",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "oracle", "sqlite", "cassandra", "dynamodb", "firebase",
    # Design / Creative
    "adobe photoshop", "photoshop", "illustrator", "figma", "sketch",
    "adobe xd", "indesign", "after effects", "premiere pro",
    "blender", "autocad", "solidworks", "revit", "3ds max",
    # Engineering
    "autocad", "sta4cad", "idecad", "sap2000", "etabs", "tekla",
    "matlab", "simulink", "catia", "ansys",
    # Mobile
    "android", "ios", "flutter", "react native", "swiftui",
    # Other
    "rest api", "graphql", "microservices", "agile", "scrum",
    "jira", "confluence", "power bi", "tableau", "excel",
}

# Soft skills — bilingual
SOFT_SKILLS = {
    # English
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "time management", "adaptability",
    "creativity", "attention to detail", "analytical thinking",
    "project management", "presentation", "negotiation",
    "conflict resolution", "decision making", "mentoring",
    "collaboration", "self-motivation", "work ethic",
    "organizational skills", "multitasking", "flexibility",
    # Turkish
    "iletişim", "iletisim", "takım çalışması", "takim calismasi",
    "liderlik", "problem çözme", "problem cozme",
    "eleştirel düşünme", "elestirel dusunme",
    "zaman yönetimi", "zaman yonetimi",
    "uyum sağlama", "uyum saglama",
    "yaratıcılık", "yaraticilik",
    "detaylara dikkat", "analitik düşünme", "analitik dusunme",
    "proje yönetimi", "proje yonetimi",
    "sunum", "müzakere", "muzakere",
    "karar verme", "mentorluk",
    "işbirliği", "isbirligi",
    "öz motivasyon", "oz motivasyon",
    "organizasyon", "çoklu görev", "coklu gorev",
    "esneklik",
}

# Education levels — mapped for degree matching
EDUCATION_LEVELS = {
    # English
    "bachelor": "bachelor",
    "bachelor's": "bachelor",
    "bachelors": "bachelor",
    "bs": "bachelor",
    "bsc": "bachelor",
    "b.sc": "bachelor",
    "b.s.": "bachelor",
    "ba": "bachelor",
    "b.a.": "bachelor",
    "undergraduate": "bachelor",
    "master": "master",
    "master's": "master",
    "masters": "master",
    "ms": "master",
    "msc": "master",
    "m.sc": "master",
    "m.s.": "master",
    "ma": "master",
    "m.a.": "master",
    "mba": "master",
    "graduate": "master",
    "postgraduate": "master",
    "phd": "phd",
    "ph.d": "phd",
    "ph.d.": "phd",
    "doctorate": "phd",
    "doctoral": "phd",
    # Turkish
    "lisans": "bachelor",
    "ön lisans": "associate",
    "önlisans": "associate",
    "üniversite": "bachelor",
    "yüksek lisans": "master",
    "yükseklisans": "master",
    "yuksek lisans": "master",
    "doktora": "phd",
    "mezun": "bachelor",
    "mezuniyet": "bachelor",
}

# Education fields
EDUCATION_FIELDS = {
    # English
    "computer science", "computer engineering", "software engineering",
    "information technology", "electrical engineering",
    "mechanical engineering", "civil engineering",
    "industrial engineering", "chemical engineering",
    "data science", "artificial intelligence",
    "mathematics", "statistics", "physics",
    "business administration", "economics", "finance",
    "marketing", "management", "architecture",
    "graphic design", "visual design",
    # Turkish
    "bilgisayar mühendisliği", "bilgisayar muhendisligi",
    "yazılım mühendisliği", "yazilim muhendisligi",
    "bilgisayar bilimleri", "bilişim", "bilisim",
    "elektrik mühendisliği", "elektrik elektronik",
    "makine mühendisliği", "makine muhendisligi",
    "inşaat mühendisliği", "insaat muhendisligi",
    "endüstri mühendisliği", "endustri muhendisligi",
    "kimya mühendisliği", "kimya muhendisligi",
    "veri bilimi", "yapay zeka",
    "matematik", "istatistik", "fizik",
    "işletme", "isletme", "iktisat", "ekonomi", "finans",
    "pazarlama", "yönetim", "yonetim", "mimarlık", "mimarlik",
    "grafik tasarım", "grafik tasarim", "görsel tasarım",
    "görsel iletişim", "gorsel iletisim",
}

# Experience-related keywords / patterns
EXPERIENCE_KEYWORDS = {
    # English
    "experience", "years", "year", "senior", "junior", "mid-level",
    "entry-level", "intern", "internship", "lead", "manager",
    "director", "head", "principal", "staff", "expert",
    # Turkish
    "deneyim", "tecrübe", "tecrube", "yıl", "yil", "kıdemli", "kidemli",
    "uzman", "stajyer", "staj", "müdür", "mudur", "şef", "sef",
    "yönetici", "yonetici", "sorumlu", "başkan", "baskan",
}

# Required vs Preferred signal words
REQUIRED_SIGNALS = {
    # English
    "required", "must have", "must", "mandatory", "essential",
    "necessary", "requirement", "need", "needed",
    # Turkish
    "gerekli", "zorunlu", "şart", "sart", "olmalı", "olmali",
    "aranan", "istenen", "beklenen",
}

PREFERRED_SIGNALS = {
    # English
    "preferred", "nice to have", "bonus", "plus", "desired",
    "advantageous", "optional", "ideally",
    # Turkish
    "tercih edilir", "tercih sebebi", "avantaj", "artı", "arti",
    "olması tercih", "olmasi tercih", "iyi olur",
}
