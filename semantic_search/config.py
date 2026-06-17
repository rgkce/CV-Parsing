"""
config.py — Central configuration for the Semantic Search System
================================================================

All paths, model settings, section definitions, default weights,
and tuning parameters live here.
"""

from pathlib import Path

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────

# Project root = parent of the semantic_search/ package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "final_dataset.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
FAISS_DIR = PROJECT_ROOT / "faiss_indexes"

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────

MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

# E5 models require these prefixes for best retrieval performance
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# ─────────────────────────────────────────────
#  SECTIONS
# ─────────────────────────────────────────────

SECTIONS = ["skills", "experience", "education", "summary", "projects"]

# ─────────────────────────────────────────────
#  DEFAULT WEIGHTS (must sum to 1.0)
# ─────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "skills":      0.35,
    "experience":  0.30,
    "education":   0.10,
    "summary":     0.15,
    "projects":    0.10,
}

# ─────────────────────────────────────────────
#  RETRIEVAL
# ─────────────────────────────────────────────

TOP_K = 10

# Minimum cosine similarity for a section to count as "matched"
MATCH_THRESHOLD = 0.3

# Batch size for sentence-transformer encoding
ENCODE_BATCH_SIZE = 32
