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
BM25_DIR = PROJECT_ROOT / "bm25_index"
BM25_INDEX_PATH = BM25_DIR / "bm25.pkl"

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

SECTIONS = ["title", "skills", "experience", "education", "summary", "projects"]

# ─────────────────────────────────────────────
#  DEFAULT WEIGHTS (must sum to 1.0)
# Default weights used if none are provided at query time.
# Skills and experience are given the highest weight to ensure that 
# candidates are matched on actual capabilities, preventing generic 
# title overlaps (e.g. 'mühendis') from causing false positives.
DEFAULT_WEIGHTS = {
    "title": 0.15,
    "skills": 0.35,
    "experience": 0.25,
    "education": 0.05,
    "summary": 0.10,
    "projects": 0.10,
}

# ─────────────────────────────────────────────
#  RETRIEVAL
# ─────────────────────────────────────────────

TOP_K = 10

# RRF (Reciprocal Rank Fusion) constant. Industry standard is 60.
RRF_K = 60

# Minimum cosine similarity for a section to count as "matched".
# Dense embedding models produce a high baseline (~0.75) even for
# unrelated texts, so the threshold must be well above that.
MATCH_THRESHOLD = 0.80

# Candidates below this final weighted score are filtered out
# entirely.  This prevents returning irrelevant results when
# no CV in the dataset truly matches the query.
MIN_SCORE_THRESHOLD = 0.79

# Maximum drop from the #1 score allowed.  Candidates whose score
# is more than this value below the top score are filtered out.
# This implements "relative relevance" — only return results that
# are close to the best match.
MAX_SCORE_DROP = 0.04

# Batch size for sentence-transformer encoding
ENCODE_BATCH_SIZE = 32

# Max characters per section before truncation.
# Long sections produce generic embeddings that match everything.
# 512 chars ≈ 100-130 tokens, enough to capture key content
# without diluting the signal with filler text.
MAX_SECTION_LENGTH = 512

