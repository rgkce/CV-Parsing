import os
import re
import pickle
import logging
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

from .config import BM25_DIR, BM25_INDEX_PATH

logger = logging.getLogger(__name__)

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenizer that lowercases and splits text into words.
    Handles Turkish characters properly by lowercasing before regex split.
    """
    if not text:
        return []
    # Lowercase first
    text = text.lower()
    # Replace Turkish i variants correctly if needed, though default lower() is often okay enough for BM25 
    # if queries undergo the exact same process.
    text = text.replace('İ', 'i').replace('I', 'ı')
    
    # Split by non-alphanumeric
    tokens = re.findall(r'\w+', text)
    
    # Remove extremely common generic titles that ruin keyword matching
    stopwords = {"mühendisi", "mühendis", "mühendisliği", "uzmanı", "öğrencisi", "geliştirici"}
    tokens = [t for t in tokens if t not in stopwords]
    
    return tokens

def build_bm25_index(dataset: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[str]]:
    """
    Builds a BM25Okapi index from the raw_text of the candidates.
    Returns the bm25 object and the ordered list of resume_ids.
    """
    logger.info("Building BM25 index over full resume texts...")
    resume_ids = []
    tokenized_corpus = []
    
    for candidate in dataset:
        rid = candidate.get("resume_id", "unknown")
        raw_text = candidate.get("raw_text", "")
        
        # We index the full raw_text because keywords can appear anywhere
        tokens = tokenize_text(raw_text)
        
        resume_ids.append(rid)
        tokenized_corpus.append(tokens)
        
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"Built BM25 index for {len(resume_ids)} documents.")
    return bm25, resume_ids

def save_bm25_index(bm25: BM25Okapi, resume_ids: List[str]) -> None:
    """
    Saves the BM25 index and corresponding resume_ids to disk.
    """
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    
    data = {
        "bm25": bm25,
        "resume_ids": resume_ids
    }
    
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(data, f)
        
    logger.info(f"Saved BM25 index to {BM25_INDEX_PATH}")

def load_bm25_index() -> Tuple[BM25Okapi, List[str]]:
    """
    Loads the BM25 index from disk.
    """
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}")
        
    with open(BM25_INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
        
    return data["bm25"], data["resume_ids"]
