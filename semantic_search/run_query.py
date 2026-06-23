"""
run_query.py — Example semantic search queries
===============================================

Usage::

    # Run with default sample queries
    python -m semantic_search.run_query

    # Run with a custom query
    python -m semantic_search.run_query --query "Python developer with ML experience"

    # Run with custom weights
    python -m semantic_search.run_query \\
        --query "İnşaat mühendisi" \\
        --weight-skills 0.4 \\
        --weight-experience 0.35 \\
        --weight-education 0.15 \\
        --weight-summary 0.05 \\
        --weight-projects 0.05

    # Change top-k
    python -m semantic_search.run_query --query "Data analyst" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Dict, List

from .config import DEFAULT_WEIGHTS, TOP_K
from .embeddings import load_embeddings, load_model
from .indexer import load_indexes
from .searcher import combine_scores, encode_query, search_all_sections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  SAMPLE QUERIES (Turkish + English)
# ─────────────────────────────────────────────

SAMPLE_QUERIES = [
    "Python developer with machine learning and data analysis experience",
    "İnşaat mühendisi, şantiye yönetimi ve proje deneyimi olan",
    "Grafik tasarımcı, Adobe Photoshop ve Illustrator bilen",
    "Software engineer experienced in web development and REST APIs",
    "Muhasebe ve finans alanında deneyimli",
]


def print_results(
    query: str,
    results: List[Dict],
    weights: Dict[str, float],
) -> None:
    """Pretty-print search results as a table."""
    print("\n" + "=" * 80)
    print(f"  QUERY: {query}")
    print(f"  WEIGHTS: {weights}")
    print("=" * 80)
    print(
        f"  {'Rank':<6}{'Name':<30}{'Resume ID':<40}{'Score':<10}{'Matched Sections'}"
    )
    print("  " + "-" * 106)

    for i, result in enumerate(results, 1):
        rid = result["resume_id"]
        name = result.get("name", "Bilinmeyen Aday")[:28]
        score = result["score"]
        matched = ", ".join(result["matched_sections"]) or "-"
        print(f"  {i:<6}{name:<30}{rid:<40}{score:<10.4f}{matched}")

        # Show per-section breakdown
        if result.get("section_scores"):
            parts = [
                f"{s}={v:.3f}"
                for s, v in sorted(result["section_scores"].items())
            ]
            print(f"  {'':>6}  +-- section scores: {', '.join(parts)}")

    print("=" * 80)


def run_single_query(
    model,
    indexes,
    resume_ids,
    query: str,
    weights: Dict[str, float],
    top_k: int,
    dataset: List[Dict] = None,
    bm25 = None,
    resume_ids_bm25 = None,
) -> List[Dict]:
    """Encode a query, search, combine scores, and return results."""
    query_emb = encode_query(model, query)
    section_results = search_all_sections(indexes, query_emb)
    results = combine_scores(
        section_results,
        resume_ids,
        weights=weights,
        query_text=query,
        dataset=dataset,
        top_k=top_k,
        bm25=bm25,
        resume_ids_bm25=resume_ids_bm25
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over CV sections",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Custom query string.  If omitted, runs sample queries.",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--weight-title", type=float, default=None)
    parser.add_argument("--weight-skills", type=float, default=None)
    parser.add_argument("--weight-experience", type=float, default=None)
    parser.add_argument("--weight-education", type=float, default=None)
    parser.add_argument("--weight-summary", type=float, default=None)
    parser.add_argument("--weight-projects", type=float, default=None)
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output results as JSON instead of a table",
    )
    args = parser.parse_args()

    # ── Build weights dict ────────────────────
    weights = dict(DEFAULT_WEIGHTS)
    overrides = {
        "title": args.weight_title,
        "skills": args.weight_skills,
        "experience": args.weight_experience,
        "education": args.weight_education,
        "summary": args.weight_summary,
        "projects": args.weight_projects,
    }
    any_override = False
    for section, val in overrides.items():
        if val is not None:
            weights[section] = val
            any_override = True

    # Re-normalise if any overrides were given
    if any_override:
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        logger.info("Using custom weights (normalised): %s", weights)

    # ── Load resources ────────────────────────
    logger.info("Loading embedding model...")
    model = load_model()

    logger.info("Loading embeddings from disk...")
    _, resume_ids = load_embeddings()

    logger.info("Loading FAISS indexes from disk...")
    indexes = load_indexes()

    logger.info("Loading dataset for lexical checks...")
    with open('final_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info("Loading BM25 keyword index...")
    from semantic_search.bm25_indexer import load_bm25_index
    bm25, resume_ids_bm25 = load_bm25_index()

    # ── Run queries ───────────────────────────
    queries = [args.query] if args.query else SAMPLE_QUERIES

    import os
    import hashlib
    from pathlib import Path
    
    out_dir = Path("search_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for query in queries:
        results = run_single_query(
            model, indexes, resume_ids, query,
            weights=weights, top_k=args.top_k,
            dataset=dataset,
            bm25=bm25,
            resume_ids_bm25=resume_ids_bm25
        )
        
        # Generate query ID
        h = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
        query_id = f"Q-{h}"
        
        all_results.append({"query_id": query_id, "query": query, "results": results})

        if not args.json_output:
            print_results(query, results, weights)
            
        # Save Text Report
        report_path = out_dir / f"{query_id}_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"  QUERY: {query}\n")
            f.write(f"  WEIGHTS: {weights}\n")
            f.write("=" * 80 + "\n")
            f.write(f"  {'Rank':<6}{'Name':<30}{'Resume ID':<40}{'Score':<10}{'Matched Sections'}\n")
            f.write("  " + "-" * 106 + "\n")
            for i, result in enumerate(results, 1):
                rid = result["resume_id"]
                name = result.get("name", "Bilinmeyen Aday")[:28]
                score = result["score"]
                matched = ", ".join(result["matched_sections"]) or "-"
                f.write(f"  {i:<6}{name:<30}{rid:<40}{score:<10.4f}{matched}\n")
                
                if result.get("section_scores"):
                    parts = [f"{s}={v:.3f}" for s, v in sorted(result["section_scores"].items())]
                    f.write(f"  {'':>6}  +-- section scores: {', '.join(parts)}\n")
            f.write("=" * 80 + "\n")
            
        logger.info("Saved text report -> %s", report_path)
            
        # Save JSON Report
        json_path = out_dir / f"{query_id}_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"query_id": query_id, "query": query, "weights": weights, "results": results}, f, ensure_ascii=False, indent=2)
            
        logger.info("Saved JSON results -> %s", json_path)

    if args.json_output:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
