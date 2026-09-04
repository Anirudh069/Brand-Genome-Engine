#!/usr/bin/env python3
"""
evaluate_rag_retrieval.py – Stage 5 retrieval sanity evaluation.

Usage
-----
  python -m scripts.evaluate_rag_retrieval --db-path data/brand_data.db

Runs several handpicked queries against several brands and prints ranked
results so a human can qualitatively confirm retrieval is semantic (not
just row order), and that no cross-brand leakage occurs.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.retrieval.rag_service import RagError, retrieve_chunks

logger = logging.getLogger(__name__)

# (brand_id, query) pairs — vocabulary chosen to exist in the watch-brand corpus.
_QUERIES = [
    ("rolex", "precision and chronometer accuracy"),
    ("rolex", "heritage and tradition since founding"),
    ("omega", "innovation in materials and engineering"),
    ("omega", "heritage and tradition since founding"),
    ("cartier", "craftsmanship and elegant design"),
]


def _preview(text: str, length: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 1] + "…"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Run RAG retrieval sanity evaluation.")
    parser.add_argument("--db-path", type=str, default="data/brand_data.db")
    parser.add_argument("--out-dir", type=str, default="data/processed/rag")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    seen_orderings: dict[str, list[str]] = {}
    leakage = False

    for brand_id, query in _QUERIES:
        print("=" * 70)
        print(f"QUERY : {query!r}")
        print(f"BRAND : {brand_id}")
        try:
            result = retrieve_chunks(query, brand_id, top_k=args.top_k, db_path=args.db_path, artifact_dir=args.out_dir)
        except RagError as exc:
            print(f"  ERROR: {exc.detail}")
            sys.exit(1)

        print(f"TOP-{result['top_k']}:")
        ordering = []
        for item in result["results"]:
            if item["brand_id"] != brand_id:
                leakage = True
            ordering.append(item["chunk_id"])
            print(
                f"  rank={item['rank']}  score={item['score']:.4f}  "
                f"chunk_id={item['chunk_id']}  preview={_preview(item['chunk_text'])!r}"
            )
        seen_orderings[f"{brand_id}::{query}"] = ordering
        print()

    print("=" * 70)
    print("SANITY CHECKS")
    print(f"  Cross-brand leakage detected : {leakage}")

    same_brand_queries = [k for k in seen_orderings if k.startswith("rolex::")]
    if len(same_brand_queries) >= 2:
        orderings = [seen_orderings[k] for k in same_brand_queries]
        differ = any(o != orderings[0] for o in orderings[1:])
        print(f"  Different queries (same brand) produce different rankings : {differ}")

    print("=" * 70)


if __name__ == "__main__":
    main()
