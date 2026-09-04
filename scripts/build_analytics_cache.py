"""
build_analytics_cache.py – Reproducible Stage 4 Analytics artifact builder.

Usage
-----
    python -m scripts.build_analytics_cache --db-path data/brand_data.db

Computes, from the canonical SQLite DB, ONLY the corpus-derived Analytics
components (does not touch source rows):
    - automatically derived Messaging Pillar keyword sets (5 fixed pillars)
    - TF-IDF pillar heatmap (brands x pillars)
    - chunk-level t-SNE sample (brand_chunks)
    - competitor tone distribution

and writes the artifact atomically to ``data/processed/analytics_cache.json``
(or ``--output-path``).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

from src.analytics.cache import build_and_save


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/brand_data.db")
    parser.add_argument("--output-path", default="data/processed/analytics_cache.json")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    if integrity != "ok":
        print(f"FAIL: PRAGMA integrity_check returned '{integrity}', aborting build.")
        return 1

    print(f"Building analytics artifact from {args.db_path} ...")
    artifact = build_and_save(args.db_path, args.output_path)

    print(f"Embedding mode: {artifact['embedding_mode']}")
    print(f"Brand texts: {artifact['source_counts']['brand_texts']}")
    print(f"Brand chunks: {artifact['source_counts']['brand_chunks']}")
    print(f"Competitors: {artifact['source_counts']['competitor_count']}")
    print(f"Pillars: {artifact['pillars']['names']}")
    for pillar, terms in artifact["pillars"]["keywords"].items():
        print(f"  {pillar}: {[t['term'] for t in terms]}")
    print(f"Heatmap shape: {len(artifact['heatmap']['brands'])} x {len(artifact['heatmap']['pillars'])}")
    print(f"t-SNE sampled points: {artifact['tsne']['sample_total']} (perplexity={artifact['tsne']['perplexity']})")
    print(f"Tone totals: {artifact['tone']['totals']}")
    print(f"Fingerprint: {artifact['fingerprint']}")
    print(f"Artifact written to: {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
