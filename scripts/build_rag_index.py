#!/usr/bin/env python3
"""
build_rag_index.py – Stage 5 canonical chunk-level RAG index build CLI.

Usage
-----
  python -m scripts.build_rag_index --db-path data/brand_data.db

Builds one FAISS IndexFlatIP per brand over brand_chunks embeddings and
atomically publishes the artifact directory (manifest.json, metadata.json,
indexes/<brand_id>.faiss).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from src.retrieval.rag_builder import DEFAULT_MODEL_NAME, RagBuildError, build_index

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stage 5 chunk-level RAG index.")
    parser.add_argument("--db-path", type=str, default="data/brand_data.db")
    parser.add_argument("--out-dir", type=str, default="data/processed/rag")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
    args = parse_args(argv)

    start = time.time()
    try:
        manifest = build_index(args.db_path, args.out_dir, model_name=args.model_name)
    except RagBuildError as exc:
        logger.error("RAG index build failed: %s", exc)
        sys.exit(1)
    duration = time.time() - start

    brands = manifest["brands"]
    print("\n" + "=" * 60)
    print("  RAG INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"  Model            : {manifest['model_name']}")
    print(f"  Embedding dim    : {manifest['embedding_dim']}")
    print(f"  Fingerprint      : {manifest['fingerprint']}")
    print(f"  Total chunks     : {manifest['chunk_count']}")
    print(f"  Brands indexed   : {len(brands)}")
    print(f"  Artifact dir     : {args.out_dir}")
    print(f"  Build duration   : {duration:.1f}s")
    print("-" * 60)
    for brand_id in sorted(brands):
        info = brands[brand_id]
        print(f"    {brand_id:<20} {info['brand_name']:<25} {info['count']:>4} chunks")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
