#!/usr/bin/env python3
"""
query_competitors.py – Look up the nearest competitor brands for a given brand.

Usage
-----
  python -m scripts.query_competitors --brand_name Rolex
  python -m scripts.query_competitors --brand_id rolex --k 3
  python -m scripts.query_competitors \\
      --features data/processed/features.parquet \\
      --index    embeddings/brand_profile_index.faiss \\
      --metadata embeddings/metadata.json \\
      --brand_name "TAG Heuer" --k 5

Output
------
Prints a table of the *k* most similar brands (excluding the query brand
itself) with columns: rank, brand_id, brand_name, cosine_distance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarking.retrieval import load_index, query

logger = logging.getLogger(__name__)

# Re-use the same aggregation helpers from build_embeddings_index
from scripts.build_embeddings_index import (
    _aggregate_brand_embeddings,
    _REQUIRED_COLUMNS,
)


# ── Result dataclass ─────────────────────────────────────────────────────

class CompetitorMatch:
    """One row in the results table."""

    __slots__ = ("rank", "brand_id", "brand_name", "distance")

    def __init__(self, rank: int, brand_id: str, brand_name: str, distance: float) -> None:
        self.rank = rank
        self.brand_id = brand_id
        self.brand_name = brand_name
        self.distance = distance

    def __repr__(self) -> str:
        return (
            f"CompetitorMatch(rank={self.rank}, brand_id={self.brand_id!r}, "
            f"brand_name={self.brand_name!r}, distance={self.distance:.6f})"
        )


# ── Core logic ────────────────────────────────────────────────────────────

def find_competitors(
    features_path: str,
    index_path: str,
    metadata_path: str,
    brand_id: str | None = None,
    brand_name: str | None = None,
    k: int = 5,
) -> list[CompetitorMatch]:
    """
    Return the *k* nearest competitor brands for a query brand.

    Exactly one of *brand_id* or *brand_name* must be supplied.

    Parameters
    ----------
    features_path : str
        Path to the features parquet (needs brand_id, brand_name, embedding).
    index_path : str
        Path to the saved FAISS / pkl index file.
    metadata_path : str
        Path to ``metadata.json`` produced by ``build_embeddings_index``.
    brand_id : str | None
        Query brand by ID.
    brand_name : str | None
        Query brand by name (case-insensitive match).
    k : int
        Number of competitor results to return.

    Returns
    -------
    list[CompetitorMatch]

    Raises
    ------
    FileNotFoundError
        If any of the input files are missing.
    ValueError
        If both / neither of brand_id / brand_name are given, or the
        brand cannot be found.
    """
    # ── Validate args ─────────────────────────────────────────────────────
    if (brand_id is None) == (brand_name is None):
        raise ValueError("Supply exactly one of --brand_id or --brand_name.")

    # ── Check files exist ─────────────────────────────────────────────────
    for label, path in [
        ("features", features_path),
        ("index", index_path),
        ("metadata", metadata_path),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    # ── Load metadata ─────────────────────────────────────────────────────
    with open(metadata_path) as fh:
        meta_map: dict[str, dict] = json.load(fh)

    # ── Load index ────────────────────────────────────────────────────────
    index = load_index(index_path)

    # Sanity: metadata rows must match index size
    if len(meta_map) != index.n:
        raise ValueError(
            f"Metadata has {len(meta_map)} entries but index has {index.n} vectors."
        )

    # ── Load features & aggregate ─────────────────────────────────────────
    df = pd.read_parquet(features_path)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in features: {sorted(missing)}")

    brand_embeddings, brand_meta = _aggregate_brand_embeddings(df)

    # ── Resolve query brand ───────────────────────────────────────────────
    query_vec: list[float] | None = None
    query_brand_id: str | None = None

    if brand_id is not None:
        target = str(brand_id).strip().lower()
        for emb, meta in zip(brand_embeddings, brand_meta):
            if meta["brand_id"].lower() == target:
                query_vec = emb
                query_brand_id = meta["brand_id"]
                break
    else:
        assert brand_name is not None
        target = brand_name.strip().lower()
        for emb, meta in zip(brand_embeddings, brand_meta):
            if meta["brand_name"].lower() == target:
                query_vec = emb
                query_brand_id = meta["brand_id"]
                break

    if query_vec is None:
        searched_by = f"brand_id={brand_id!r}" if brand_id else f"brand_name={brand_name!r}"
        available = [m["brand_id"] for m in brand_meta]
        raise ValueError(
            f"Brand not found ({searched_by}). "
            f"Available brand_ids: {available}"
        )

    # ── Query index (k+1 to allow removing self-match) ───────────────────
    fetch_k = min(k + 1, index.n)
    ids, dists = query(index, query_vec, k=fetch_k)

    # ── Build results, excluding self-match ───────────────────────────────
    results: list[CompetitorMatch] = []
    for idx, dist in zip(ids, dists):
        row_meta = meta_map.get(str(idx))
        if row_meta is None:
            continue
        if row_meta["brand_id"].lower() == query_brand_id.lower():
            continue  # skip self
        results.append(
            CompetitorMatch(
                rank=0,  # assigned below
                brand_id=row_meta["brand_id"],
                brand_name=row_meta["brand_name"],
                distance=round(dist, 6),
            )
        )
        if len(results) >= k:
            break

    # Assign final ranks
    for i, match in enumerate(results, 1):
        match.rank = i

    return results


# ── Pretty-print ──────────────────────────────────────────────────────────

def _print_table(
    query_label: str,
    matches: list[CompetitorMatch],
) -> None:
    """Print a nicely formatted results table to stdout."""
    print(f"\n  Competitors for: {query_label}")
    print(f"  {'─' * 56}")
    print(f"  {'Rank':<6} {'Brand ID':<20} {'Brand Name':<20} {'Distance':<10}")
    print(f"  {'─' * 56}")
    for m in matches:
        print(f"  {m.rank:<6} {m.brand_id:<20} {m.brand_name:<20} {m.distance:<10.6f}")
    if not matches:
        print("  (no matches found)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the brand embedding index for competitor brands."
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features.parquet",
        help="Path to the features parquet file.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="embeddings/brand_profile_index.faiss",
        help="Path to the FAISS (.faiss) or sklearn (.pkl) index file.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="embeddings/metadata.json",
        help="Path to the metadata JSON file.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--brand_id", type=str, default=None, help="Query by brand ID.")
    group.add_argument("--brand_name", type=str, default=None, help="Query by brand name.")

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of competitor results (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[CompetitorMatch]:
    """
    CLI entry-point.  Returns the match list so tests can inspect it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    args = parse_args(argv)

    try:
        matches = find_competitors(
            features_path=args.features,
            index_path=args.index,
            metadata_path=args.metadata,
            brand_id=args.brand_id,
            brand_name=args.brand_name,
            k=args.k,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    query_label = args.brand_name or args.brand_id
    _print_table(query_label, matches)
    return matches


if __name__ == "__main__":
    main()
