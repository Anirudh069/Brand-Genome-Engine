#!/usr/bin/env python3
"""
build_embeddings_index.py – Aggregate text-level embeddings to brand-level
and build a nearest-neighbour index for competitor retrieval.

Usage
-----
  python -m scripts.build_embeddings_index \
      --features data/processed/features.parquet \
      --out_dir  embeddings/ \
      --k 5

Outputs
-------
* ``embeddings/brand_profile_index.faiss`` (or ``.pkl`` for sklearn backend)
* ``embeddings/metadata.json`` – maps row position → brand info.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarking.retrieval import (
    backend_name,
    build_index,
    save_index,
    query,
)

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"brand_id", "brand_name", "embedding"}


# ── Helpers ───────────────────────────────────────────────────────────────

def _parse_embedding(raw: object) -> list[float] | None:
    """
    Flexibly parse an embedding column value into a list[float].

    The parquet file may store the column as:
    * a Python ``list[float]`` (PyArrow list type)
    * a NumPy array
    * a JSON-encoded string  (``"[0.1, 0.2, ...]"``)

    Returns ``None`` if the value is unusable.
    """
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    return None


def _aggregate_brand_embeddings(
    df: pd.DataFrame,
) -> tuple[list[list[float]], list[dict]]:
    """
    Mean-pool text-level embeddings to one vector per brand.

    Returns
    -------
    brand_embeddings : list[list[float]]
        One 384-d vector per brand.
    metadata : list[dict]
        Per-brand metadata: ``{brand_id, brand_name, n_texts}``.
    """
    brand_embeddings: list[list[float]] = []
    metadata: list[dict] = []

    grouped = df.groupby("brand_id", sort=True)

    for brand_id, group in grouped:
        vecs: list[list[float]] = []
        for raw in group["embedding"]:
            parsed = _parse_embedding(raw)
            if parsed is not None and any(v != 0.0 for v in parsed):
                vecs.append(parsed)

        if not vecs:
            logger.warning(
                "Brand %s has no valid embeddings — skipping.", brand_id
            )
            continue

        mean_vec = np.mean(vecs, axis=0).astype(np.float32).tolist()
        brand_name = group["brand_name"].iloc[0]

        brand_embeddings.append(mean_vec)
        metadata.append(
            {
                "brand_id": str(brand_id),
                "brand_name": str(brand_name),
                "n_texts": len(vecs),
            }
        )

    return brand_embeddings, metadata


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a brand-level embedding index for competitor retrieval."
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features.parquet",
        help="Path to the features parquet file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="embeddings/",
        help="Directory to write the index and metadata.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbours for the demo query (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    args = parse_args(argv)
    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    k = args.k

    # ── Load ──────────────────────────────────────────────────────────────
    if not features_path.exists():
        logger.error("Features file not found: %s", features_path)
        sys.exit(1)

    logger.info("Loading features from %s …", features_path)
    df = pd.read_parquet(features_path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error(
            "Missing required columns: %s (found: %s)",
            sorted(missing),
            sorted(df.columns.tolist()),
        )
        sys.exit(1)

    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    # ── Aggregate ─────────────────────────────────────────────────────────
    brand_embeddings, metadata = _aggregate_brand_embeddings(df)
    n_brands = len(brand_embeddings)
    if n_brands == 0:
        logger.error("No brands with valid embeddings — aborting.")
        sys.exit(1)

    logger.info("Aggregated embeddings for %d brands.", n_brands)

    # ── Build index ───────────────────────────────────────────────────────
    index = build_index(brand_embeddings, metric="cosine")
    back = backend_name()

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = ".faiss" if back == "faiss" else ".pkl"
    index_path = out_dir / f"brand_profile_index{ext}"
    save_index(index, str(index_path))

    meta_path = out_dir / "metadata.json"
    # Map row_id (int) → brand info
    meta_map = {str(i): m for i, m in enumerate(metadata)}
    meta_path.write_text(json.dumps(meta_map, indent=2))
    logger.info("Metadata saved → %s", meta_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Brands indexed : {n_brands}")
    print(f"  Backend        : {back}")
    print(f"  Index file     : {index_path}")
    print(f"  Metadata file  : {meta_path}")
    print("=" * 60)

    # ── Demo query (first brand) ──────────────────────────────────────────
    if n_brands >= 2:
        ids, dists = query(index, brand_embeddings[0], k=min(k, n_brands))
        print(f"\n  Demo: nearest {len(ids)} brands to '{metadata[0]['brand_name']}':")
        for rank, (idx, dist) in enumerate(zip(ids, dists), 1):
            print(f"    {rank}. {metadata[idx]['brand_name']}  (distance={dist:.4f})")
        print()


if __name__ == "__main__":
    main()
