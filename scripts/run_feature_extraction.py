#!/usr/bin/env python3
"""
run_feature_extraction.py – Extract features from the ingestion DB (or a CSV)
and write a Parquet file ready for downstream ML / indexing.

Usage examples
--------------
  # From the default DB
  python -m scripts.run_feature_extraction --db watches.db --out data/processed/features.parquet

  # From a CSV override
  python -m scripts.run_feature_extraction --csv raw_texts_watches.csv --out features.parquet --limit 100
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.feature_extraction.text_features import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NUM_TOPICS,
    EMBEDDING_DIM,
    TextFeatureExtractor,
)

logger = logging.getLogger(__name__)

# ── Stable column order (matches ExtractedFeatures fields) ────────────────
COLUMN_ORDER: list[str] = [
    "text_id",
    "brand_id",
    "brand_name",
    "text",
    "sentiment",
    "formality",
    "readability_flesch",
    "avg_sentence_length",
    "punctuation_density",
    "vocab_diversity",
    "top_topics",
    "topic_weights",
    "embedding",
    "embedding_model",
]

REQUIRED_DB_COLS = {"text_id", "brand_id", "brand_name", "text"}


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_from_db(
    db_path: str,
    table: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read rows from *table* in a SQLite database."""
    if not os.path.isfile(db_path):
        print(f"ERROR: database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    try:
        # Check table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (table,),
        )
        if cursor.fetchone() is None:
            print(
                f"ERROR: table '{table}' does not exist in {db_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Check required columns
        col_info = conn.execute(f'PRAGMA table_info("{table}");').fetchall()
        existing_cols = {row[1] for row in col_info}
        missing = REQUIRED_DB_COLS - existing_cols
        if missing:
            print(
                f"ERROR: table '{table}' is missing required columns: "
                f"{sorted(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)

        query = f'SELECT text_id, brand_id, brand_name, text FROM "{table}"'
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    return df


def _load_from_csv(csv_path: str, limit: int | None = None) -> pd.DataFrame:
    """Read rows from a CSV file."""
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_DB_COLS - set(df.columns)
    if missing:
        print(
            f"ERROR: CSV is missing required columns: {sorted(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = df[["text_id", "brand_id", "brand_name", "text"]]
    if limit is not None:
        df = df.head(int(limit))

    return df


# ── Core logic ────────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    out_path: str,
    n_topics: int = DEFAULT_NUM_TOPICS,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> pd.DataFrame:
    """
    Extract features for every row in *df* and write a Parquet file.

    Returns the resulting DataFrame (useful for tests / chaining).
    """
    extractor = TextFeatureExtractor(
        embedding_model=embedding_model,
        num_topics=n_topics,
    )

    records: list[dict] = []
    for _, row in df.iterrows():
        text = str(row.get("text", "") or "")
        text_id = str(row.get("text_id", "") or "")
        brand_id = str(row.get("brand_id", "") or "")
        brand_name = str(row.get("brand_name", "") or "")

        features = extractor.extract_all_features(
            text=text,
            text_id=text_id,
            brand_id=brand_id,
            brand_name=brand_name,
        )
        records.append(asdict(features))

    result_df = pd.DataFrame(records)

    # Enforce stable column order (only keep columns that exist)
    ordered_cols = [c for c in COLUMN_ORDER if c in result_df.columns]
    # Append any extra columns not in COLUMN_ORDER (future-proofing)
    extra = [c for c in result_df.columns if c not in COLUMN_ORDER]
    result_df = result_df[ordered_cols + extra]

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result_df.to_parquet(out_path, index=False, engine="pyarrow")
    return result_df


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract brand-text features → Parquet",
    )
    parser.add_argument(
        "--db",
        default="watches.db",
        help="Path to SQLite database (default: watches.db)",
    )
    parser.add_argument(
        "--table",
        default="brand_texts",
        help="Table name inside the DB (default: brand_texts)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="If provided, read from this CSV instead of the DB",
    )
    parser.add_argument(
        "--out",
        default="data/processed/features.parquet",
        help="Output Parquet path (default: data/processed/features.parquet)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to process (default: all)",
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=DEFAULT_NUM_TOPICS,
        help=f"Number of topics to extract (default: {DEFAULT_NUM_TOPICS})",
    )
    parser.add_argument(
        "--embedding_model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence-transformers model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point – parse args, load data, extract, write parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    )

    args = build_parser().parse_args(argv)

    # ── Load ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if args.csv:
        logger.info("Reading from CSV: %s", args.csv)
        df = _load_from_csv(args.csv, limit=args.limit)
    else:
        logger.info("Reading from DB: %s  table=%s", args.db, args.table)
        df = _load_from_db(args.db, args.table, limit=args.limit)

    logger.info("Loaded %d rows", len(df))

    # ── Extract ───────────────────────────────────────────────────────
    result = run(
        df,
        out_path=args.out,
        n_topics=args.n_topics,
        embedding_model=args.embedding_model,
    )

    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────
    n_rows = len(result)
    n_brands = result["brand_name"].nunique() if n_rows else 0
    n_empty = int((result["text"].str.strip() == "").sum()) if n_rows else 0
    pct_empty = (n_empty / n_rows * 100) if n_rows else 0.0

    summary = (
        f"\n{'─' * 50}\n"
        f"  Rows processed : {n_rows}\n"
        f"  Unique brands  : {n_brands}\n"
        f"  Empty texts    : {n_empty} ({pct_empty:.1f}%)\n"
        f"  Runtime        : {elapsed:.2f}s\n"
        f"  Output         : {args.out}\n"
        f"{'─' * 50}"
    )
    print(summary)
    logger.info("Done.")


if __name__ == "__main__":
    main()
