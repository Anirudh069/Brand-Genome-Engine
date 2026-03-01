"""
Tests for scripts/run_feature_extraction.py

Verifies that ``main()`` can:
1. Read from a temporary SQLite DB
2. Read from a temporary CSV
3. Produce a valid Parquet file with the expected columns
4. Report errors for missing tables / columns

Note: This module triggers sentence-transformers model loading via the
feature extraction pipeline.  Marked ``requires_model`` to avoid segfault
when combined with faiss tests on CPython 3.9 + macOS.
"""

from __future__ import annotations

import os
import sqlite3
import textwrap

import pandas as pd
import pytest

pytestmark = pytest.mark.requires_model

from scripts.run_feature_extraction import (
    COLUMN_ORDER,
    REQUIRED_DB_COLS,
    _load_from_csv,
    _load_from_db,
    main,
    run,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_ROWS = [
    ("t1", "b1", "Rolex", "Rolex is an iconic luxury watch brand."),
    ("t2", "b1", "Rolex", "Precision and craftsmanship define every piece."),
    ("t3", "b2", "Casio", "Affordable and reliable digital watches."),
]

REQUIRED_COLUMNS = {
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
}


@pytest.fixture()
def sample_db(tmp_path):
    """Create a temporary SQLite DB with a brand_texts table."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE brand_texts (
            text_id   TEXT PRIMARY KEY,
            brand_id  TEXT,
            brand_name TEXT,
            text      TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO brand_texts VALUES (?, ?, ?, ?);",
        SAMPLE_ROWS,
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def sample_csv(tmp_path):
    """Create a temporary CSV with brand text rows."""
    csv_path = str(tmp_path / "test.csv")
    df = pd.DataFrame(SAMPLE_ROWS, columns=["text_id", "brand_id", "brand_name", "text"])
    df.to_csv(csv_path, index=False)
    return csv_path


# ── DB loading ────────────────────────────────────────────────────────────

class TestLoadFromDB:
    def test_loads_all_rows(self, sample_db):
        df = _load_from_db(sample_db, "brand_texts")
        assert len(df) == len(SAMPLE_ROWS)

    def test_respects_limit(self, sample_db):
        df = _load_from_db(sample_db, "brand_texts", limit=2)
        assert len(df) == 2

    def test_missing_table_exits(self, sample_db):
        with pytest.raises(SystemExit):
            _load_from_db(sample_db, "nonexistent_table")

    def test_missing_db_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _load_from_db(str(tmp_path / "no_such.db"), "brand_texts")

    def test_missing_columns_exits(self, tmp_path):
        db_path = str(tmp_path / "bad.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE brand_texts (id TEXT, foo TEXT);")
        conn.close()
        with pytest.raises(SystemExit):
            _load_from_db(db_path, "brand_texts")


# ── CSV loading ───────────────────────────────────────────────────────────

class TestLoadFromCSV:
    def test_loads_all_rows(self, sample_csv):
        df = _load_from_csv(sample_csv)
        assert len(df) == len(SAMPLE_ROWS)

    def test_respects_limit(self, sample_csv):
        df = _load_from_csv(sample_csv, limit=1)
        assert len(df) == 1

    def test_missing_csv_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _load_from_csv(str(tmp_path / "nope.csv"))


# ── End-to-end via main() ────────────────────────────────────────────────

class TestMainDB:
    """Run main() against a temp DB and verify the output Parquet."""

    def test_produces_parquet_from_db(self, sample_db, tmp_path):
        out = str(tmp_path / "out" / "features.parquet")
        main([
            "--db", sample_db,
            "--table", "brand_texts",
            "--out", out,
        ])
        assert os.path.isfile(out)

        result = pd.read_parquet(out)
        assert len(result) == len(SAMPLE_ROWS)
        assert REQUIRED_COLUMNS.issubset(set(result.columns))

    def test_column_order_is_stable(self, sample_db, tmp_path):
        out = str(tmp_path / "features.parquet")
        main(["--db", sample_db, "--out", out])
        result = pd.read_parquet(out)
        # The first N columns must match COLUMN_ORDER
        actual_prefix = list(result.columns[: len(COLUMN_ORDER)])
        assert actual_prefix == COLUMN_ORDER

    def test_limit_flag(self, sample_db, tmp_path):
        out = str(tmp_path / "features.parquet")
        main(["--db", sample_db, "--out", out, "--limit", "1"])
        result = pd.read_parquet(out)
        assert len(result) == 1


class TestMainCSV:
    """Run main() against a temp CSV and verify the output Parquet."""

    def test_produces_parquet_from_csv(self, sample_csv, tmp_path):
        out = str(tmp_path / "features.parquet")
        main([
            "--csv", sample_csv,
            "--out", out,
        ])
        assert os.path.isfile(out)

        result = pd.read_parquet(out)
        assert len(result) == len(SAMPLE_ROWS)
        assert REQUIRED_COLUMNS.issubset(set(result.columns))


# ── Feature value sanity ──────────────────────────────────────────────────

class TestFeatureValues:
    """Quick sanity checks on the extracted feature values."""

    def test_sentiment_in_range(self, sample_db, tmp_path):
        out = str(tmp_path / "features.parquet")
        main(["--db", sample_db, "--out", out])
        result = pd.read_parquet(out)
        assert result["sentiment"].between(0.0, 1.0).all()

    def test_formality_in_range(self, sample_db, tmp_path):
        out = str(tmp_path / "features.parquet")
        main(["--db", sample_db, "--out", out])
        result = pd.read_parquet(out)
        assert result["formality"].between(0.0, 1.0).all()

    def test_embedding_length(self, sample_db, tmp_path):
        out = str(tmp_path / "features.parquet")
        main(["--db", sample_db, "--out", out])
        result = pd.read_parquet(out)
        from src.feature_extraction.text_features import EMBEDDING_DIM
        for emb in result["embedding"]:
            assert len(emb) == EMBEDDING_DIM

    def test_brands_detected(self, sample_db, tmp_path):
        out = str(tmp_path / "features.parquet")
        main(["--db", sample_db, "--out", out])
        result = pd.read_parquet(out)
        assert set(result["brand_name"].unique()) == {"Rolex", "Casio"}
