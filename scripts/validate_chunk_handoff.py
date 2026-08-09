#!/usr/bin/env python3
"""
validate_chunk_handoff.py - Non-destructive readiness check for handing the
populated ``brand_chunks`` table off to independent data-quality validation.

Opens the DB read-only (URI mode=ro) and never writes. Reports hard
failures (which cause a nonzero exit) separately from warnings (which do
not).

Usage:
  python -m scripts.validate_chunk_handoff
  python -m scripts.validate_chunk_handoff --db-path data/brand_data.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/brand_data.db")

MIN_TEXTS_PER_BRAND = 30
MIN_CHUNKS_PER_BRAND = 50
SHORT_CHUNK_THRESHOLD = 80
HARD_MAX_CHARS = 400

REQUIRED_TABLES = {
    "brands",
    "brand_texts_raw",
    "brand_texts",
    "brand_chunks",
    "brand_profile",
    "brand_profiles",
    "analysis_history",
}


def _fail(msgs, msg):
    msgs.append(msg)
    print(f"  x FAIL: {msg}", file=sys.stderr)


def _ok(msg):
    print(f"  v PASS: {msg}")


def _warn(msg):
    print(f"  ! WARN: {msg}")


def find_repo_db_files(repo_root: Path, canonical: Path) -> list[Path]:
    found = []
    for db_file in repo_root.rglob("*.db"):
        resolved = db_file.resolve()
        parts = db_file.parts
        if any(p in (".git", "node_modules", "venv", ".venv", "__pycache__") for p in parts):
            continue
        found.append(resolved)
    return found


def run_checks(db_path: str) -> tuple[bool, list[str], list[str]]:
    failures: list[str] = []
    warnings_out: list[str] = []

    print()
    print("=" * 70)
    print(f"  Chunk Handoff Validator - {db_path}")
    print("=" * 70)
    print()

    db_file = Path(db_path)
    if not db_file.exists():
        _fail(failures, f"Database file does not exist: {db_path}")
        return False, failures, warnings_out

    # [1] Exactly one physical repo DB
    print("[1] Canonical DB uniqueness")
    canonical = db_file.resolve()
    repo_root = Path(__file__).resolve().parent.parent
    all_dbs = find_repo_db_files(repo_root, canonical)
    others = [str(p) for p in all_dbs if p != canonical]
    if others:
        _fail(failures, f"Additional physical .db files found in repo: {others}")
    else:
        _ok(f"Exactly one physical .db file in repo: {canonical}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    # [2] Integrity check
    print()
    print("[2] SQLite integrity check")
    cur.execute("PRAGMA integrity_check;")
    result = cur.fetchone()[0]
    if result != "ok":
        _fail(failures, f"PRAGMA integrity_check returned: {result}")
    else:
        _ok("PRAGMA integrity_check = ok")

    # [3] Table presence
    print()
    print("[3] Table presence")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    missing = REQUIRED_TABLES - tables
    if missing:
        _fail(failures, f"Missing expected tables: {sorted(missing)}")
    else:
        _ok(f"All expected tables present: {sorted(REQUIRED_TABLES)}")

    # [4] Source text volume
    print()
    print("[4] Source corpus (brand_texts / brand_texts_raw)")
    cur.execute("SELECT COUNT(*) FROM brand_texts")
    total_texts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM brand_texts_raw")
    total_texts_raw = cur.fetchone()[0]
    _ok(f"brand_texts = {total_texts}, brand_texts_raw = {total_texts_raw}")

    cur.execute("SELECT brand_id, COUNT(*) FROM brand_texts GROUP BY brand_id ORDER BY brand_id")
    texts_per_brand = dict(cur.fetchall())
    below_texts = {b: n for b, n in texts_per_brand.items() if n < MIN_TEXTS_PER_BRAND}
    if below_texts:
        _fail(failures, f"Brands below {MIN_TEXTS_PER_BRAND} brand_texts: {below_texts}")
    else:
        _ok(f"All {len(texts_per_brand)} brands >= {MIN_TEXTS_PER_BRAND} brand_texts "
            f"(lowest={min(texts_per_brand.values()) if texts_per_brand else 0})")

    # [5] Chunk volume
    print()
    print("[5] Chunk volume (brand_chunks)")
    cur.execute("SELECT COUNT(*) FROM brand_chunks")
    total_chunks = cur.fetchone()[0]
    cur.execute("SELECT brand_id, COUNT(*) FROM brand_chunks GROUP BY brand_id ORDER BY brand_id")
    chunks_per_brand = dict(cur.fetchall())
    _ok(f"brand_chunks total = {total_chunks}")
    for b, n in chunks_per_brand.items():
        print(f"      {b:16s} {n}")
    below_chunks = {b: n for b, n in chunks_per_brand.items() if n < MIN_CHUNKS_PER_BRAND}
    if below_chunks:
        _fail(failures, f"Brands below {MIN_CHUNKS_PER_BRAND} brand_chunks: {below_chunks}")
    else:
        _ok(f"All {len(chunks_per_brand)} brands >= {MIN_CHUNKS_PER_BRAND} brand_chunks "
            f"(lowest={min(chunks_per_brand.values()) if chunks_per_brand else 0}, "
            f"highest={max(chunks_per_brand.values()) if chunks_per_brand else 0})")

    # [6] Blank / null chunks and provenance fields
    print()
    print("[6] Blank / null validation")
    cur.execute("SELECT COUNT(*) FROM brand_chunks WHERE chunk_text IS NULL OR TRIM(chunk_text) = ''")
    blank_chunks = cur.fetchone()[0]
    if blank_chunks:
        _fail(failures, f"{blank_chunks} chunks have NULL/blank chunk_text")
    else:
        _ok("0 blank/null chunk_text rows")

    for col in ("chunk_id", "text_id", "brand_id"):
        cur.execute(f"SELECT COUNT(*) FROM brand_chunks WHERE {col} IS NULL OR TRIM({col}) = ''")
        n = cur.fetchone()[0]
        if n:
            _fail(failures, f"{n} chunks have NULL/blank {col}")
        else:
            _ok(f"0 blank/null {col} rows")

    # [7] Source-link integrity
    print()
    print("[7] Source-link integrity")
    cur.execute(
        "SELECT COUNT(*) FROM brand_chunks c LEFT JOIN brand_texts t "
        "ON c.text_id = t.text_id WHERE t.text_id IS NULL"
    )
    orphans = cur.fetchone()[0]
    if orphans:
        _fail(failures, f"{orphans} brand_chunks rows have orphan text_id")
    else:
        _ok("0 orphan text_id references")

    cur.execute(
        "SELECT COUNT(*) FROM brand_chunks c JOIN brand_texts t "
        "ON c.text_id = t.text_id WHERE c.brand_id != t.brand_id"
    )
    brand_mismatches = cur.fetchone()[0]
    if brand_mismatches:
        _fail(failures, f"{brand_mismatches} chunks have brand_id mismatched vs source text")
    else:
        _ok("0 brand_id mismatches")

    cur.execute("PRAGMA table_info(brand_chunks)")
    chunk_cols = {row[1] for row in cur.fetchall()}
    cur.execute("PRAGMA table_info(brand_texts)")
    text_cols = {row[1] for row in cur.fetchall()}
    if {"brand_name", "source_type"} <= chunk_cols and {"brand_name", "source_type"} <= text_cols:
        cur.execute(
            "SELECT COUNT(*) FROM brand_chunks c JOIN brand_texts t ON c.text_id = t.text_id "
            "WHERE c.brand_name != t.brand_name OR c.source_type != t.source_type"
        )
        meta_mismatches = cur.fetchone()[0]
        if meta_mismatches:
            _fail(failures, f"{meta_mismatches} chunks have brand_name/source_type mismatched vs source")
        else:
            _ok("0 brand_name/source_type mismatches")

    # [8] Character-length validation
    print()
    print("[8] Character-length validation")
    cur.execute("SELECT chunk_id, chunk_text, char_count FROM brand_chunks")
    rows = cur.fetchall()
    mismatches = [r for r in rows if len(r[1]) != r[2]]
    if mismatches:
        _fail(failures, f"{len(mismatches)} chunks have char_count != len(chunk_text)")
    else:
        _ok("0 char_count mismatches")

    lengths = [len(r[1]) for r in rows]
    if lengths:
        min_len = min(lengths)
        max_len = max(lengths)
        median_len = statistics.median(lengths)
        mean_len = statistics.fmean(lengths)
        count_under = sum(1 for l in lengths if l < SHORT_CHUNK_THRESHOLD)
        count_over = sum(1 for l in lengths if l > HARD_MAX_CHARS)
    else:
        min_len = max_len = median_len = mean_len = 0
        count_under = count_over = 0

    _ok(f"min={min_len} median={median_len} mean={mean_len:.1f} max={max_len}")
    if count_over:
        _fail(failures, f"{count_over} chunks exceed hard max of {HARD_MAX_CHARS} chars")
    else:
        _ok(f"0 chunks exceed hard max of {HARD_MAX_CHARS} chars")
    if count_under:
        warnings_out.append(f"{count_under} chunks below {SHORT_CHUNK_THRESHOLD} chars (short-chunk warning, not a failure)")
        _warn(f"{count_under} chunks below {SHORT_CHUNK_THRESHOLD} chars")

    # [9] Chunk ID uniqueness
    print()
    print("[9] Chunk ID uniqueness")
    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)):
        _fail(failures, f"Duplicate chunk_id values detected ({len(ids)} rows, {len(set(ids))} distinct)")
    else:
        _ok(f"All {len(ids)} chunk_id values unique")

    # [10] Source coverage
    print()
    print("[10] Source coverage")
    cur.execute("SELECT text_id, text FROM brand_texts")
    text_rows = cur.fetchall()
    all_text_ids = {t[0] for t in text_rows}
    cur.execute("SELECT DISTINCT text_id FROM brand_chunks")
    covered_text_ids = {r[0] for r in cur.fetchall()}
    zero_chunk_ids = all_text_ids - covered_text_ids

    def _is_meaningful(text):
        return text is not None and text.strip() != ""

    text_map = dict(text_rows)
    meaningful_zero = [tid for tid in zero_chunk_ids if _is_meaningful(text_map.get(tid))]
    _ok(f"{len(covered_text_ids)}/{len(all_text_ids)} source texts represented in brand_chunks")
    if meaningful_zero:
        _fail(failures, f"{len(meaningful_zero)} non-blank source texts produced zero chunks: {meaningful_zero[:10]}")
    else:
        _ok("0 non-blank source texts with zero chunks")
    if zero_chunk_ids - set(meaningful_zero):
        warnings_out.append(
            f"{len(zero_chunk_ids) - len(meaningful_zero)} blank source texts legitimately produced zero chunks"
        )

    # [11] Duplicate-content warnings (non-destructive, report only)
    print()
    print("[11] Duplicate-content review (warnings only)")
    cur.execute("SELECT brand_id, text_id, text FROM brand_texts")
    src_groups = defaultdict(list)
    for brand_id, text_id, text in cur.fetchall():
        src_groups[(brand_id, text)].append(text_id)
    src_dupe_groups = {k: v for k, v in src_groups.items() if len(v) > 1}
    src_dupe_rows = sum(len(v) for v in src_dupe_groups.values())
    if src_dupe_groups:
        warnings_out.append(
            f"{len(src_dupe_groups)} duplicate source-text groups within same brand "
            f"({src_dupe_rows} affected brand_texts rows)"
        )
        _warn(f"{len(src_dupe_groups)} duplicate source-text groups ({src_dupe_rows} rows)")
    else:
        _ok("0 duplicate source-text groups")

    cur.execute("SELECT chunk_id, chunk_text, text_id FROM brand_chunks")
    chunk_groups = defaultdict(list)
    for chunk_id, chunk_text, text_id in cur.fetchall():
        chunk_groups[chunk_text].append((chunk_id, text_id))
    chunk_dupe_groups = {
        k: v for k, v in chunk_groups.items() if len({tid for _, tid in v}) > 1
    }
    chunk_dupe_rows = sum(len(v) for v in chunk_dupe_groups.values())
    if chunk_dupe_groups:
        warnings_out.append(
            f"{len(chunk_dupe_groups)} duplicate chunk_text groups across distinct text_id provenance "
            f"({chunk_dupe_rows} affected chunk rows)"
        )
        _warn(f"{len(chunk_dupe_groups)} duplicate chunk_text groups across distinct provenance ({chunk_dupe_rows} rows)")
    else:
        _ok("0 duplicate chunk_text groups across distinct provenance")

    conn.close()

    all_ok = len(failures) == 0

    print()
    print("=" * 70)
    if all_ok:
        print(f"  RESULT: PASS ({len(warnings_out)} warnings, 0 hard failures)")
    else:
        print(f"  RESULT: FAIL ({len(failures)} hard failures, {len(warnings_out)} warnings)")
        for f in failures:
            print(f"    - {f}")
    print("=" * 70)
    print()

    return all_ok, failures, warnings_out


def main():
    parser = argparse.ArgumentParser(description="Non-destructive brand_chunks handoff validator")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})")
    args = parser.parse_args()
    ok, _, _ = run_checks(args.db_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
