#!/usr/bin/env python3
"""
validate_db.py - Post-build validation checks for the Brand Genome Engine DB.

Runs five checks (A-E) and exits non-zero if any fail.

Usage:
  python -m scripts.validate_db
  python -m scripts.validate_db --db-path data/brand_data.db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/brand_data.db")

REQUIRED_TABLES = {"brand_texts_raw", "brand_texts", "brand_chunks", "brand_profiles"}
EXPECTED_COUNTS = {
    "brand_texts_raw": 150,
    "brand_texts": 150,
    "brand_chunks": 150,
}
REQUIRED_PROFILE_KEYS = {"top_keywords", "mean_sentiment", "mean_flesch"}


def _fail(msg):
    print(f"  x FAIL: {msg}", file=sys.stderr)


def _ok(msg):
    print(f"  v PASS: {msg}")


def run_checks(db_path):
    all_ok = True
    print()
    print("=" * 64)
    print(f"  DB Validation - {db_path}")
    print("=" * 64)
    print()

    if not Path(db_path).exists():
        _fail(f"Database file does not exist: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check A: Required tables
    print("[A] Required tables")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    missing = REQUIRED_TABLES - tables
    if missing:
        _fail(f"Missing tables: {sorted(missing)}")
        all_ok = False
    else:
        _ok(f"All required tables present: {sorted(REQUIRED_TABLES)}")

    # Check B: Row counts
    print()
    print("[B] Row counts")
    for table, expected in EXPECTED_COUNTS.items():
        if table not in tables:
            _fail(f"{table} - table missing")
            all_ok = False
            continue
        cur.execute(f"SELECT COUNT(*) FROM [{table}]")
        actual = cur.fetchone()[0]
        if actual != expected:
            _fail(f"{table}: expected {expected}, got {actual}")
            all_ok = False
        else:
            _ok(f"{table} = {actual}")

    if "brand_profiles" in tables:
        cur.execute("SELECT COUNT(*) FROM brand_profiles")
        profile_count = cur.fetchone()[0]
        if profile_count == 0:
            _fail("brand_profiles has 0 rows (expected > 0)")
            all_ok = False
        else:
            _ok(f"brand_profiles = {profile_count}")
    else:
        _fail("brand_profiles table missing")
        all_ok = False

    # Check C: Profile rows sanity
    print()
    print("[C] Profile rows (first 5)")
    if "brand_profiles" in tables:
        cur.execute(
            "SELECT brand_id, brand_name, n_texts, version, built_at "
            "FROM brand_profiles ORDER BY brand_id LIMIT 5"
        )
        rows = cur.fetchall()
        if not rows:
            _fail("No profile rows found")
            all_ok = False
        else:
            for r in rows:
                bid, bname, ntxt, ver, bat = r
                print(f"      {bid:18s}  name={bname:14s}  n={ntxt}  ver={ver}  at={bat}")
            _ok(f"Displayed {len(rows)} profile rows")

    # Check D: profile_json sanity
    print()
    print("[D] profile_json sanity")
    if "brand_profiles" in tables:
        cur.execute("SELECT brand_id, profile_json FROM brand_profiles")
        all_profiles = cur.fetchall()
        d_ok = True
        for brand_id, pjson in all_profiles:
            try:
                profile = json.loads(pjson)
            except (json.JSONDecodeError, TypeError) as exc:
                _fail(f"{brand_id}: invalid JSON - {exc}")
                d_ok = False
                all_ok = False
                continue

            missing_keys = REQUIRED_PROFILE_KEYS - set(profile.keys())
            if missing_keys:
                _fail(f"{brand_id}: missing keys {sorted(missing_keys)}")
                d_ok = False
                all_ok = False
                continue

            emb = profile.get("mean_embedding", [])
            emb_status = profile.get("embedding_status", "")
            if isinstance(emb, list) and len(emb) == 384:
                pass
            elif emb_status == "missing":
                pass
            else:
                emb_len = len(emb) if isinstance(emb, list) else "?"
                _fail(f"{brand_id}: embedding len={emb_len}, status='{emb_status}'")
                d_ok = False
                all_ok = False

        if d_ok:
            _ok(f"All {len(all_profiles)} profiles: valid JSON, required keys, embedding OK")

    # Check E: No stray .db files
    print()
    print("[E] Stray .db file check")
    canonical = Path(db_path).resolve()
    repo_root = canonical.parent.parent
    if canonical.parent.name != "data":
        repo_root = canonical.parent

    stray = []
    for db_file in repo_root.rglob("*.db"):
        if db_file.resolve() == canonical:
            continue
        parts = db_file.parts
        if any(p in (".git", "node_modules", "venv", ".venv", "__pycache__") for p in parts):
            continue
        stray.append(str(db_file))

    if stray:
        _fail(f"Found stray .db files: {stray}")
        all_ok = False
    else:
        _ok("No stray .db files found outside canonical path")

    conn.close()

    print()
    print("=" * 64)
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED", file=sys.stderr)
    print("=" * 64)
    print()

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Validate Brand Genome Engine DB")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    ok = run_checks(args.db_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
