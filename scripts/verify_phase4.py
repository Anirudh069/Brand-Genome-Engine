#!/usr/bin/env python3
"""
verify_phase4.py - Authoritative, non-destructive Phase 4 Definition-of-Done
verifier for the final (Stage 9) Brand Genome Engine.

Does NOT require a running backend server, does NOT call OpenAI, and does
NOT mutate ``data/brand_data.db``. A fresh checked-in DB with no user genome
initialised yet is expected to PASS every check here (user-genome-dependent
checks are reported as INFO, never FAIL).

Complements (does not duplicate) the deeper row-level checks already done by:
  * scripts/validate_db.py
  * scripts/validate_chunk_handoff.py

Usage:
  python -m scripts.verify_phase4
  python -m scripts.verify_phase4 --db-path data/brand_data.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/brand_data.db")
DEFAULT_ANALYTICS_CACHE = os.getenv("ANALYTICS_CACHE_PATH", "data/processed/analytics_cache.json")
DEFAULT_RAG_DIR = os.getenv("RAG_INDEX_DIR", "data/processed/rag")

REQUIRED_TABLES = {
    "brands", "brand_texts_raw", "brand_texts", "brand_chunks",
    "brand_profiles", "brand_profile", "analysis_history",
}
MIN_TEXTS_PER_BRAND = 30
MIN_CHUNKS_PER_BRAND = 50
CANONICAL_ROUTES = {
    "/api/genome/init", "/api/genome",
    "/api/consistency/score",
    "/api/benchmark/brands", "/api/benchmark/run",
    "/api/analytics",
    "/api/rag/retrieve",
    "/api/rewrite",
    "/api/rebuild/profile", "/api/rebuild/chunks", "/api/rebuild/index",
}
# Values that must never appear as *active* fabricated numbers/paths again.
LEGACY_MARKERS = ["n_brands: 5", "n_texts: 87", "150 rows", "15 texts", "5 brands"]

_failures: list[str] = []
_warnings: list[str] = []


def _fail(msg: str) -> None:
    _failures.append(msg)
    print(f"  x FAIL: {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    _warnings.append(msg)
    print(f"  ! WARN: {msg}")


def _ok(msg: str) -> None:
    print(f"  v PASS: {msg}")


def _info(msg: str) -> None:
    print(f"  i INFO: {msg}")


def check_database(db_path: str) -> None:
    print("\n[1] Canonical database")
    if not Path(db_path).exists():
        _fail(f"Database file does not exist: {db_path}")
        return
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    integrity = cur.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity == "ok":
        _ok("PRAGMA integrity_check = ok")
    else:
        _fail(f"PRAGMA integrity_check = {integrity}")

    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = REQUIRED_TABLES - tables
    if missing:
        _fail(f"Missing required tables: {sorted(missing)}")
    else:
        _ok(f"All {len(REQUIRED_TABLES)} required tables present")

    counts = {}
    for t in ("brand_texts", "brand_texts_raw", "brand_chunks", "brand_profiles"):
        counts[t] = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    if counts["brand_texts"] == 450 and counts["brand_texts_raw"] == 450:
        _ok(f"brand_texts = {counts['brand_texts']}, brand_texts_raw = {counts['brand_texts_raw']}")
    else:
        _fail(f"Expected 450/450 competitor texts, found brand_texts={counts['brand_texts']}, "
              f"brand_texts_raw={counts['brand_texts_raw']}")
    if counts["brand_chunks"] == 657:
        _ok(f"brand_chunks = {counts['brand_chunks']}")
    else:
        _warn(f"Expected 657 competitor chunks (baseline), found {counts['brand_chunks']}")
    if counts["brand_profiles"] == 10:
        _ok(f"brand_profiles (competitor profiles) = {counts['brand_profiles']}")
    else:
        _fail(f"Expected 10 competitor profiles, found {counts['brand_profiles']}")

    per_brand_texts = dict(cur.execute(
        "SELECT brand_id, COUNT(*) FROM brand_texts GROUP BY brand_id"
    ).fetchall())
    low_texts = {b: n for b, n in per_brand_texts.items() if n < MIN_TEXTS_PER_BRAND}
    if low_texts:
        _fail(f"Brands below {MIN_TEXTS_PER_BRAND} texts: {low_texts}")
    else:
        _ok(f"All {len(per_brand_texts)} competitor brands >= {MIN_TEXTS_PER_BRAND} texts")

    per_brand_chunks = dict(cur.execute(
        "SELECT brand_id, COUNT(*) FROM brand_chunks WHERE brand_id != 'user_brand' "
        "GROUP BY brand_id"
    ).fetchall())
    low_chunks = {b: n for b, n in per_brand_chunks.items() if n < MIN_CHUNKS_PER_BRAND}
    if low_chunks:
        _fail(f"Competitor brands below {MIN_CHUNKS_PER_BRAND} chunks: {low_chunks}")
    else:
        _ok(f"All {len(per_brand_chunks)} competitor brands >= {MIN_CHUNKS_PER_BRAND} chunks")

    blank = cur.execute(
        "SELECT COUNT(*) FROM brand_chunks WHERE chunk_text IS NULL OR TRIM(chunk_text) = ''"
    ).fetchone()[0]
    orphans = cur.execute(
        "SELECT COUNT(*) FROM brand_chunks c LEFT JOIN brand_texts t ON c.text_id = t.text_id "
        "WHERE t.text_id IS NULL"
    ).fetchone()[0]
    dup_ids = cur.execute(
        "SELECT COUNT(*) FROM (SELECT chunk_id FROM brand_chunks GROUP BY chunk_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    over_max = cur.execute("SELECT COUNT(*) FROM brand_chunks WHERE LENGTH(chunk_text) > 400").fetchone()[0]
    if blank == 0 and orphans == 0 and dup_ids == 0 and over_max == 0:
        _ok("No blank chunks, no orphan text_id, unique chunk_id, all chunks <= 400 chars")
    else:
        _fail(f"Chunk integrity issue: blank={blank} orphans={orphans} dup_ids={dup_ids} over_max={over_max}")

    user_profile_rows = cur.execute(
        "SELECT COUNT(*) FROM brand_profile WHERE brand_id = 0"
    ).fetchone()[0]
    if user_profile_rows:
        _info("User genome (brand_profile, brand_id=0) is initialised in this DB.")
    else:
        _info("User genome not yet initialised (brand_profile empty) - expected for a clean checked-in DB.")

    history_rows = cur.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
    _info(f"analysis_history rows = {history_rows} "
          f"({'expected 0 on clean baseline' if history_rows == 0 else 'runtime activity present'})")

    conn.close()


def check_no_stray_db_files() -> None:
    print("\n[2] Single canonical .db file")
    repo_root = Path(__file__).resolve().parent.parent
    dbs = [p for p in repo_root.rglob("*.db")
           if ".venv" not in p.parts and "node_modules" not in p.parts]
    canonical = (repo_root / "data" / "brand_data.db").resolve()
    stray = [p for p in dbs if p.resolve() != canonical]
    if stray:
        _fail(f"Stray .db files found: {[str(p) for p in stray]}")
    else:
        _ok(f"Exactly one physical .db file in repo: {canonical}")


def check_analytics_artifact(db_path: str, cache_path: str) -> None:
    print("\n[3] Analytics artifact readiness")
    try:
        from src.analytics.cache import get_cache_state
    except ImportError as e:
        _fail(f"Cannot import src.analytics.cache: {e}")
        return
    if not Path(cache_path).exists():
        _info(f"Analytics cache not yet built at {cache_path} "
              f"(build with: python -m scripts.build_analytics_cache)")
        return
    try:
        state = get_cache_state(db_path, cache_path)
    except Exception as e:
        _fail(f"get_cache_state raised: {e}")
        return
    if state.get("state") == "valid":
        _ok("Analytics cache exists and its fingerprint matches the current DB")
    elif state.get("state") == "stale":
        _warn("Analytics cache exists but is stale relative to the current DB "
              "(rebuild with scripts/build_analytics_cache.py)")
    else:
        _info("Analytics cache fingerprint could not be matched (missing cache or empty DB corpus)")


def check_rag_artifact(db_path: str, rag_dir: str) -> None:
    print("\n[4] RAG (chunk-level FAISS) artifact readiness")
    try:
        from src.retrieval.rag_builder import current_db_fingerprint, load_manifest
    except ImportError as e:
        _fail(f"Cannot import src.retrieval.rag_builder: {e}")
        return
    manifest_path = Path(rag_dir) / "manifest.json"
    if not manifest_path.exists():
        _info(f"RAG index not yet built at {rag_dir} "
              f"(build with: python -m scripts.build_rag_index)")
        return
    try:
        manifest = load_manifest(rag_dir)
        live_fp = current_db_fingerprint(db_path)
        if manifest.get("fingerprint") == live_fp:
            _ok(f"RAG manifest fingerprint matches current DB ({len(manifest.get('brands', {}))} brand indexes)")
        else:
            _warn("RAG manifest fingerprint is stale relative to the current DB "
                  "(rebuild with scripts/build_rag_index.py or POST /api/rebuild/index)")
    except Exception as e:
        _fail(f"Failed reading RAG manifest: {e}")


def check_api_routes() -> None:
    print("\n[5] FastAPI canonical route registration")
    try:
        from src.api.main import app
    except Exception as e:
        _fail(f"Failed to import src.api.main:app: {e}")
        return
    paths: dict[str, int] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        paths[path] = paths.get(path, 0) + 1
    missing = CANONICAL_ROUTES - set(paths)
    if missing:
        _fail(f"Missing canonical routes: {sorted(missing)}")
    else:
        _ok(f"All {len(CANONICAL_ROUTES)} canonical routes are registered")
    # A route registered twice under the SAME path with the SAME methods would
    # be a real conflict; FastAPI allows the same path with different aliases
    # (e.g. GET vs POST) so only exact duplicate (path, methods) pairs count.
    seen: dict[tuple, bool] = {}
    dupes = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = frozenset(getattr(route, "methods", None) or [])
        if path is None:
            continue
        key = (path, methods)
        if key in seen:
            dupes.append(key)
        seen[key] = True
    if dupes:
        _fail(f"Duplicate route registrations (path, methods): {dupes}")
    else:
        _ok("No duplicate (path, method) route registrations")


def check_no_legacy_fake_markers() -> None:
    print("\n[6] No fabricated legacy values in active backend source")
    repo_root = Path(__file__).resolve().parent.parent
    this_file = Path(__file__).resolve()
    active_dirs = [repo_root / "src", repo_root / "scripts"]
    hits = []
    for base in active_dirs:
        for py_file in base.rglob("*.py"):
            if py_file.resolve() == this_file:
                continue  # this file legitimately lists the marker strings
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for marker in LEGACY_MARKERS:
                if marker in text:
                    hits.append(f"{py_file.relative_to(repo_root)}: {marker!r}")
    if hits:
        _fail(f"Legacy fabricated-value markers found in active source: {hits}")
    else:
        _ok("No legacy fabricated-value markers found in src/ or scripts/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--analytics-cache", default=DEFAULT_ANALYTICS_CACHE)
    parser.add_argument("--rag-dir", default=DEFAULT_RAG_DIR)
    args = parser.parse_args()

    print("=" * 64)
    print("  Brand Genome Engine - Phase 4 Final Verifier (Stage 9)")
    print("=" * 64)

    check_database(args.db_path)
    check_no_stray_db_files()
    check_analytics_artifact(args.db_path, args.analytics_cache)
    check_rag_artifact(args.db_path, args.rag_dir)
    check_api_routes()
    check_no_legacy_fake_markers()

    print("\n" + "=" * 64)
    if _failures:
        print(f"  RESULT: FAIL ({len(_failures)} failures, {len(_warnings)} warnings)")
        print("=" * 64)
        return 1
    print(f"  RESULT: PASS ({len(_warnings)} warnings, 0 hard failures)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
