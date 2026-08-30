#!/usr/bin/env python3
"""
build_brand_chunks.py – Read brand_texts from SQLite, deterministically split
each text into sentence-aware chunks, and (re)populate the ``brand_chunks``
table.

Usage
-----
  python -m scripts.build_brand_chunks --db-path data/brand_data.db --dry-run
  python -m scripts.build_brand_chunks --db-path data/brand_data.db --replace

``brand_chunks`` is DERIVED data; ``brand_texts`` / ``brand_texts_raw`` are
SOURCE data and are never written to by this script.

Requires only stdlib + ``src.feature_extraction.feature_utils`` (deterministic,
no heavy-NLP / no model downloads).
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.feature_extraction.feature_utils import clean_text, sentence_split

logger = logging.getLogger(__name__)

# ── Default DB path (env-overridable, matches main.py) ───────────────────
DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/brand_data.db")

# ── Chunking constants ────────────────────────────────────────────────────
TARGET_MIN_CHARS = 80
TARGET_MAX_CHARS = 375
HARD_MAX_CHARS = 400
MAX_SENTENCES = 4
MIN_CHUNKS_PER_BRAND = 50

REQUIRED_PROVENANCE_FIELDS = ("text_id", "brand_id", "brand_name", "source_type")


class OversizedTokenError(Exception):
    """Raised when a single word/token exceeds HARD_MAX_CHARS."""


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class PackResult:
    chunks: list[dict]              # [{"text": str, "n_sentences": int}, ...]
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class CandidateResult:
    candidates: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    zero_chunk_text_ids: list[str] = field(default_factory=list)
    duplicate_warnings: list[dict] = field(default_factory=list)


# ── Sentence packing ────────────────────────────────────────────────────────

def split_long_sentence(
    sentence: str,
    *,
    target_max: int = TARGET_MAX_CHARS,
    hard_max: int = HARD_MAX_CHARS,
) -> list[str]:
    """
    Deterministically split an over-long sentence at word boundaries.

    Each returned piece is <= ``hard_max`` characters. Words are never
    split and never discarded. Raises :class:`OversizedTokenError` if a
    single word by itself exceeds ``hard_max``.
    """
    words = sentence.split(" ")
    for w in words:
        if len(w) > hard_max:
            raise OversizedTokenError(
                f"single token exceeds {hard_max} chars ({len(w)} chars): {w[:60]!r}..."
            )

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        add_len = len(w) if not current else len(w) + 1
        if current and current_len + add_len > target_max:
            pieces.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += add_len
    if current:
        pieces.append(" ".join(current))
    return pieces


def pack_text_into_chunks(
    text_id: str,
    cleaned_text: str,
    *,
    target_min: int = TARGET_MIN_CHARS,
    target_max: int = TARGET_MAX_CHARS,
    hard_max: int = HARD_MAX_CHARS,
    max_sentences: int = MAX_SENTENCES,
) -> PackResult:
    """
    Deterministic sentence-aware chunking for a single source text.

    Sentences are packed greedily in original order; adjacent sentences
    from the SAME source are combined while staying within the char/
    sentence limits. Oversized single sentences are split at word
    boundaries. A short final remainder is merged into the previous
    chunk when that stays within hard limits; otherwise it is kept and
    reported as a warning.
    """
    sentences = sentence_split(cleaned_text)
    if not sentences:
        return PackResult(chunks=[])

    chunks: list[dict] = []
    buffer_sentences: list[str] = []

    def flush() -> None:
        nonlocal buffer_sentences
        if buffer_sentences:
            chunks.append({
                "text": " ".join(buffer_sentences),
                "n_sentences": len(buffer_sentences),
            })
            buffer_sentences = []

    warnings: list[str] = []

    for sentence in sentences:
        if len(sentence) > hard_max:
            flush()
            try:
                pieces = split_long_sentence(
                    sentence, target_max=target_max, hard_max=hard_max
                )
            except OversizedTokenError as exc:
                return PackResult(chunks=chunks, warnings=warnings, error=str(exc))
            for piece in pieces:
                chunks.append({"text": piece, "n_sentences": 1})
            continue

        prospective = buffer_sentences + [sentence]
        prospective_text = " ".join(prospective)
        if buffer_sentences and (
            len(prospective_text) > target_max or len(prospective) > max_sentences
        ):
            flush()
            buffer_sentences = [sentence]
        else:
            buffer_sentences = prospective

    flush()

    # Short final-remainder handling
    if len(chunks) >= 2 and len(chunks[-1]["text"]) < target_min:
        prev, last = chunks[-2], chunks[-1]
        merged_text = prev["text"] + " " + last["text"]
        merged_n = prev["n_sentences"] + last["n_sentences"]
        if len(merged_text) <= hard_max and merged_n <= max_sentences:
            chunks[-2] = {"text": merged_text, "n_sentences": merged_n}
            chunks.pop()
        else:
            warnings.append(
                f"{text_id}: short final chunk ({len(last['text'])} chars) "
                "could not be merged with previous chunk; preserved as-is"
            )
    elif len(chunks) == 1 and len(chunks[0]["text"]) < target_min:
        warnings.append(
            f"{text_id}: entire source text is short "
            f"({len(chunks[0]['text'])} chars); preserved as single chunk"
        )

    return PackResult(chunks=chunks, warnings=warnings)


# ── Candidate generation ─────────────────────────────────────────────────

def build_candidates(
    rows: list[dict],
    *,
    target_min: int = TARGET_MIN_CHARS,
    target_max: int = TARGET_MAX_CHARS,
    hard_max: int = HARD_MAX_CHARS,
    max_sentences: int = MAX_SENTENCES,
) -> CandidateResult:
    """
    Build the full in-memory candidate chunk set from ``brand_texts`` rows.

    Pure function — does not touch any database.
    """
    result = CandidateResult()
    seen_chunk_ids: set[str] = set()
    duplicate_text_map: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        text_id = row.get("text_id")
        missing = [f for f in REQUIRED_PROVENANCE_FIELDS if not row.get(f)]
        if missing:
            result.errors.append({
                "text_id": text_id,
                "issue": f"missing required provenance fields: {missing}",
            })
            continue

        brand_id = row["brand_id"]
        brand_name = row["brand_name"]
        source_type = row["source_type"]

        cleaned = clean_text(row.get("text"))
        if not cleaned:
            # Genuinely blank source text -> legitimately zero chunks.
            continue

        pack = pack_text_into_chunks(
            text_id,
            cleaned,
            target_min=target_min,
            target_max=target_max,
            hard_max=hard_max,
            max_sentences=max_sentences,
        )
        result.warnings.extend(pack.warnings)

        if pack.error:
            result.errors.append({"text_id": text_id, "issue": pack.error})
            result.zero_chunk_text_ids.append(text_id)
            continue

        if not pack.chunks:
            result.zero_chunk_text_ids.append(text_id)
            continue

        for idx, c in enumerate(pack.chunks):
            chunk_id = f"{text_id}__chunk_{idx:03d}"
            if chunk_id in seen_chunk_ids:
                result.errors.append({
                    "text_id": text_id,
                    "issue": f"chunk_id collision: {chunk_id}",
                })
                continue
            seen_chunk_ids.add(chunk_id)

            chunk_text = c["text"]
            candidate = {
                "chunk_id": chunk_id,
                "text_id": text_id,
                "brand_id": brand_id,
                "brand_name": brand_name,
                "source_type": source_type,
                "chunk_text": chunk_text,
                "char_count": len(chunk_text),
            }
            result.candidates.append(candidate)
            duplicate_text_map[chunk_text].add(text_id)

    result.duplicate_warnings = [
        {
            "chunk_text_preview": t[:80] + ("…" if len(t) > 80 else ""),
            "text_ids": sorted(ids),
        }
        for t, ids in duplicate_text_map.items()
        if len(ids) > 1
    ]

    return result


def validate_candidates(
    candidate_result: CandidateResult,
    rows: list[dict],
    *,
    min_chunks_per_brand: int = MIN_CHUNKS_PER_BRAND,
    hard_max: int = HARD_MAX_CHARS,
) -> list[str]:
    """
    Validate the candidate set against hard constraints. Returns a list of
    failure messages (empty list == all hard checks passed).
    """
    failures: list[str] = []

    for e in candidate_result.errors:
        failures.append(f"text_id={e['text_id']}: {e['issue']}")

    brands_in_source = sorted({r["brand_id"] for r in rows if r.get("brand_id")})
    counts = Counter(c["brand_id"] for c in candidate_result.candidates)
    for b in brands_in_source:
        n = counts.get(b, 0)
        if n < min_chunks_per_brand:
            failures.append(
                f"brand '{b}' has {n} candidate chunks (< required {min_chunks_per_brand})"
            )

    for c in candidate_result.candidates:
        if c["char_count"] > hard_max:
            failures.append(
                f"chunk {c['chunk_id']} exceeds hard max: {c['char_count']} chars"
            )
        if not c["chunk_text"] or not c["chunk_text"].strip():
            failures.append(f"chunk {c['chunk_id']} is blank")
        for f_ in ("chunk_id", "text_id", "brand_id", "brand_name", "source_type"):
            if not c.get(f_):
                failures.append(f"chunk {c.get('chunk_id')} missing field '{f_}'")

    ids = [c["chunk_id"] for c in candidate_result.candidates]
    if len(ids) != len(set(ids)):
        failures.append("duplicate chunk_id detected in candidate set")

    source_text_by_id = {r["text_id"]: r.get("text") for r in rows}
    for text_id in candidate_result.zero_chunk_text_ids:
        raw = source_text_by_id.get(text_id)
        if raw and clean_text(raw):
            failures.append(
                f"text_id={text_id} produced zero chunks despite non-blank source content"
            )

    return failures


# ── DB access ─────────────────────────────────────────────────────────────

def load_source_rows(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT text_id, brand_id, brand_name, source_type, text FROM brand_texts"
    )
    cols = ["text_id", "brand_id", "brand_name", "source_type", "text"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def replace_chunks(conn: sqlite3.Connection, candidates: list[dict]) -> None:
    """
    Transactionally replace the contents of ``brand_chunks``.

    On any exception the transaction is rolled back and the existing
    table contents are left untouched.
    """
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM brand_chunks")
        cur.executemany(
            """
            INSERT INTO brand_chunks
                (chunk_id, text_id, brand_id, brand_name, source_type, chunk_text, char_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c["chunk_id"],
                    c["text_id"],
                    c["brand_id"],
                    c["brand_name"],
                    c["source_type"],
                    c["chunk_text"],
                    c["char_count"],
                )
                for c in candidates
            ],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── Reporting ─────────────────────────────────────────────────────────────

def build_report(
    rows: list[dict],
    candidate_result: CandidateResult,
    failures: list[str],
    *,
    min_chunks_per_brand: int = MIN_CHUNKS_PER_BRAND,
    hard_max: int = HARD_MAX_CHARS,
    target_min: int = TARGET_MIN_CHARS,
) -> dict:
    per_brand_source = Counter(r["brand_id"] for r in rows if r.get("brand_id"))
    per_brand_candidates = Counter(c["brand_id"] for c in candidate_result.candidates)
    lengths = [c["char_count"] for c in candidate_result.candidates]

    report = {
        "source_rows": len(rows),
        "source_brands": sorted(per_brand_source.keys()),
        "source_per_brand": dict(sorted(per_brand_source.items())),
        "total_candidates": len(candidate_result.candidates),
        "candidates_per_brand": dict(sorted(per_brand_candidates.items())),
        "lowest_per_brand": min(per_brand_candidates.values()) if per_brand_candidates else 0,
        "highest_per_brand": max(per_brand_candidates.values()) if per_brand_candidates else 0,
        "min_len": min(lengths) if lengths else None,
        "median_len": statistics.median(lengths) if lengths else None,
        "mean_len": statistics.fmean(lengths) if lengths else None,
        "max_len": max(lengths) if lengths else None,
        "count_under_min": sum(1 for l in lengths if l < target_min),
        "count_over_hard_max": sum(1 for l in lengths if l > hard_max),
        "zero_chunk_text_ids": candidate_result.zero_chunk_text_ids,
        "errors": candidate_result.errors,
        "warnings": candidate_result.warnings,
        "duplicate_warnings": candidate_result.duplicate_warnings,
        "failures": failures,
        "pass": len(failures) == 0,
    }
    return report


def print_report(report: dict, *, db_path: str, mode: str) -> None:
    print()
    print("=" * 70)
    print("  Brand Chunk Builder — Report")
    print("=" * 70)
    print(f"  DB path : {db_path}")
    print(f"  Mode    : {mode}")
    print()
    print("SOURCE")
    print(f"  brand_texts rows     : {report['source_rows']}")
    print(f"  brands                : {len(report['source_brands'])}")
    for b, n in report["source_per_brand"].items():
        print(f"    {b:16s} {n}")
    print()
    print("CANDIDATE CHUNKS")
    print(f"  total candidates      : {report['total_candidates']}")
    for b, n in report["candidates_per_brand"].items():
        flag = "" if n >= MIN_CHUNKS_PER_BRAND else "  <== BELOW MINIMUM"
        print(f"    {b:16s} {n}{flag}")
    print(f"  lowest per brand      : {report['lowest_per_brand']}")
    print(f"  highest per brand     : {report['highest_per_brand']}")
    print()
    print("LENGTHS")
    print(f"  min char_count        : {report['min_len']}")
    print(f"  median char_count     : {report['median_len']}")
    print(f"  mean char_count       : "
          f"{report['mean_len']:.1f}" if report["mean_len"] is not None else "  mean char_count       : n/a")
    print(f"  max char_count        : {report['max_len']}")
    print(f"  count < {TARGET_MIN_CHARS} chars     : {report['count_under_min']}")
    print(f"  count > {HARD_MAX_CHARS} chars     : {report['count_over_hard_max']}")
    print()
    print("TRACEABILITY")
    print(f"  source texts with zero chunks : {len(report['zero_chunk_text_ids'])}")
    if report["zero_chunk_text_ids"]:
        for tid in report["zero_chunk_text_ids"]:
            print(f"    - {tid}")
    print(f"  errors                         : {len(report['errors'])}")
    for e in report["errors"]:
        print(f"    - text_id={e['text_id']}: {e['issue']}")
    print()
    print("DUPLICATE WARNINGS (same chunk text, different source text_id)")
    print(f"  count: {len(report['duplicate_warnings'])}")
    for d in report["duplicate_warnings"][:20]:
        print(f"    - text_ids={d['text_ids']}  preview={d['chunk_text_preview']!r}")
    if len(report["duplicate_warnings"]) > 20:
        print(f"    ... and {len(report['duplicate_warnings']) - 20} more")
    print()
    print("SHORT-CHUNK / OTHER WARNINGS")
    print(f"  count: {len(report['warnings'])}")
    for w in report["warnings"][:20]:
        print(f"    - {w}")
    if len(report["warnings"]) > 20:
        print(f"    ... and {len(report['warnings']) - 20} more")
    print()
    print("=" * 70)
    if report["pass"]:
        print("  RESULT: PASS — every brand >= "
              f"{MIN_CHUNKS_PER_BRAND} candidate chunks, no hard-constraint failures")
    else:
        print("  RESULT: FAIL — hard constraint violations:")
        for f_ in report["failures"]:
            print(f"    - {f_}")
    print("=" * 70)
    print()


# ── Orchestration ─────────────────────────────────────────────────────────

def run(
    db_path: str,
    *,
    mode: str,
    min_chunks_per_brand: int = MIN_CHUNKS_PER_BRAND,
) -> dict:
    """
    Full pipeline: load source rows -> build candidates -> validate ->
    (dry-run: report only) or (replace: transactionally write iff valid).

    Returns the report dict. Never writes to brand_chunks unless
    ``mode == "replace"`` AND validation passes.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='brand_texts'"
        )
        if not cur.fetchone():
            raise RuntimeError(f"Table 'brand_texts' does not exist in {db_path}")
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='brand_chunks'"
        )
        if not cur.fetchone():
            raise RuntimeError(
                f"Table 'brand_chunks' does not exist in {db_path}. "
                "This script populates an existing table; it does not create one."
            )

        rows = load_source_rows(conn)
        candidate_result = build_candidates(rows)
        failures = validate_candidates(
            candidate_result, rows, min_chunks_per_brand=min_chunks_per_brand
        )
        report = build_report(rows, candidate_result, failures)

        if mode == "dry-run":
            return report

        if mode == "replace":
            if not report["pass"]:
                logger.error("Validation failed — brand_chunks was NOT modified.")
                return report
            replace_chunks(conn, candidate_result.candidates)
            report["written"] = True
            return report

        raise ValueError(f"Unknown mode: {mode}")
    finally:
        conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build brand_chunks from brand_texts (deterministic, sentence-aware).",
    )
    p.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH}). "
             "Must be passed explicitly to avoid ambiguity.",
        required=False,
    )
    mode_group = p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute candidate chunks and report; do not write to the DB.",
    )
    mode_group.add_argument(
        "--replace",
        action="store_true",
        help="Transactionally replace brand_chunks with the validated candidate set.",
    )
    p.add_argument(
        "--min-chunks-per-brand",
        type=int,
        default=MIN_CHUNKS_PER_BRAND,
        help=f"Minimum required chunks per brand (default: {MIN_CHUNKS_PER_BRAND}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = parse_args(argv)
    mode = "dry-run" if args.dry_run else "replace"

    logger.info("Running brand chunk builder against %s (mode=%s) …", args.db_path, mode)

    report = run(
        args.db_path,
        mode=mode,
        min_chunks_per_brand=args.min_chunks_per_brand,
    )
    print_report(report, db_path=args.db_path, mode=mode)

    if not report["pass"]:
        sys.exit(1)
    if mode == "replace" and not report.get("written"):
        sys.exit(1)


if __name__ == "__main__":
    main()
