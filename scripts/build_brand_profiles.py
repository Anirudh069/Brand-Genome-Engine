#!/usr/bin/env python3
"""
build_brand_profiles.py – Read brand_texts from SQLite, compute per-brand
statistical profiles, and upsert them into the ``brand_profiles`` table.

Usage
-----
  python -m scripts.build_brand_profiles
  python -m scripts.build_brand_profiles --db-path data/brand_data.db --version v2
  python -m scripts.build_brand_profiles --dry-run --limit-brands rolex,omega

Requires
--------
Only stdlib + the lightweight extractors already in ``src.feature_extraction``.
Embedding computation is **optional** — if it fails the profile stores
``embedding_status: "missing"`` instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Default DB path (env-overridable, matches main.py) ───────────────────
DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/brand_data.db")

# ── Lightweight text helpers (no heavy deps) ──────────────────────────────
_WORD_RE = re.compile(r"[a-zA-Z']+")

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "as", "at", "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those", "by", "from", "you", "we",
    "they", "he", "she", "i", "our", "your", "their", "its", "not", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "can", "all", "been", "more", "also", "than", "into", "which",
    "about", "so", "if", "when", "what", "there", "each", "just", "most",
    "other", "some", "such", "only", "over", "new", "very", "after",
    "before", "between",
})


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _content_words(text: str) -> list[str]:
    return [w for w in _tokenize(text) if w not in _STOPWORDS and len(w) >= 3]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ── Feature proxies (lightweight, no external deps) ──────────────────────
# We try to import the real extractors first; fall back to simple proxies.

def _make_sentiment_fn():
    """Return best available sentiment function → float in [0, 1]."""
    try:
        from src.feature_extraction.sentiment_extractor import extract_sentiment
        return extract_sentiment
    except ImportError:
        pass

    # Proxy: tiny lexicon approach
    _pos = frozenset({
        "excellence", "exceptional", "extraordinary", "remarkable", "enduring",
        "precision", "innovative", "iconic", "perfect", "ultimate", "heritage",
        "trust", "superior", "finest", "premium", "outstanding", "passion",
        "inspire", "legendary", "timeless", "craftsmanship", "champion",
        "victory", "best", "beautiful", "elegant", "luxury", "prestigious",
    })
    _neg = frozenset({
        "fail", "failure", "poor", "cheap", "bad", "terrible", "awful",
        "inferior", "weak", "broken", "wrong", "defect", "problem",
    })

    def _proxy(text: str) -> float:
        words = set(_tokenize(text))
        p, n = len(words & _pos), len(words & _neg)
        total = p + n
        if total == 0:
            return 0.55
        return round((p / total), 4)

    return _proxy


def _make_readability_fn():
    """Return best available Flesch Reading Ease function → float."""
    try:
        from src.feature_extraction.readability_extractor import flesch_reading_ease
        return flesch_reading_ease
    except ImportError:
        pass

    def _proxy(text: str) -> float:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        words = _tokenize(text)
        if not sentences or not words:
            return 50.0

        def _syllables(w: str) -> int:
            return max(1, len(re.findall(r"[aeiou]+", w.lower())))

        total_syl = sum(_syllables(w) for w in words)
        asl = len(words) / len(sentences)
        asw = total_syl / len(words)
        return 206.835 - 1.015 * asl - 84.6 * asw

    return _proxy


def _make_formality_fn():
    """Return best available formality function → float in [0, 1]."""
    try:
        from src.feature_extraction.formality_extractor import extract_formality
        return extract_formality
    except ImportError:
        pass

    def _proxy(text: str) -> float:
        content = _content_words(text)
        if not content:
            return 0.5
        long_words = sum(1 for w in content if len(w) >= 7)
        return min(1.0, long_words / len(content))

    return _proxy


def _vocab_richness(text: str) -> float:
    words = _tokenize(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _top_keywords(texts: list[str], k: int = 10) -> list[str]:
    counts: Counter[str] = Counter()
    for t in texts:
        for w in _content_words(t):
            counts[w] += 1
    return [w for w, _ in counts.most_common(k)]


def _tone_label(formality: float, sentiment: float) -> str:
    if formality >= 0.55 and sentiment >= 0.6:
        return "authoritative"
    if formality >= 0.55:
        return "formal"
    if sentiment >= 0.6:
        return "motivational"
    return "neutral"


def _try_compute_mean_embedding(texts: list[str]) -> tuple[list[float], str]:
    """
    Attempt to compute a mean embedding for the brand.

    Returns
    -------
    (embedding_list, status)
        embedding_list: 384-d list[float] or empty list
        status: "ok" | "missing"
    """
    try:
        import numpy as np
        from src.feature_extraction.embedding_extractor import get_embedding
        vecs = []
        for t in texts:
            emb, _ = get_embedding(t)
            if emb is not None and any(v != 0.0 for v in emb):
                vecs.append(emb)
        if vecs:
            mean_vec = list(np.mean(vecs, axis=0).astype(float))
            return mean_vec, "ok"
    except Exception as exc:
        logger.warning("Embedding computation skipped: %s", exc)
    return [], "missing"


# ── Core builder ──────────────────────────────────────────────────────────

def build_profiles(
    db_path: str,
    version: str = "v1",
    min_texts: int = 10,
    limit_brands: list[str] | None = None,
    dry_run: bool = False,
    compute_embeddings: bool = True,
) -> dict:
    """
    Build brand profiles from ``brand_texts`` and upsert into ``brand_profiles``.

    Returns a summary dict with counts for processed, skipped, errors.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Ensure the canonical DB exists at data/brand_data.db "
            "(or set SQLITE_DB_PATH)."
        )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Verify brand_texts table exists
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='brand_texts'"
    )
    if not cur.fetchone():
        conn.close()
        raise RuntimeError(
            f"Table 'brand_texts' does not exist in {db_path}. "
            "Run the data ingestion pipeline first."
        )

    # Ensure brand_profiles table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS brand_profiles (
            brand_id     TEXT PRIMARY KEY,
            brand_name   TEXT NOT NULL,
            n_texts      INTEGER NOT NULL DEFAULT 0,
            version      TEXT NOT NULL,
            built_at     TEXT NOT NULL DEFAULT (datetime('now')),
            profile_json TEXT NOT NULL
        )
    """)
    conn.commit()

    # Load all texts
    cur.execute("SELECT brand_id, brand_name, text FROM brand_texts")
    rows = cur.fetchall()

    # Group by brand
    by_brand: dict[tuple[str, str], list[str]] = {}
    for brand_id, brand_name, text in rows:
        by_brand.setdefault((brand_id, brand_name), []).append(text or "")

    logger.info("Found %d brands with %d total texts.", len(by_brand), len(rows))

    # Init extractors
    sentiment_fn = _make_sentiment_fn()
    readability_fn = _make_readability_fn()
    formality_fn = _make_formality_fn()

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = {"processed": 0, "skipped": 0, "errors": 0, "brands": []}

    for (brand_id, brand_name), texts in sorted(by_brand.items()):
        # Filter brands if --limit-brands was set
        if limit_brands and brand_id not in limit_brands:
            logger.debug("Skipping %s (not in --limit-brands).", brand_id)
            summary["skipped"] += 1
            continue

        if len(texts) < min_texts:
            logger.warning(
                "Skipping %s: only %d texts (min=%d).",
                brand_id, len(texts), min_texts,
            )
            summary["skipped"] += 1
            continue

        try:
            sentiments = [sentiment_fn(t) for t in texts]
            readabilities = [readability_fn(t) for t in texts]
            formalities = [formality_fn(t) for t in texts]
            richnesses = [_vocab_richness(t) for t in texts]

            mean_sentiment = _mean(sentiments)
            std_sentiment = max(_std(sentiments), 0.01)
            mean_flesch = _mean(readabilities)
            std_flesch = max(_std(readabilities), 0.01)
            mean_formality = _mean(formalities)
            std_formality = max(_std(formalities), 0.01)
            mean_vocab = _mean(richnesses)
            std_vocab = max(_std(richnesses), 0.01)
            keywords = _top_keywords(texts, k=10)
            tone = _tone_label(mean_formality, mean_sentiment)

            # Embeddings (optional)
            if compute_embeddings:
                mean_embedding, embedding_status = _try_compute_mean_embedding(texts)
            else:
                mean_embedding, embedding_status = [], "missing"

            profile = {
                "brand_id": brand_id,
                "brand_name": brand_name,
                # Aggregates (rich format with std for Gaussian scoring)
                "mean_sentiment": round(mean_sentiment, 4),
                "std_sentiment": round(std_sentiment, 4),
                "mean_flesch": round(mean_flesch, 2),
                "std_flesch": round(std_flesch, 2),
                "mean_formality": round(mean_formality, 4),
                "std_formality": round(std_formality, 4),
                "mean_vocab_richness": round(mean_vocab, 4),
                "std_vocab_richness": round(std_vocab, 4),
                # Compat aliases (used by scoring.py / main.py)
                "avg_sentiment": round(mean_sentiment, 4),
                "avg_formality": round(mean_formality, 4),
                "avg_readability_flesch": round(mean_flesch, 2),
                "vocabulary_richness": round(mean_vocab, 4),
                # Derived
                "top_keywords": keywords,
                "tone_label": tone,
                # Embeddings
                "mean_embedding": mean_embedding,
                "embedding_status": embedding_status,
                # Metadata
                "n_texts": len(texts),
                "version": version,
                "built_at": built_at,
            }

            if dry_run:
                logger.info(
                    "[DRY RUN] Would upsert %s  (%d texts, embedding=%s)",
                    brand_id, len(texts), embedding_status,
                )
            else:
                cur.execute("""
                    INSERT OR REPLACE INTO brand_profiles
                        (brand_id, brand_name, n_texts, version, built_at, profile_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    brand_id,
                    brand_name,
                    len(texts),
                    version,
                    built_at,
                    json.dumps(profile),
                ))

            summary["processed"] += 1
            summary["brands"].append(brand_id)
            logger.info(
                "✓ %-16s  texts=%2d  sentiment=%.2f  flesch=%.1f  "
                "formality=%.2f  embedding=%s",
                brand_id, len(texts), mean_sentiment, mean_flesch,
                mean_formality, embedding_status,
            )

        except Exception as exc:
            logger.error("✗ %s — %s", brand_id, exc, exc_info=True)
            summary["errors"] += 1

    if not dry_run:
        conn.commit()
        logger.info("Committed %d profiles to %s", summary["processed"], db_path)

    conn.close()
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build brand profiles from brand_texts and store in brand_profiles table.",
    )
    p.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    p.add_argument(
        "--version",
        default="v1",
        help="Version tag stored in each profile (default: v1)",
    )
    p.add_argument(
        "--min-texts-per-brand",
        type=int,
        default=10,
        help="Skip brands with fewer texts (default: 10)",
    )
    p.add_argument(
        "--limit-brands",
        default=None,
        help="Comma-separated brand IDs to process (default: all)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute profiles but do not write to DB.",
    )
    p.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding computation (faster, marks embedding_status='missing').",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = parse_args(argv)

    limit_brands = None
    if args.limit_brands:
        limit_brands = [b.strip() for b in args.limit_brands.split(",") if b.strip()]

    logger.info("Building brand profiles from %s …", args.db_path)

    summary = build_profiles(
        db_path=args.db_path,
        version=args.version,
        min_texts=args.min_texts_per_brand,
        limit_brands=limit_brands,
        dry_run=args.dry_run,
        compute_embeddings=not args.no_embeddings,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  Brand Profile Builder — Summary")
    print("=" * 62)
    print(f"  DB path          : {args.db_path}")
    print(f"  Version          : {args.version}")
    print(f"  Dry run          : {args.dry_run}")
    print(f"  Brands processed : {summary['processed']}")
    print(f"  Brands skipped   : {summary['skipped']}")
    print(f"  Errors           : {summary['errors']}")
    if summary["brands"]:
        print(f"  Brand IDs        : {', '.join(summary['brands'])}")
    print("=" * 62)

    if summary["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
