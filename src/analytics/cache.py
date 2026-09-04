"""
cache.py – Analytics artifact builder / loader for Stage 4.

Bundles the corpus-derived (relatively expensive) components:
    - automatically derived Messaging Pillar keyword sets
    - TF-IDF pillar heatmap
    - chunk-level t-SNE sample
    - competitor tone distribution

Live analysis_history counters/trend are NOT included here; see
``src/analytics/history.py`` which reads SQLite directly at request time.

The artifact is validated against a corpus fingerprint (chunk/text ids +
counts) before being trusted; a stale or missing artifact is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analytics.chunk_tsne import compute_chunk_tsne
from src.analytics.heatmap import compute_pillar_heatmap
from src.analytics.pillars import PILLAR_NAMES, derive_pillar_keywords
from src.analytics.tone import compute_tone_distribution

ARTIFACT_VERSION = 1


def _fetch_brand_texts(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT text_id, brand_id, brand_name, text FROM brand_texts ORDER BY text_id")
    return [
        {"text_id": row[0], "brand_id": row[1], "brand_name": row[2], "text": row[3]}
        for row in cur.fetchall()
    ]


def _fetch_brand_chunks(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, brand_id, brand_name, chunk_text FROM brand_chunks ORDER BY chunk_id")
    return [
        {"chunk_id": row[0], "brand_id": row[1], "brand_name": row[2], "chunk_text": row[3]}
        for row in cur.fetchall()
    ]


def _fingerprint(brand_texts: list[dict], brand_chunks: list[dict]) -> str:
    hasher = hashlib.sha256()
    for row in brand_texts:
        hasher.update(str(row["text_id"]).encode("utf-8"))
    for row in brand_chunks:
        hasher.update(str(row["chunk_id"]).encode("utf-8"))
    hasher.update(str(len(brand_texts)).encode("utf-8"))
    hasher.update(str(len(brand_chunks)).encode("utf-8"))
    return hasher.hexdigest()


def _embedding_mode() -> str:
    from src.feature_extraction.embedding_extractor import get_embedding

    _, model_name = get_embedding("probe")
    try:
        import sentence_transformers  # noqa: F401

        return f"sentence-transformers:{model_name}"
    except ImportError:
        return f"hash-fallback:{model_name}"


def build_analytics_artifact(db_path: str) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        brand_texts = _fetch_brand_texts(conn)
        brand_chunks = _fetch_brand_chunks(conn)
    finally:
        conn.close()

    if not brand_texts:
        raise ValueError("No brand_texts rows found; cannot build analytics artifact.")

    embedding_mode = _embedding_mode()

    # 1. Pillar keyword derivation (corpus-driven, no hardcoded dictionary)
    documents = [row["text"] for row in brand_texts]
    pillar_terms = derive_pillar_keywords(documents)

    # 2. Heatmap (one concatenated document per competitor brand)
    brand_order: list[str] = []
    brand_name_by_id: dict[str, str] = {}
    for row in brand_texts:
        if row["brand_id"] not in brand_name_by_id:
            brand_name_by_id[row["brand_id"]] = row["brand_name"]
            brand_order.append(row["brand_id"])
    brand_order.sort()

    brand_docs = [
        " ".join(row["text"] for row in brand_texts if row["brand_id"] == brand_id)
        for brand_id in brand_order
    ]

    heatmap = compute_pillar_heatmap(
        brand_ids=brand_order,
        brand_names=[brand_name_by_id[b] for b in brand_order],
        brand_documents=brand_docs,
        pillar_names=PILLAR_NAMES,
        pillar_terms=pillar_terms,
    )

    # 3. Chunk-level t-SNE (source: brand_chunks, never centroids)
    tsne = compute_chunk_tsne(brand_chunks)

    # 4. Real tone distribution over competitor brand_texts
    tone = compute_tone_distribution(brand_texts)

    fingerprint = _fingerprint(brand_texts, brand_chunks)

    return {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "source_counts": {
            "brand_texts": len(brand_texts),
            "brand_chunks": len(brand_chunks),
            "competitor_count": len(brand_order),
        },
        "embedding_mode": embedding_mode,
        "pillars": {"names": PILLAR_NAMES, "keywords": pillar_terms},
        "heatmap": heatmap,
        "tsne": tsne,
        "tone": tone,
    }


def _atomic_write_json(path: str, data: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def build_and_save(db_path: str, cache_path: str) -> dict[str, Any]:
    artifact = build_analytics_artifact(db_path)
    _atomic_write_json(cache_path, artifact)
    return artifact


def _current_fingerprint(db_path: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        brand_texts = _fetch_brand_texts(conn)
        brand_chunks = _fetch_brand_chunks(conn)
    finally:
        conn.close()
    if not brand_texts:
        return None
    return _fingerprint(brand_texts, brand_chunks)


def load_or_build(db_path: str, cache_path: str) -> dict[str, Any]:
    """Load the cached artifact if present and fingerprint-valid; else rebuild."""
    if Path(cache_path).exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            current_fp = _current_fingerprint(db_path)
            if current_fp is not None and cached.get("fingerprint") == current_fp:
                return cached
        except (json.JSONDecodeError, OSError):
            pass
    return build_and_save(db_path, cache_path)
