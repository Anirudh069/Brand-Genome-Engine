"""
rag_service.py – Stage 5 canonical chunk-level semantic retrieval service.

Thin, framework-agnostic wrapper intended to be imported directly by
Stage 6 (Rewrite grounding) and by the ``/api/rag/retrieve`` endpoint.

Given a query and a brand context, embeds the query with the same local
MiniLM abstraction used everywhere else in this repo
(``src.feature_extraction.embedding_extractor.get_embedding``), searches
the brand-scoped FAISS ``IndexFlatIP`` built by
``src.retrieval.rag_builder``, and returns ranked chunks whose text is
fetched live from the canonical SQLite ``brand_chunks`` table.

Brand scoping is structural: each brand has its own FAISS index file, so a
query against brand A can mathematically never surface brand B vectors.
"""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.feature_extraction.embedding_extractor import get_embedding
from src.retrieval.rag_builder import (
    current_db_fingerprint,
    load_brand_index,
    load_manifest,
    load_metadata,
)

DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 10


@dataclass(slots=True)
class RagError(Exception):
    """Structured retrieval error, mirroring ``BenchmarkError``'s shape."""

    status_code: int
    detail: dict[str, Any]

    def __str__(self) -> str:
        return str(self.detail.get("message") or self.detail.get("error") or "rag_error")


def _default_db_path() -> str:
    return os.getenv("SQLITE_DB_PATH", "data/brand_data.db")


def _default_artifact_dir() -> str:
    return os.getenv("RAG_INDEX_DIR", "data/processed/rag")


def _validate_query_text(query_text: Any) -> str:
    if not isinstance(query_text, str) or not query_text.strip():
        raise RagError(400, {"error": "invalid_query", "message": "query text must be non-blank"})
    return query_text.strip()


def _validate_top_k(top_k: int | None) -> int:
    if top_k is None:
        return DEFAULT_TOP_K
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise RagError(400, {"error": "invalid_top_k", "message": "top_k must be an integer"})
    if top_k < MIN_TOP_K or top_k > MAX_TOP_K:
        raise RagError(
            400,
            {
                "error": "invalid_top_k",
                "message": f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}",
            },
        )
    return top_k


def _load_index_state(artifact_dir: str, db_path: str) -> dict[str, Any]:
    try:
        manifest = load_manifest(artifact_dir)
    except FileNotFoundError as exc:
        raise RagError(503, {"error": "index_missing", "message": str(exc)}) from exc

    live_fingerprint = current_db_fingerprint(db_path)
    if live_fingerprint is None:
        raise RagError(503, {"error": "index_missing", "message": "brand_chunks table is empty"})
    if live_fingerprint != manifest["fingerprint"]:
        raise RagError(
            503,
            {
                "error": "index_stale",
                "message": "RAG index fingerprint does not match current brand_chunks corpus; rebuild required",
            },
        )
    return manifest


def _fetch_chunk_texts(db_path: str, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in chunk_ids)
        cur.execute(
            f"SELECT chunk_id, chunk_text FROM brand_chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def retrieve_chunks(
    query_text: str,
    brand_id: str,
    top_k: int | None = DEFAULT_TOP_K,
    db_path: str | None = None,
    artifact_dir: str | None = None,
) -> dict[str, Any]:
    """
    Strict brand-scoped semantic retrieval over the canonical chunk RAG index.

    Returns
    -------
    dict with keys: brand_id, top_k, model, fingerprint, results
        Each result: rank, chunk_id, text_id, brand_id, brand_name,
        source_type, chunk_text, score.
    """
    query_text = _validate_query_text(query_text)
    top_k = _validate_top_k(top_k)
    db_path = db_path or _default_db_path()
    artifact_dir = artifact_dir or _default_artifact_dir()

    manifest = _load_index_state(artifact_dir, db_path)

    brands = manifest["brands"]
    if brand_id not in brands:
        raise RagError(
            404,
            {"error": "unknown_brand", "brand_id": brand_id, "message": "brand_id is not present in the RAG index"},
        )

    brand_info = brands[brand_id]
    metadata_map = load_metadata(artifact_dir)
    brand_metadata = metadata_map[brand_id]

    index = load_brand_index(artifact_dir, brand_id, brand_info["index_file"])
    k = min(top_k, index.ntotal)

    model_name = manifest["model_name"]
    query_vec, _ = get_embedding(query_text, model_name=model_name)
    query_arr = np.asarray(query_vec, dtype=np.float32)
    norm = float(np.linalg.norm(query_arr))
    if norm > 0.0:
        query_arr = query_arr / norm

    scores, ids = index.search(query_arr.reshape(1, -1), k)

    chunk_ids = [brand_metadata[idx]["chunk_id"] for idx in ids[0] if idx != -1]
    chunk_texts = _fetch_chunk_texts(db_path, chunk_ids)

    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx == -1:
            continue
        meta = brand_metadata[idx]
        score_f = float(score)
        if not math.isfinite(score_f):
            raise RagError(500, {"error": "non_finite_score", "chunk_id": meta["chunk_id"]})
        results.append(
            {
                "rank": rank,
                "chunk_id": meta["chunk_id"],
                "text_id": meta["text_id"],
                "brand_id": meta["brand_id"],
                "brand_name": meta["brand_name"],
                "source_type": meta["source_type"],
                "chunk_text": chunk_texts.get(meta["chunk_id"], ""),
                "score": score_f,
            }
        )

    return {
        "brand_id": brand_id,
        "top_k": top_k,
        "model": model_name,
        "fingerprint": manifest["fingerprint"],
        "results": results,
    }
