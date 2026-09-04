"""
rag_builder.py – Stage 5 canonical chunk-level RAG index builder.

Builds one FAISS ``IndexFlatIP`` per brand over L2-normalised MiniLM
embeddings of ``brand_chunks`` rows (inner product on normalised vectors ==
cosine similarity).  SQLite (``brand_chunks``) remains the canonical source
of chunk text/provenance; the artifact only stores lightweight metadata
(chunk_id, text_id, brand_id, brand_name, source_type, char_count, local
row position) plus per-brand FAISS index files and a manifest.

This module intentionally does NOT touch:
    * the brand-centroid index (``src/benchmarking/retrieval.py`` +
      ``scripts/build_embeddings_index.py``) – that is a separate,
      unrelated competitor-similarity tool.
    * ``embeddings/metadata.json`` – legacy five-brand centroid metadata,
      unrelated to this chunk-level RAG artifact.

Public API
----------
* ``fetch_all_chunks(db_path)``
* ``compute_fingerprint(chunks)``
* ``build_index(db_path, out_dir, model_name=...)`` – atomic build.
* ``load_manifest(artifact_dir)`` / ``load_metadata(artifact_dir)``
* ``load_brand_index(artifact_dir, brand_id)``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.feature_extraction.embedding_extractor import get_embedding

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
EMBEDDING_DIM = 384
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
NORMALIZATION = "l2"
SIMILARITY_METRIC = "cosine_via_inner_product"

MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.json"


@dataclass(slots=True)
class RagBuildError(Exception):
    """Raised when the corpus or a generated embedding fails a hard invariant."""

    message: str

    def __str__(self) -> str:
        return self.message


# ── SQLite source of truth ─────────────────────────────────────────────────

def fetch_all_chunks(db_path: str) -> list[dict[str, Any]]:
    """Read every ``brand_chunks`` row in deterministic ``chunk_id`` order."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_id, text_id, brand_id, brand_name, source_type, "
            "chunk_text, char_count FROM brand_chunks ORDER BY chunk_id ASC"
        )
        cols = ("chunk_id", "text_id", "brand_id", "brand_name", "source_type", "chunk_text", "char_count")
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def compute_fingerprint(chunks: list[dict[str, Any]]) -> str:
    """Deterministic SHA-256 over ordered (chunk_id, text_id, brand_id, chunk_text)."""
    hasher = hashlib.sha256()
    for row in chunks:
        hasher.update(str(row["chunk_id"]).encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(str(row["text_id"]).encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(str(row["brand_id"]).encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(str(row["chunk_text"]).encode("utf-8"))
        hasher.update(b"\x1e")
    hasher.update(str(len(chunks)).encode("utf-8"))
    return hasher.hexdigest()


def current_db_fingerprint(db_path: str) -> str | None:
    """Fingerprint of the DB's current brand_chunks state, or ``None`` if empty."""
    chunks = fetch_all_chunks(db_path)
    if not chunks:
        return None
    return compute_fingerprint(chunks)


def validate_corpus(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        raise RagBuildError("brand_chunks is empty; cannot build RAG index")

    seen: set[str] = set()
    for row in chunks:
        chunk_id = row["chunk_id"]
        if chunk_id in seen:
            raise RagBuildError(f"duplicate chunk_id detected: {chunk_id}")
        seen.add(chunk_id)
        if not str(row["chunk_text"] or "").strip():
            raise RagBuildError(f"blank chunk_text for chunk_id={chunk_id}")


def _group_by_brand(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_brand: dict[str, list[dict[str, Any]]] = {}
    for row in chunks:
        by_brand.setdefault(row["brand_id"], []).append(row)
    return by_brand


def _embed_chunk(chunk_text: str, model_name: str) -> np.ndarray:
    vec, _ = get_embedding(chunk_text, model_name=model_name)
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise RagBuildError(f"embedding dimension mismatch: got {arr.shape[0]}, expected {EMBEDDING_DIM}")
    if not np.isfinite(arr).all():
        raise RagBuildError("embedding contains non-finite values")
    return arr


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


def build_index(
    db_path: str,
    out_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """
    Build the canonical per-brand chunk RAG index and atomically publish it
    to *out_dir*.

    Safe sequence: read → validate → fingerprint → embed → build per-brand
    FAISS indexes → validate in a TEMP directory → atomically replace the
    canonical *out_dir*.  On any failure the previous *out_dir* (if any) is
    left untouched.
    """
    chunks = fetch_all_chunks(db_path)
    validate_corpus(chunks)
    fingerprint = compute_fingerprint(chunks)
    by_brand = _group_by_brand(chunks)

    out_path = Path(out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix=".rag_build_tmp_", dir=str(out_path.parent)))
    try:
        indexes_dir = tmp_dir / "indexes"
        indexes_dir.mkdir(parents=True, exist_ok=True)

        manifest_brands: dict[str, Any] = {}
        metadata_map: dict[str, list[dict[str, Any]]] = {}
        total = 0

        for brand_id in sorted(by_brand.keys()):
            brand_chunks = sorted(by_brand[brand_id], key=lambda c: c["chunk_id"])
            vectors: list[np.ndarray] = []
            meta_list: list[dict[str, Any]] = []

            for local_index, row in enumerate(brand_chunks):
                vec = _embed_chunk(row["chunk_text"], model_name)
                vec = _l2_normalize(vec)
                vectors.append(vec)
                meta_list.append(
                    {
                        "local_index": local_index,
                        "chunk_id": row["chunk_id"],
                        "text_id": row["text_id"],
                        "brand_id": row["brand_id"],
                        "brand_name": row["brand_name"],
                        "source_type": row["source_type"],
                        "char_count": row["char_count"],
                    }
                )

            mat = np.vstack(vectors).astype(np.float32)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            index.add(mat)

            index_file = f"{brand_id}.faiss"
            faiss.write_index(index, str(indexes_dir / index_file))

            manifest_brands[brand_id] = {
                "brand_name": brand_chunks[0]["brand_name"],
                "count": len(brand_chunks),
                "index_file": f"indexes/{index_file}",
            }
            metadata_map[brand_id] = meta_list
            total += len(brand_chunks)

        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "embedding_dim": EMBEDDING_DIM,
            "normalization": NORMALIZATION,
            "similarity_metric": SIMILARITY_METRIC,
            "fingerprint": fingerprint,
            "chunk_count": total,
            "brands": manifest_brands,
        }

        (tmp_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
        (tmp_dir / METADATA_FILENAME).write_text(json.dumps(metadata_map, indent=2))

        _validate_build(tmp_dir, manifest, metadata_map, chunks)

        if out_path.exists():
            shutil.rmtree(out_path)
        shutil.move(str(tmp_dir), str(out_path))
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return manifest


def _validate_build(
    tmp_dir: Path,
    manifest: dict[str, Any],
    metadata_map: dict[str, list[dict[str, Any]]],
    source_chunks: list[dict[str, Any]],
) -> None:
    """Hard invariants that must hold before an index build is trusted."""
    all_chunk_ids = {row["chunk_id"] for row in source_chunks}
    seen_chunk_ids: set[str] = set()
    total_vectors = 0

    for brand_id, brand_info in manifest["brands"].items():
        meta_list = metadata_map[brand_id]
        if len(meta_list) != brand_info["count"]:
            raise RagBuildError(f"metadata count mismatch for brand {brand_id}")

        index = faiss.read_index(str(tmp_dir / brand_info["index_file"]))
        if index.ntotal != len(meta_list):
            raise RagBuildError(f"FAISS ntotal mismatch for brand {brand_id}")
        if index.d != EMBEDDING_DIM:
            raise RagBuildError(f"FAISS dimension mismatch for brand {brand_id}")

        for meta in meta_list:
            chunk_id = meta["chunk_id"]
            if chunk_id not in all_chunk_ids:
                raise RagBuildError(f"metadata chunk_id not found in source corpus: {chunk_id}")
            if chunk_id in seen_chunk_ids:
                raise RagBuildError(f"duplicate chunk_id across index: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

        total_vectors += index.ntotal

    if seen_chunk_ids != all_chunk_ids:
        raise RagBuildError("indexed chunk_id set does not exactly match source brand_chunks")
    if total_vectors != manifest["chunk_count"]:
        raise RagBuildError("total indexed vector count does not match manifest chunk_count")


# ── Load helpers (used by retrieval service) ───────────────────────────────

def load_manifest(artifact_dir: str) -> dict[str, Any]:
    path = Path(artifact_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"RAG manifest not found at {path}")
    return json.loads(path.read_text())


def load_metadata(artifact_dir: str) -> dict[str, list[dict[str, Any]]]:
    path = Path(artifact_dir) / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"RAG metadata not found at {path}")
    return json.loads(path.read_text())


def load_brand_index(artifact_dir: str, brand_id: str, index_file: str) -> Any:
    path = Path(artifact_dir) / index_file
    if not path.exists():
        raise FileNotFoundError(f"RAG index file not found for brand {brand_id!r} at {path}")
    return faiss.read_index(str(path))
