from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.build_brand_chunks import build_candidates
from src.analytics.cache import get_cache_state
from src.api.genome_service import (
    EMBEDDING_DIM,
    SNIPPET_COUNT,
    USER_BRAND_ALIAS,
    USER_BRAND_DB_ID,
    _build_feature_bundle,
    ensure_canonical_schema,
)
from src.feature_extraction.feature_utils import clean_text
from src.retrieval.rag_builder import DEFAULT_MODEL_NAME, RagBuildError, build_index, compute_fingerprint, current_db_fingerprint
from src.retrieval.rag_service import RagError, retrieve_chunks


_REBUILD_LOCK = threading.Lock()


class RebuildError(Exception):
    def __init__(self, status_code: int, detail: dict[str, Any]):
        super().__init__(detail.get("message") or detail.get("error") or "rebuild_error")
        self.status_code = status_code
        self.detail = detail


def _default_db_path() -> str:
    return os.getenv("SQLITE_DB_PATH", "data/brand_data.db")


def _default_rag_dir() -> str:
    return os.getenv("RAG_INDEX_DIR", "data/processed/rag")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@contextmanager
def _rebuild_guard():
    if not _REBUILD_LOCK.acquire(blocking=False):
        raise RebuildError(
            409,
            {
                "error": "rebuild_in_progress",
                "message": "Another rebuild is already running.",
            },
        )
    try:
        yield
    finally:
        _REBUILD_LOCK.release()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_canonical_schema(conn)
    return conn


def _fetch_rows(conn: sqlite3.Connection, table: str, brand_id: str) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        f"SELECT * FROM {table} WHERE brand_id = ? ORDER BY text_id",
        (brand_id,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    return rows


def _load_user_rows(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    texts = _fetch_rows(conn, "brand_texts", USER_BRAND_ALIAS)
    raw_rows = _fetch_rows(conn, "brand_texts_raw", USER_BRAND_ALIAS)
    return texts, raw_rows


def _validate_user_source_rows(text_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not text_rows:
        raise RebuildError(
            409,
            {
                "error": "user_corpus_missing",
                "message": "No user_brand source rows were found in brand_texts.",
            },
        )

    if len(text_rows) != SNIPPET_COUNT + 1:
        raise RebuildError(
            409,
            {
                "error": "invalid_user_corpus",
                "message": "user_brand source corpus must contain exactly one mission row and seven snippet rows.",
            },
        )

    expected_ids = {"user_brand__mission", *{f"user_brand__snippet_{index:03d}" for index in range(1, SNIPPET_COUNT + 1)}}
    expected_types = {"user_brand__mission": "genome_mission", **{f"user_brand__snippet_{index:03d}": "genome_snippet" for index in range(1, SNIPPET_COUNT + 1)}}

    text_rows_by_id = {row["text_id"]: row for row in text_rows}
    raw_rows_by_id = {row["text_id"]: row for row in raw_rows}

    if set(text_rows_by_id) != expected_ids:
        raise RebuildError(
            409,
            {
                "error": "invalid_user_corpus",
                "message": "user_brand source text_ids do not match the canonical mission/snippet contract.",
            },
        )
    if len(raw_rows) != SNIPPET_COUNT + 1:
        raise RebuildError(
            409,
            {
                "error": "invalid_user_corpus",
                "message": "user_brand raw source corpus must contain exactly one mission row and seven snippet rows.",
            },
        )
    if set(raw_rows_by_id) != expected_ids:
        raise RebuildError(
            409,
            {
                "error": "invalid_user_corpus",
                "message": "user_brand raw source rows do not match the canonical mission/snippet contract.",
            },
        )

    for text_id, row in text_rows_by_id.items():
        expected_type = expected_types[text_id]
        if row.get("source_type") != expected_type:
            raise RebuildError(
                409,
                {
                    "error": "invalid_user_corpus",
                    "message": f"user_brand source_type mismatch for {text_id}.",
                },
            )
        if not clean_text(row.get("text")):
            raise RebuildError(
                409,
                {
                    "error": "invalid_user_corpus",
                    "message": f"user_brand source text is blank for {text_id}.",
                },
            )
        raw_row = raw_rows_by_id.get(text_id)
        if not raw_row or (raw_row.get("text") or "") != (row.get("text") or ""):
            raise RebuildError(
                409,
                {
                    "error": "invalid_user_corpus",
                    "message": f"user_brand raw/source mismatch for {text_id}.",
                },
            )

    ordered = [text_rows_by_id["user_brand__mission"]] + [
        text_rows_by_id[f"user_brand__snippet_{index:03d}"] for index in range(1, SNIPPET_COUNT + 1)
    ]
    mission = ordered[0]["text"]
    snippets = [row["text"] for row in ordered[1:]]
    designation = ordered[0].get("brand_name") or ordered[0].get("brand_id") or ""
    return {
        "designation": designation,
        "mission_core_vision": mission,
        "snippets": snippets,
        "rows": ordered,
        "source_count": len(ordered),
    }


def _current_profile_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT metadata_json FROM brand_profile WHERE brand_id = ? ORDER BY id DESC LIMIT 1",
        (USER_BRAND_DB_ID,),
    ).fetchone()
    if not row:
        return 1
    try:
        metadata = json.loads(row[0] or "{}")
        return int(metadata.get("genome_version", 1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 1


def _current_profile_created_at(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT created_at FROM brand_profile WHERE brand_id = ? ORDER BY id DESC LIMIT 1",
        (USER_BRAND_DB_ID,),
    ).fetchone()
    return row[0] if row else None


def _count_brand_chunks(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.cursor()
    cur.execute("SELECT brand_id, COUNT(*) FROM brand_chunks GROUP BY brand_id ORDER BY brand_id")
    return {row[0]: row[1] for row in cur.fetchall()}


def _count_brand_texts(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.cursor()
    cur.execute("SELECT brand_id, COUNT(*) FROM brand_texts GROUP BY brand_id ORDER BY brand_id")
    return {row[0]: row[1] for row in cur.fetchall()}


def _validate_candidate_rows(rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    source_by_text_id = {row["text_id"]: row for row in rows}
    seen_ids: set[str] = set()
    failures: list[str] = []

    counts = Counter(candidate["brand_id"] for candidate in candidates)
    source_counts = Counter(row["brand_id"] for row in rows)

    for row in rows:
        if not clean_text(row.get("text")):
            failures.append(f"blank source text detected for text_id={row.get('text_id')}")

    for brand_id, count in source_counts.items():
        if brand_id == USER_BRAND_ALIAS:
            continue
        if count < 30:
            failures.append(f"brand '{brand_id}' has {count} source texts (< 30)")
        if counts.get(brand_id, 0) < 50:
            failures.append(f"brand '{brand_id}' has {counts.get(brand_id, 0)} candidate chunks (< 50)")

    user_rows = [row for row in rows if row.get("brand_id") == USER_BRAND_ALIAS]
    if user_rows:
        user_text_ids = {row["text_id"] for row in user_rows}
        if len(user_rows) != SNIPPET_COUNT + 1:
            failures.append("user_brand must contain exactly 8 source rows")
        if user_text_ids != {"user_brand__mission", *{f"user_brand__snippet_{i:03d}" for i in range(1, SNIPPET_COUNT + 1)}}:
            failures.append("user_brand text_ids do not match the canonical mission/snippet contract")
        user_candidate_text_ids = {candidate["text_id"] for candidate in candidates if candidate["brand_id"] == USER_BRAND_ALIAS}
        missing_user_sources = sorted(user_text_ids - user_candidate_text_ids)
        if missing_user_sources:
            failures.append(f"user_brand source coverage missing for {missing_user_sources}")

    for candidate in candidates:
        if candidate["chunk_id"] in seen_ids:
            failures.append(f"duplicate chunk_id detected: {candidate['chunk_id']}")
        seen_ids.add(candidate["chunk_id"])
        if candidate["char_count"] > 400:
            failures.append(f"chunk {candidate['chunk_id']} exceeds 400 chars")
        if not (candidate.get("chunk_text") or "").strip():
            failures.append(f"chunk {candidate['chunk_id']} is blank")
        if candidate["text_id"] not in source_by_text_id:
            failures.append(f"chunk {candidate['chunk_id']} references missing text_id {candidate['text_id']}")

    if failures:
        raise RebuildError(
            409,
            {
                "error": "invalid_candidate_corpus",
                "message": "Chunk candidate validation failed.",
                "failures": failures,
            },
        )


def _replace_brand_chunks(conn: sqlite3.Connection, candidates: list[dict[str, Any]]) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM brand_chunks")
    cur.executemany(
        "INSERT INTO brand_chunks (chunk_id, text_id, brand_id, brand_name, source_type, chunk_text, char_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (item["chunk_id"], item["text_id"], item["brand_id"], item["brand_name"], item["source_type"], item["chunk_text"], item["char_count"])
            for item in candidates
        ],
    )


def _validate_index_manifest(db_path: str, artifact_dir: str, manifest: dict[str, Any]) -> None:
    live_fingerprint = current_db_fingerprint(db_path)
    if live_fingerprint != manifest.get("fingerprint"):
        raise RebuildError(
            500,
            {
                "error": "index_fingerprint_mismatch",
                "message": "Built RAG manifest does not match the current SQLite corpus.",
            },
        )

    conn = _connect(db_path)
    try:
        total_chunks = conn.execute("SELECT COUNT(*) FROM brand_chunks").fetchone()[0]
        if int(manifest.get("chunk_count", -1)) != int(total_chunks):
            raise RebuildError(
                500,
                {
                    "error": "index_metadata_mismatch",
                    "message": "RAG manifest chunk_count does not match SQLite brand_chunks.",
                },
            )

        metadata = json.loads(Path(artifact_dir, "metadata.json").read_text())
        for brand_id, info in manifest.get("brands", {}).items():
            brand_metadata = metadata.get(brand_id, [])
            if len(brand_metadata) != info.get("count"):
                raise RebuildError(
                    500,
                    {
                        "error": "index_metadata_mismatch",
                        "message": f"Brand metadata count mismatch for {brand_id}.",
                    },
                )

        smoke_brand_id = USER_BRAND_ALIAS if USER_BRAND_ALIAS in manifest.get("brands", {}) else next(iter(manifest.get("brands", {})), None)
        if smoke_brand_id:
            sample_row = conn.execute(
                "SELECT chunk_text FROM brand_chunks WHERE brand_id = ? ORDER BY chunk_id LIMIT 1",
                (smoke_brand_id,),
            ).fetchone()
            if sample_row:
                smoke_query = sample_row[0].split(" ")[0] or sample_row[0][:16]
                smoke = retrieve_chunks(
                    smoke_query,
                    smoke_brand_id,
                    artifact_dir=artifact_dir,
                    db_path=db_path,
                )
                if not smoke.get("results"):
                    raise RebuildError(
                        500,
                        {
                            "error": "index_retrieval_smoke_failed",
                            "message": "RAG retrieval smoke returned no results.",
                        },
                    )
                if smoke_brand_id == USER_BRAND_ALIAS and not any(item["brand_id"] == USER_BRAND_ALIAS for item in smoke["results"]):
                    raise RebuildError(
                        500,
                        {
                            "error": "index_retrieval_smoke_failed",
                            "message": "User brand retrieval smoke did not return user_brand chunks.",
                        },
                    )
    finally:
        conn.close()


def rebuild_user_profile(db_path: str | None = None) -> dict[str, Any]:
    db_path = db_path or _default_db_path()
    with _rebuild_guard():
        conn = _connect(db_path)
        try:
            text_rows, raw_rows = _load_user_rows(conn)
            user_source = _validate_user_source_rows(text_rows, raw_rows)
            bundle = _build_feature_bundle(
                user_source["designation"],
                user_source["mission_core_vision"],
                user_source["snippets"],
            )
            now = _utc_now()
            current_version = _current_profile_version(conn)
            created_at = _current_profile_created_at(conn) or now

            tone_features = dict(bundle["tone_features"])
            tone_features["genome_version"] = current_version
            tone_features["profile_rebuilt_at"] = now

            metadata = {
                "brand_id": USER_BRAND_ALIAS,
                "brand_db_id": USER_BRAND_DB_ID,
                "designation": user_source["designation"],
                "mission_core_vision": user_source["mission_core_vision"],
                "snippets": user_source["snippets"],
                "sample_texts": bundle["sample_texts"],
                "sample_embeddings": bundle["sample_embeddings"],
                "embedding_model": bundle["embedding_model"],
                "embedding_dim": bundle["embedding_dim"],
                "genome_version": current_version,
                "sample_count": len(bundle["sample_texts"]),
                "snippet_count": len(user_source["snippets"]),
                "initialized_at": created_at,
                "updated_at": now,
                "profile_rebuilt_at": now,
            }

            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO brands (id, designation, mission_core_vision, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        USER_BRAND_DB_ID,
                        user_source["designation"],
                        user_source["mission_core_vision"],
                        created_at,
                        now,
                    ),
                )
                conn.execute("DELETE FROM brand_profile WHERE brand_id = ?", (USER_BRAND_DB_ID,))
                conn.execute(
                    """
                    INSERT INTO brand_profile (
                        brand_id,
                        keywords_json,
                        tone_features_json,
                        aggregate_embedding,
                        metadata_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        USER_BRAND_DB_ID,
                        json.dumps(bundle["top_keywords"]),
                        json.dumps(tone_features),
                        json.dumps(bundle["aggregate_embedding"]),
                        json.dumps(metadata),
                        created_at,
                        now,
                    ),
                )

            profile_row = conn.execute(
                "SELECT * FROM brand_profile WHERE brand_id = ? ORDER BY id DESC LIMIT 1",
                (USER_BRAND_DB_ID,),
            ).fetchone()
            return {
                "status": "ok",
                "action": "profile",
                "brand_id": USER_BRAND_DB_ID,
                "brand_alias": USER_BRAND_ALIAS,
                "designation": user_source["designation"],
                "source_texts": len(user_source["rows"]),
                "mission_sources": 1,
                "snippet_sources": SNIPPET_COUNT,
                "embedding_dim": bundle["embedding_dim"],
                "genome_version": current_version,
                "updated_at": profile_row["updated_at"] if profile_row else None,
                "message": "User profile rebuilt from canonical SQLite source texts.",
            }
        finally:
            conn.close()


def _build_and_validate_index(db_path: str, artifact_dir: str) -> dict[str, Any]:
    manifest = build_index(db_path, artifact_dir, model_name=DEFAULT_MODEL_NAME)
    _validate_index_manifest(db_path, artifact_dir, manifest)
    return manifest


def rebuild_rag_index(db_path: str | None = None, artifact_dir: str | None = None) -> dict[str, Any]:
    db_path = db_path or _default_db_path()
    artifact_dir = artifact_dir or _default_rag_dir()
    with _rebuild_guard():
        conn = _connect(db_path)
        try:
            total_chunks = conn.execute("SELECT COUNT(*) FROM brand_chunks").fetchone()[0]
            if not total_chunks:
                raise RebuildError(
                    409,
                    {
                        "error": "empty_brand_chunks",
                        "message": "brand_chunks is empty; cannot rebuild the RAG index.",
                    },
                )
        finally:
            conn.close()

        manifest = _build_and_validate_index(db_path, artifact_dir)
        conn = _connect(db_path)
        try:
            counts = _count_brand_chunks(conn)
        finally:
            conn.close()
        return {
            "status": "ok",
            "action": "index",
            "total_chunks": int(manifest["chunk_count"]),
            "brands_indexed": len(manifest["brands"]),
            "embedding_dim": int(manifest["embedding_dim"]),
            "model_name": manifest["model_name"],
            "fingerprint": manifest["fingerprint"],
            "fingerprint_short": manifest["fingerprint"][:12],
            "per_brand": {brand_id: info["count"] for brand_id, info in manifest["brands"].items()},
            "per_brand_db": counts,
            "message": "RAG index rebuilt from current canonical brand_chunks.",
        }


def rebuild_chunks(db_path: str | None = None, artifact_dir: str | None = None) -> dict[str, Any]:
    db_path = db_path or _default_db_path()
    artifact_dir = artifact_dir or _default_rag_dir()
    with _rebuild_guard():
        conn = _connect(db_path)
        try:
            source_rows = [dict(row) for row in conn.execute("SELECT text_id, brand_id, brand_name, source_type, text FROM brand_texts ORDER BY text_id").fetchall()]
            if not source_rows:
                raise RebuildError(
                    409,
                    {
                        "error": "empty_brand_texts",
                        "message": "brand_texts is empty; cannot rebuild chunks.",
                    },
                )

            candidate_result = build_candidates(source_rows)
            _validate_candidate_rows(source_rows, candidate_result.candidates)

            with conn:
                _replace_brand_chunks(conn, candidate_result.candidates)

            fingerprint = current_db_fingerprint(db_path)
            per_brand = _count_brand_chunks(conn)
            total_chunks = conn.execute("SELECT COUNT(*) FROM brand_chunks").fetchone()[0]
            analytics_state = get_cache_state(db_path, os.getenv("ANALYTICS_CACHE_PATH", "data/processed/analytics_cache.json"))

        finally:
            conn.close()

        index_result: dict[str, Any] | None = None
        index_error: dict[str, Any] | None = None
        try:
            manifest = _build_and_validate_index(db_path, artifact_dir)
            index_result = {
                "index_rebuilt": True,
                "index_status": "ok",
                "index_manifest": {
                    "fingerprint": manifest["fingerprint"],
                    "chunk_count": manifest["chunk_count"],
                    "brands": len(manifest["brands"]),
                },
                "index_message": "RAG index rebuilt from current chunks.",
            }
        except Exception as exc:
            index_error = {
                "index_rebuilt": False,
                "index_status": "failed",
                "index_error": str(exc),
            }

        body = {
            "status": "ok" if index_error is None else "partial",
            "action": "chunks",
            "chunks_rebuilt": True,
            "total_chunks": int(total_chunks),
            "competitor_chunks": int(sum(count for brand, count in per_brand.items() if brand != USER_BRAND_ALIAS)),
            "user_chunks": int(per_brand.get(USER_BRAND_ALIAS, 0)),
            "per_brand": per_brand,
            "fingerprint": fingerprint,
            "analytics_cache": analytics_state["state"],
            "message": "brand_chunks rebuilt from canonical texts." if index_error is None else "brand_chunks rebuilt, but RAG index rebuild failed.",
        }
        if index_error is not None:
            body.update(index_error)
            body["analytics_cache"] = analytics_state["state"] if analytics_state else "stale"
        else:
            body.update(index_result or {})
        return body


rebuild_index = rebuild_rag_index
