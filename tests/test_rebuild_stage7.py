from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_embedding(text, model_name="all-MiniLM-L6-v2"):
    seed = sum(ord(char) for char in str(text))
    vector = [float((seed + index) % 97) / 97.0 for index in range(384)]
    return vector, model_name


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    shutil.copy2(REPO_ROOT / "data" / "brand_data.db", db_path)
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "rag"))
    monkeypatch.setenv("ANALYTICS_CACHE_PATH", str(tmp_path / "analytics_cache.json"))
    monkeypatch.setattr("src.api.genome_service.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.retrieval.rag_builder.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.retrieval.rag_service.get_embedding", _fake_embedding)
    return db_path


@pytest.fixture()
def client(temp_db, monkeypatch):
    from src.api.main import app

    return TestClient(app)


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_genome(client):
    payload = {
        "designation": "Acme Watches",
        "mission_core_vision": "To craft timeless instruments of uncompromising precision.",
        "snippets": [
            "Precision and craftsmanship guide every line.",
            "The tone stays measured, confident, and refined.",
            "We speak with clarity, heritage, and discipline.",
            "Each message should feel exacting and premium.",
            "The voice remains consistent across all touchpoints.",
            "Luxury copy should still feel human and readable.",
            "We value detail, restraint, and unmistakable identity.",
        ],
    }
    response = client.post("/api/genome/init", json=payload)
    assert response.status_code == 200
    return response.json()["profile"]


def _user_profile_rows(db_path):
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT * FROM brand_profile WHERE brand_id = 0").fetchall()
    finally:
        conn.close()


def _history_count(db_path):
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
    finally:
        conn.close()


def _set_user_mission(db_path, text):
    conn = _connect(db_path)
    try:
        conn.execute("UPDATE brand_texts SET text = ? WHERE text_id = 'user_brand__mission'", (text,))
        conn.execute("UPDATE brand_texts_raw SET text = ? WHERE text_id = 'user_brand__mission'", (text,))
        conn.commit()
    finally:
        conn.close()


def _delete_one_user_snippet(db_path):
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM brand_texts WHERE text_id = 'user_brand__snippet_007'")
        conn.execute("DELETE FROM brand_texts_raw WHERE text_id = 'user_brand__snippet_007'")
        conn.commit()
    finally:
        conn.close()


def test_profile_rebuild_requires_user_corpus(client):
    response = client.post("/api/rebuild/profile")
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "user_corpus_missing"


def test_profile_rebuild_is_real_and_idempotent(client, temp_db):
    _init_genome(client)
    before_history = _history_count(temp_db)

    first = client.post("/api/rebuild/profile")
    assert first.status_code == 200
    body1 = first.json()
    assert body1["status"] == "ok"
    assert body1["source_texts"] == 8
    assert body1["embedding_dim"] == 384

    second = client.post("/api/rebuild/profile")
    assert second.status_code == 200
    body2 = second.json()
    assert body2["embedding_dim"] == 384
    assert body2["genome_version"] == body1["genome_version"]

    rows = _user_profile_rows(temp_db)
    assert len(rows) == 1
    conn = _connect(temp_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0] == before_history
        assert conn.execute("SELECT COUNT(*) FROM brand_texts WHERE brand_id = 'user_brand'").fetchone()[0] == 8
        assert conn.execute("SELECT COUNT(*) FROM brand_texts_raw WHERE brand_id = 'user_brand'").fetchone()[0] == 8
    finally:
        conn.close()


def test_profile_rebuild_reads_changed_sqlite_source(client, temp_db):
    _init_genome(client)
    _set_user_mission(temp_db, "Updated mission text from direct SQLite edit.")

    response = client.post("/api/rebuild/profile")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"

    conn = _connect(temp_db)
    try:
        metadata = json.loads(conn.execute("SELECT metadata_json FROM brand_profile WHERE brand_id = 0").fetchone()[0])
    finally:
        conn.close()
    assert metadata["mission_core_vision"] == "Updated mission text from direct SQLite edit."


def test_profile_rebuild_rejects_invalid_source_cardinality_without_corrupting_prior_profile(client, temp_db):
    _init_genome(client)
    before_rows = _user_profile_rows(temp_db)
    assert len(before_rows) == 1

    _delete_one_user_snippet(temp_db)
    response = client.post("/api/rebuild/profile")
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "invalid_user_corpus"

    after_rows = _user_profile_rows(temp_db)
    assert len(after_rows) == 1


def test_chunks_rebuild_updates_chunks_and_index(client, temp_db):
    _init_genome(client)
    response = client.post("/api/rebuild/chunks")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_rebuilt"] is True
    assert body["user_chunks"] > 0
    assert body["per_brand"]["user_brand"] == body["user_chunks"]


def test_chunks_rebuild_requires_source_texts(client, temp_db):
    conn = _connect(temp_db)
    try:
        conn.execute("DELETE FROM brand_texts")
        conn.execute("DELETE FROM brand_texts_raw")
        conn.commit()
    finally:
        conn.close()

    response = client.post("/api/rebuild/chunks")
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "empty_brand_texts"


def test_chunks_rebuild_is_repeatable(client, temp_db):
    _init_genome(client)
    first = client.post("/api/rebuild/chunks")
    second = client.post("/api/rebuild/chunks")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["fingerprint"] == second.json()["fingerprint"]

    conn = _connect(temp_db)
    try:
        count1 = conn.execute("SELECT COUNT(*) FROM brand_chunks").fetchone()[0]
    finally:
        conn.close()
    assert count1 > 0


def test_chunks_rebuild_partial_failure_reports_stale_index(client, temp_db, monkeypatch):
    _init_genome(client)
    first_index = client.post("/api/rebuild/index")
    assert first_index.status_code == 200

    _set_user_mission(temp_db, "Changed mission text before chunk rebuild failure.")

    from src.rebuild import rebuild_service

    def _fail_build(*args, **kwargs):
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(rebuild_service, "build_index", _fail_build)
    response = client.post("/api/rebuild/chunks")
    assert response.status_code == 207
    body = response.json()
    assert body["status"] == "partial"
    assert body["chunks_rebuilt"] is True
    assert body["index_rebuilt"] is False
    assert body["analytics_cache"] in {"stale", "missing"}


def test_index_rebuild_is_real_and_idempotent(client, temp_db):
    _init_genome(client)
    first = client.post("/api/rebuild/index")
    assert first.status_code == 200
    body1 = first.json()
    assert body1["embedding_dim"] == 384

    second = client.post("/api/rebuild/index")
    assert second.status_code == 200
    body2 = second.json()
    assert body2["fingerprint"] == body1["fingerprint"]
    assert body2["brands_indexed"] == body1["brands_indexed"]


def test_rebuild_sequence_preserves_retrieval_and_consistency(client, temp_db):
    _init_genome(client)
    before_history = _history_count(temp_db)
    assert client.post("/api/rebuild/profile").status_code == 200
    assert client.post("/api/rebuild/profile").status_code == 200
    assert client.post("/api/rebuild/chunks").status_code in {200, 207}
    assert client.post("/api/rebuild/chunks").status_code in {200, 207}
    assert client.post("/api/rebuild/index").status_code == 200
    assert client.post("/api/rebuild/index").status_code == 200
    assert _history_count(temp_db) == before_history

    retrieval = client.post(
        "/api/rag/retrieve",
        json={"text": "precision and craftsmanship", "brand_id": "user_brand", "top_k": 5},
    )
    assert retrieval.status_code == 200
    assert retrieval.json()["results"]

    consistency = client.post("/api/consistency/score", json={"text": "precision and craftsmanship"})
    assert consistency.status_code == 200
    assert 0 <= consistency.json()["score_overall"] <= 100
