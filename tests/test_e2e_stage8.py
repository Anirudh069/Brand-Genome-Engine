"""
tests/test_e2e_stage8.py – Stage 8 canonical end-to-end product journey.

GENOME SETUP -> CONSISTENCY -> BENCHMARK -> ANALYTICS -> REWRITE (stale index
-> rebuild -> success) -> ANALYTICS refresh -> REBUILD (profile/chunks) ->
retrieval/consistency still work.

Uses a TEMP copy of the canonical DB, a TEMP RAG artifact directory and a
TEMP analytics cache. Embeddings are mocked (deterministic, no real model
load) and the Rewrite provider is a dependency-injected fake (no network,
no OpenAI credits spent). The checked-in data/brand_data.db is never
written to.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DB = REPO_ROOT / "data" / "brand_data.db"

_SNIPPETS = [
    "Precision and craftsmanship guide every line.",
    "The tone stays measured, confident, and refined.",
    "We speak with clarity, heritage, and discipline.",
    "Each message should feel exacting and premium.",
    "The voice remains consistent across all touchpoints.",
    "Luxury copy should still feel human and readable.",
    "We value detail, restraint, and unmistakable identity.",
]
_MISSION = "To craft timeless instruments of uncompromising precision."
_DESIGNATION = "Acme Watches"


def _fake_embedding(text, model_name="all-MiniLM-L6-v2"):
    seed = sum(ord(char) for char in str(text))
    vector = [float((seed + index) % 97) / 97.0 for index in range(384)]
    return vector, model_name


class FakeRewriteProvider:
    """Deterministic, dependency-injected mock provider. No network calls."""

    name = "openai"
    model = "gpt-5.6-luna-test"

    def __init__(self, response_text="A refined line of enduring craftsmanship and precision."):
        self.response_text = response_text
        self.calls: list[dict] = []

    def rewrite(self, *, instructions, input_text, max_output_tokens=400):
        self.calls.append({"instructions": instructions, "input_text": input_text})
        return self.response_text


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    shutil.copy2(CANONICAL_DB, db_path)
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "rag"))
    monkeypatch.setenv("ANALYTICS_CACHE_PATH", str(tmp_path / "analytics_cache.json"))

    monkeypatch.setattr("src.api.genome_service.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.retrieval.rag_builder.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.retrieval.rag_service.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.scoring.consistency.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.analytics.pillars.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.analytics.chunk_tsne.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.feature_extraction.embedding_extractor.get_embedding", _fake_embedding)
    return db_path


@pytest.fixture()
def fake_provider(monkeypatch):
    provider = FakeRewriteProvider()
    monkeypatch.setattr("src.rewrite.rewrite_service.build_provider", lambda: provider)
    return provider


@pytest.fixture()
def client(temp_db, fake_provider):
    from src.api.main import app

    return TestClient(app)


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _history_counts(db_path):
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM analysis_history GROUP BY event_type"
        ).fetchall()
        return {row["event_type"]: row["n"] for row in rows}
    finally:
        conn.close()


def test_full_stage8_journey(client, temp_db):
    # 1. GET /api/genome -> uninitialized
    resp = client.get("/api/genome")
    assert resp.status_code == 200
    assert resp.json()["initialized"] is False

    # 2. POST /api/genome/init -> success
    init_resp = client.post(
        "/api/genome/init",
        json={
            "designation": _DESIGNATION,
            "mission_core_vision": _MISSION,
            "snippets": _SNIPPETS,
        },
    )
    assert init_resp.status_code == 200
    assert init_resp.json()["status"] == "success"

    # 3. GET /api/genome -> initialized
    resp = client.get("/api/genome")
    assert resp.status_code == 200
    genome = resp.json()
    assert genome["initialized"] is True
    assert genome["designation"] == _DESIGNATION

    conn = _connect(temp_db)
    try:
        snippet_count = conn.execute(
            "SELECT COUNT(*) FROM brand_texts WHERE brand_id = 'user_brand' AND source_type = 'genome_snippet'"
        ).fetchone()[0]
        assert snippet_count == 7
        assert conn.execute(
            "SELECT COUNT(*) FROM brand_texts WHERE brand_id != 'user_brand'"
        ).fetchone()[0] == 450
    finally:
        conn.close()

    # 4. POST /api/consistency/score -> success -> history consistency count = 1
    consistency_resp = client.post(
        "/api/consistency/score",
        json={"text": "Precision and craftsmanship define every collection we release."},
    )
    assert consistency_resp.status_code == 200
    consistency_body = consistency_resp.json()
    assert 0 <= consistency_body["score_overall"] <= 100
    assert "feature_breakdown" in consistency_body
    assert "diagnostic_breakdown" in consistency_body

    # 5. GET /api/benchmark/brands -> real competitor list
    brands_resp = client.get("/api/benchmark/brands")
    assert brands_resp.status_code == 200
    brands = brands_resp.json()
    assert isinstance(brands, list)
    assert len(brands) == 10
    competitor_id = brands[0]["brand_id"]

    # 6. POST /api/benchmark/run -> success -> history benchmark count = 1
    benchmark_resp = client.post(
        "/api/benchmark/run",
        json={"competitor_brand_id": competitor_id, "metric": "tone"},
    )
    assert benchmark_resp.status_code == 200
    benchmark_body = benchmark_resp.json()
    assert benchmark_body["metric"] == "tone"

    # Checkpoint (before any further consistency calls): 1 consistency / 1 benchmark.
    counts = _history_counts(temp_db)
    assert counts.get("consistency") == 1
    assert counts.get("benchmark") == 1
    assert "rewrite" not in counts

    # 7. GET /api/analytics -> counters reflect consistency + benchmark
    analytics_resp = client.get("/api/analytics")
    assert analytics_resp.status_code == 200
    analytics_body = analytics_resp.json()
    assert analytics_body["history"]["counts"]["consistency"] == 1
    assert analytics_body["history"]["counts"]["benchmark"] == 1
    assert analytics_body["history"]["counts"]["total"] == 2
    assert analytics_body["pillars"]["names"]
    assert analytics_body["heatmap"]["brands"]

    # 8. Attempt Rewrite BEFORE index rebuild -> explicit stale/missing index error
    rewrite_before = client.post(
        "/api/rewrite",
        json={"text": "this watch is super cool and awesome, and pretty nice to wear."},
    )
    assert rewrite_before.status_code == 503
    assert rewrite_before.json()["detail"]["error"] in {"index_missing", "user_grounding_not_indexed"}

    # 9. POST /api/rebuild/index -> success
    rebuild_index_resp = client.post("/api/rebuild/index")
    assert rebuild_index_resp.status_code == 200
    assert rebuild_index_resp.json()["status"] == "ok"

    # 10. POST /api/rewrite -> mocked provider success, real user RAG chunks, before/after score
    rewrite_resp = client.post(
        "/api/rewrite",
        json={"text": "this watch is super cool and awesome, and pretty nice to wear."},
    )
    assert rewrite_resp.status_code == 200
    rewrite_body = rewrite_resp.json()
    assert rewrite_body["provider"]["name"] == "openai"
    assert rewrite_body["rewritten_text"]
    assert rewrite_body["grounding_chunks"]
    assert all(chunk["chunk_id"] for chunk in rewrite_body["grounding_chunks"])
    assert 0 <= rewrite_body["score_before"] <= 100
    assert 0 <= rewrite_body["score_after"] <= 100

    counts = _history_counts(temp_db)
    assert counts.get("rewrite") == 1

    # 11. GET /api/analytics -> counters now reflect consistency=1, benchmark=1, rewrite=1, total=3
    analytics_resp_2 = client.get("/api/analytics")
    assert analytics_resp_2.status_code == 200
    counts_2 = analytics_resp_2.json()["history"]["counts"]
    assert counts_2 == {"consistency": 1, "benchmark": 1, "rewrite": 1, "total": 3}

    # 12. Score trend contains real analysis entries (benchmark events carry
    # no pre/post score, so they are legitimately excluded from the trend).
    score_trend = analytics_resp_2.json()["history"]["score_trend"]
    assert len(score_trend) == 2
    assert {entry["event_type"] for entry in score_trend} == {"consistency", "rewrite"}

    # 13. POST /api/rebuild/profile -> success
    rebuild_profile_resp = client.post("/api/rebuild/profile")
    assert rebuild_profile_resp.status_code == 200
    assert rebuild_profile_resp.json()["status"] == "ok"

    # 14. POST /api/rebuild/chunks -> success, RAG rebuilt
    rebuild_chunks_resp = client.post("/api/rebuild/chunks")
    assert rebuild_chunks_resp.status_code == 200
    rebuild_chunks_body = rebuild_chunks_resp.json()
    assert rebuild_chunks_body["status"] == "ok"
    assert rebuild_chunks_body["index_rebuilt"] is True

    # 15. User RAG retrieval still succeeds
    retrieval_resp = client.post(
        "/api/rag/retrieve",
        json={"text": "precision and craftsmanship", "brand_id": "user_brand", "top_k": 5},
    )
    assert retrieval_resp.status_code == 200
    assert retrieval_resp.json()["results"]

    # Rebuilds are not analysis events.
    counts_before_final_consistency = _history_counts(temp_db)
    assert counts_before_final_consistency == {"consistency": 1, "benchmark": 1, "rewrite": 1}

    # 16. Consistency still succeeds (this legitimately adds a 2nd consistency event).
    final_consistency = client.post(
        "/api/consistency/score",
        json={"text": "Precision and craftsmanship define every collection we release."},
    )
    assert final_consistency.status_code == 200

    final_counts = _history_counts(temp_db)
    assert final_counts["consistency"] == 2
    assert final_counts["benchmark"] == 1
    assert final_counts["rewrite"] == 1

    # Safety: checked-in canonical DB is never touched by this test.
    assert temp_db != CANONICAL_DB
