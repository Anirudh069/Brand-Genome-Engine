from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DB = REPO_ROOT / "data" / "brand_data.db"


def _genome_payload(designation: str, mission: str, snippet_prefix: str):
    return {
        "designation": designation,
        "mission_core_vision": mission,
        "snippets": [
            f"{snippet_prefix} 1: Precision and craftsmanship guide every line.",
            f"{snippet_prefix} 2: The tone stays measured, confident, and refined.",
            f"{snippet_prefix} 3: We speak with clarity, heritage, and discipline.",
            f"{snippet_prefix} 4: Each message should feel exacting and premium.",
            f"{snippet_prefix} 5: The voice remains consistent across all touchpoints.",
            f"{snippet_prefix} 6: Luxury copy should still feel human and readable.",
            f"{snippet_prefix} 7: We value detail, restraint, and unmistakable identity.",
        ],
    }


@pytest.fixture(autouse=True)
def _fake_genome_embedding(monkeypatch):
    def fake_embedding(text, model_name="all-MiniLM-L6-v2"):
        seed = sum(ord(char) for char in str(text))
        vector = [float((seed + index) % 97) / 97.0 for index in range(384)]
        return vector, model_name

    monkeypatch.setattr("src.api.genome_service.get_embedding", fake_embedding)


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    shutil.copy2(CANONICAL_DB, db_path)
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    return db_path


@pytest.fixture()
def db_conn(temp_db):
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def api_client(temp_db):
    with TestClient(app) as client:
        yield client


def _init_genome(api_client, designation="StageThreeBrand", mission="Deliver measured, disciplined, premium copy.", prefix="StageThree"):
    response = api_client.post("/api/genome/init", json=_genome_payload(designation, mission, prefix))
    assert response.status_code == 200
    return response.json()["profile"]


def _benchmark_payload(competitor_brand_id="omega", metric="tone"):
    return {"competitor_brand_id": competitor_brand_id, "metric": metric}


class TestBenchmarkBrands:
    def test_returns_real_competitors(self, api_client, db_conn):
        before = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        response = api_client.get("/api/benchmark/brands")
        assert response.status_code == 200

        brands = response.json()
        assert len(brands) == 10
        assert all("brand_id" in item and "designation" in item for item in brands)
        assert all(item["brand_id"] != "0" for item in brands)

        after = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        assert after == before

    def test_is_database_derived(self, api_client, db_conn):
        db_conn.execute("DELETE FROM brand_profiles WHERE brand_id = ?", ("omega",))
        db_conn.commit()

        response = api_client.get("/api/benchmark/brands")
        assert response.status_code == 200
        brands = response.json()
        assert len(brands) == 9
        assert all(item["brand_id"] != "omega" for item in brands)


class TestBenchmarkValidation:
    def test_missing_genome_blocks_run(self, api_client, db_conn):
        before = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        response = api_client.post("/api/benchmark/run", json=_benchmark_payload())
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "genome_not_initialized"
        after = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        assert after == before

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"competitor_brand_id": "omega"},
            {"metric": "tone"},
            {"competitor_brand_id": None, "metric": "tone"},
            {"competitor_brand_id": "omega", "metric": "keyword_overlap"},
        ],
    )
    def test_invalid_payloads_rejected(self, api_client, payload):
        response = api_client.post("/api/benchmark/run", json=payload)
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "competitor_brand_id",
        ["not_real", "0", "user_brand"],
    )
    def test_invalid_competitors_rejected(self, api_client, competitor_brand_id):
        _init_genome(api_client)
        response = api_client.post("/api/benchmark/run", json=_benchmark_payload(competitor_brand_id=competitor_brand_id))
        assert response.status_code in {400, 404}


class TestBenchmarkMetrics:
    def test_tone_benchmark_succeeds_and_logs_history(self, api_client, db_conn):
        profile = _init_genome(api_client, designation="ToneBrand", mission="We speak with measured confidence and precision.", prefix="Tone")
        before = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]

        response = api_client.post("/api/benchmark/run", json=_benchmark_payload(metric="tone"))
        assert response.status_code == 200
        body = response.json()

        assert body["metric"] == "tone"
        assert body["labels"] == ["Formality", "Sentence Length", "Vocabulary Richness"]
        assert len(body["user_series"]) == len(body["competitor_series"]) == len(body["labels"])
        assert all(isinstance(value, (int, float)) for value in body["user_series"] + body["competitor_series"])
        assert all(value == value and value not in (float("inf"), float("-inf")) for value in body["user_series"] + body["competitor_series"])
        assert body["user_brand"]["designation"] == profile["designation"]

        after = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        assert after == before + 1
        row = db_conn.execute(
            "SELECT brand_id, event_type, extra_json FROM analysis_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["brand_id"] == 0
        assert row["event_type"] == "benchmark"
        extra = json.loads(row["extra_json"])
        assert extra["competitor_brand_id"] == "omega"
        assert extra["metric"] == "tone"

    @pytest.mark.parametrize("metric", ["sentiment", "readability"])
    def test_other_metrics_return_aligned_finite_series(self, api_client, metric):
        _init_genome(api_client, designation="MetricBrand", mission="Precise, disciplined, and readable copy.", prefix="Metric")
        response = api_client.post("/api/benchmark/run", json=_benchmark_payload(metric=metric))
        assert response.status_code == 200
        body = response.json()
        assert body["metric"] == metric
        assert len(body["user_series"]) == len(body["competitor_series"]) == len(body["labels"])
        assert all(value == value and value not in (float("inf"), float("-inf")) for value in body["user_series"] + body["competitor_series"])


class TestBenchmarkDeterminism:
    def test_same_input_is_deterministic(self, api_client):
        _init_genome(api_client, designation="DeterministicBrand", mission="Measured, disciplined, and premium copy.", prefix="Deterministic")
        first = api_client.post("/api/benchmark/run", json=_benchmark_payload(metric="sentiment")).json()
        second = api_client.post("/api/benchmark/run", json=_benchmark_payload(metric="sentiment")).json()
        assert first["labels"] == second["labels"]
        assert first["user_series"] == second["user_series"]
        assert first["competitor_series"] == second["competitor_series"]

    def test_reinitializing_user_changes_user_series_not_competitor_series(self, api_client):
        _init_genome(api_client, designation="FirstGenome", mission="We write with restraint and precision.", prefix="First")
        first = api_client.post("/api/benchmark/run", json=_benchmark_payload(metric="tone")).json()
        _init_genome(api_client, designation="SecondGenome", mission="We write with warmth, confidence, and clarity.", prefix="Second")
        second = api_client.post("/api/benchmark/run", json=_benchmark_payload(metric="tone")).json()

        assert first["competitor_series"] == second["competitor_series"]
        assert first["user_series"] != second["user_series"]
