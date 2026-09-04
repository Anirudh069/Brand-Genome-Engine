from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.genome_service import ensure_canonical_schema, write_history_event
from src.api.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DB = REPO_ROOT / "data" / "brand_data.db"


def _valid_payload(
    designation: str = "StageOneBrand",
    mission_core_vision: str = "We deliver precise, timeless brand language.",
    snippet_prefix: str = "Snippet",
):
    return {
        "designation": designation,
        "mission_core_vision": mission_core_vision,
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
    yield db_path


@pytest.fixture()
def api_client(temp_db):
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def db_conn(temp_db):
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def bootstrap_snapshot(db_conn):
    before = {
        "brand_texts": db_conn.execute("SELECT COUNT(*) FROM brand_texts").fetchone()[0],
        "brand_texts_raw": db_conn.execute("SELECT COUNT(*) FROM brand_texts_raw").fetchone()[0],
        "brand_chunks": db_conn.execute("SELECT COUNT(*) FROM brand_chunks").fetchone()[0],
    }
    ensure_canonical_schema(db_conn)
    ensure_canonical_schema(db_conn)
    after = {
        "brand_texts": db_conn.execute("SELECT COUNT(*) FROM brand_texts").fetchone()[0],
        "brand_texts_raw": db_conn.execute("SELECT COUNT(*) FROM brand_texts_raw").fetchone()[0],
        "brand_chunks": db_conn.execute("SELECT COUNT(*) FROM brand_chunks").fetchone()[0],
    }
    return before, after


class TestSchemaBootstrap:
    def test_bootstrap_is_idempotent(self, bootstrap_snapshot, db_conn):
        before, after = bootstrap_snapshot
        assert before == after
        assert db_conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db_conn.execute("SELECT COUNT(*) FROM brand_profiles").fetchone()[0] == 10
        assert db_conn.execute("SELECT COUNT(*) FROM brand_profile").fetchone()[0] == 0

    def test_runtime_db_path_is_canonical(self):
        assert (REPO_ROOT / "data" / "brand_data.db").exists()
        assert not (REPO_ROOT / "brand_data.db").exists()

    def test_competitor_counts_stable(self, bootstrap_snapshot):
        before, after = bootstrap_snapshot
        assert before["brand_texts"] == 450
        assert before["brand_texts_raw"] == 450
        assert before["brand_chunks"] == 657
        assert before == after


class TestGenomeValidation:
    def test_exactly_seven_valid_snippets_accepted(self, api_client):
        response = api_client.post("/api/genome/init", json=_valid_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["profile"]["initialized"] is True

    @pytest.mark.parametrize(
        "snippet_count",
        [6, 8],
    )
    def test_invalid_snippet_count_rejected(self, api_client, snippet_count):
        payload = _valid_payload()
        payload["snippets"] = payload["snippets"][:snippet_count]
        if snippet_count > 7:
            payload["snippets"].append("Extra snippet that should not be accepted.")
        response = api_client.post("/api/genome/init", json=payload)
        assert response.status_code == 422

    def test_blank_snippet_rejected(self, api_client):
        payload = _valid_payload()
        payload["snippets"][3] = "   "
        response = api_client.post("/api/genome/init", json=payload)
        assert response.status_code == 422

    def test_blank_designation_rejected(self, api_client):
        payload = _valid_payload(designation="   ")
        response = api_client.post("/api/genome/init", json=payload)
        assert response.status_code == 422

    def test_blank_mission_rejected(self, api_client):
        payload = _valid_payload(mission_core_vision="   ")
        response = api_client.post("/api/genome/init", json=payload)
        assert response.status_code == 422


class TestGenomePersistence:
    def test_no_genome_means_not_initialised(self, api_client):
        response = api_client.get("/api/genome")
        assert response.status_code == 200
        body = response.json()
        assert body["initialized"] is False
        assert body["snippetsCount"] == 0

    def test_init_persists_summary(self, api_client, db_conn):
        payload = _valid_payload(designation="PersistedBrand")
        response = api_client.post("/api/genome/init", json=payload)
        assert response.status_code == 200

        user_row = db_conn.execute(
            "SELECT id, designation, mission_core_vision, created_at, updated_at FROM brands WHERE id = 0"
        ).fetchone()
        assert user_row is not None
        assert user_row["id"] == 0
        assert user_row["designation"] == payload["designation"]
        assert user_row["mission_core_vision"] == payload["mission_core_vision"]
        assert user_row["created_at"] is not None
        assert user_row["updated_at"] is not None

        genome = api_client.get("/api/genome").json()
        assert genome["initialized"] is True
        assert genome["designation"] == "PersistedBrand"
        assert genome["mission_core_vision"] == payload["mission_core_vision"]
        assert genome["snippetsCount"] == 7
        assert genome["snippets"] == payload["snippets"]
        assert genome["keywords"]
        assert genome["tone_features"]
        assert genome["feature_ready"] is True
        assert genome["embedding_ready"] is True

        row = db_conn.execute(
            "SELECT keywords_json, tone_features_json, aggregate_embedding, metadata_json FROM brand_profile WHERE brand_id = 0"
        ).fetchone()
        assert row is not None
        assert len(json.loads(row["aggregate_embedding"])) == 384
        metadata = json.loads(row["metadata_json"])
        assert metadata["snippets"] == payload["snippets"]
        assert len(metadata["sample_embeddings"]) == 8
        assert all(len(vec) == 384 for vec in metadata["sample_embeddings"])
        assert json.loads(row["keywords_json"])
        assert json.loads(row["tone_features_json"])

    def test_reinit_updates_same_user_row(self, api_client, db_conn):
        api_client.post("/api/genome/init", json=_valid_payload(designation="Alpha"))
        api_client.post("/api/genome/init", json=_valid_payload(designation="Beta"))

        user_rows = db_conn.execute("SELECT id, designation, mission_core_vision FROM brands WHERE id = 0").fetchall()
        assert len(user_rows) == 1
        assert user_rows[0]["designation"] == "Beta"
        assert db_conn.execute("SELECT COUNT(*) FROM brand_profile WHERE brand_id = 0").fetchone()[0] == 1
        assert db_conn.execute("SELECT COUNT(*) FROM brand_profiles").fetchone()[0] == 10

    def test_competitor_rows_unchanged(self, api_client, db_conn):
        before = db_conn.execute("SELECT id, designation FROM brands WHERE id != 0 ORDER BY id").fetchall()
        before_profiles = db_conn.execute("SELECT brand_id, brand_name FROM brand_profiles ORDER BY brand_id").fetchall()
        api_client.post("/api/genome/init", json=_valid_payload(designation="StableUser"))
        after = db_conn.execute("SELECT id, designation FROM brands WHERE id != 0 ORDER BY id").fetchall()
        after_profiles = db_conn.execute("SELECT brand_id, brand_name FROM brand_profiles ORDER BY brand_id").fetchall()
        assert before == after
        assert before_profiles == after_profiles

    def test_restart_from_db_state(self, api_client, db_conn):
        payload = _valid_payload(designation="ReloadBrand")
        api_client.post("/api/genome/init", json=payload)
        reopened = api_client.get("/api/genome").json()
        assert reopened["designation"] == "ReloadBrand"
        assert reopened["snippetsCount"] == 7
        assert db_conn.execute("SELECT COUNT(*) FROM brand_profile WHERE brand_id = 0").fetchone()[0] == 1


class TestGenomeEmbeddings:
    def test_aggregate_embedding_is_deterministic(self, api_client, monkeypatch, db_conn):
        def fake_embedding(text, model_name="all-MiniLM-L6-v2"):
            seed = sum(ord(char) for char in str(text))
            vector = [float((seed + index) % 97) / 97.0 for index in range(384)]
            return vector, model_name

        monkeypatch.setattr("src.api.genome_service.get_embedding", fake_embedding)
        payload = _valid_payload(designation="VectorBrand")

        first = api_client.post("/api/genome/init", json=payload)
        assert first.status_code == 200
        first_row = db_conn.execute(
            "SELECT aggregate_embedding, metadata_json FROM brand_profile WHERE brand_id = 0"
        ).fetchone()
        first_embedding = json.loads(first_row["aggregate_embedding"])
        first_metadata = json.loads(first_row["metadata_json"])

        second = api_client.post("/api/genome/init", json=payload)
        assert second.status_code == 200
        second_row = db_conn.execute(
            "SELECT aggregate_embedding, metadata_json FROM brand_profile WHERE brand_id = 0"
        ).fetchone()
        second_embedding = json.loads(second_row["aggregate_embedding"])
        second_metadata = json.loads(second_row["metadata_json"])

        assert first_embedding == second_embedding
        assert first_metadata["sample_embeddings"] == second_metadata["sample_embeddings"]
        assert len(first_embedding) == 384
        assert all(isinstance(value, float) for value in first_embedding)


class TestGenomeReinitialisation:
    def test_reinitialisation_replaces_active_user_genome(self, api_client, db_conn):
        payload_a = _valid_payload(designation="GenomeA", snippet_prefix="A")
        payload_b = _valid_payload(designation="GenomeB", snippet_prefix="B")

        api_client.post("/api/genome/init", json=payload_a)
        db_conn.execute(
            "INSERT INTO analysis_history (brand_id, analysis_type, result_json, created_at) VALUES (0, 'consistency', ?, datetime('now'))",
            (json.dumps({"before": 73.2}),),
        )
        db_conn.commit()
        history_before = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]

        api_client.post("/api/genome/init", json=payload_b)

        genome = api_client.get("/api/genome").json()
        assert genome["designation"] == "GenomeB"
        assert genome["snippets"] == payload_b["snippets"]
        assert db_conn.execute("SELECT COUNT(*) FROM brand_profile WHERE brand_id = 0").fetchone()[0] == 1
        assert db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0] == history_before
        assert db_conn.execute("SELECT COUNT(*) FROM brand_profiles").fetchone()[0] == 10


class TestHistoryContract:
    def test_canonical_history_writer_round_trips(self, db_conn):
        write_history_event(
            db_conn,
            brand_id=0,
            event_type="rewrite",
            input_text="Hello world",
            pre_score={"overall_score": 41.2},
            post_score={"overall_score": 61.7},
            diagnostics_json=["d1", "d2"],
            extra_json={"note": "round-trip"},
        )

        row = db_conn.execute(
            "SELECT brand_id, event_type, input_text, pre_score, post_score, diagnostics_json, extra_json, analysis_type, result_json FROM analysis_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["brand_id"] == 0
        assert row["event_type"] == "rewrite"
        assert row["input_text"] == "Hello world"
        assert json.loads(row["pre_score"])["overall_score"] == 41.2
        assert json.loads(row["post_score"])["overall_score"] == 61.7
        assert json.loads(row["diagnostics_json"]) == ["d1", "d2"]
        assert json.loads(row["extra_json"])["note"] == "round-trip"
        assert row["analysis_type"] == "rewrite"
        legacy = json.loads(row["result_json"])
        assert legacy["event_type"] == "rewrite"

    @pytest.mark.parametrize("event_type", ["consistency", "benchmark", "rewrite"])
    def test_future_event_types_accepted(self, db_conn, event_type):
        write_history_event(
            db_conn,
            brand_id=1,
            event_type=event_type,
            input_text="Copy",
            diagnostics_json=[],
            extra_json={},
        )
        row = db_conn.execute(
            "SELECT event_type FROM analysis_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["event_type"] == event_type

    def test_reinit_does_not_delete_history(self, api_client, db_conn):
        write_history_event(db_conn, brand_id=0, event_type="rewrite", input_text="x")
        before = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        api_client.post("/api/genome/init", json=_valid_payload(designation="HistorySafe"))
        after = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        assert before == after


class TestGenomeApiCompatibility:
    def test_genome_endpoint_and_competitor_profile_do_not_conflict(self, api_client):
        genome_response = api_client.get("/api/genome")
        assert genome_response.status_code == 200

        competitor_response = api_client.post(
            "/api/check-consistency",
            json={"text": "Precision craftsmanship and timeless excellence define the watch.", "brand_id": "rolex"},
        )
        assert competitor_response.status_code == 422
