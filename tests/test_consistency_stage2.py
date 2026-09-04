from __future__ import annotations

import json
import math
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DB = REPO_ROOT / "data" / "brand_data.db"
TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="brand_genome_consistency_tests_"))
TEST_DB_PATH = TEST_DB_DIR / "brand_data.db"
shutil.copy2(CANONICAL_DB, TEST_DB_PATH)

from src.api.main import app  # noqa: E402
from src.api.genome_service import load_active_user_genome  # noqa: E402
from src.scoring.consistency import compute_consistency_score, score_against_user_genome  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    def fake_embedding(text, model_name="all-MiniLM-L6-v2"):
        seed = sum(ord(char) for char in str(text))
        vector = [float((seed + index) % 97) / 97.0 for index in range(384)]
        return vector, model_name

    monkeypatch.setattr("src.api.genome_service.get_embedding", fake_embedding)
    monkeypatch.setattr("src.scoring.consistency.get_embedding", fake_embedding)


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


def genome_payload(designation: str, mission: str, snippet_prefix: str):
    return {
        "designation": designation,
        "mission_core_vision": mission,
        "snippets": [
            f"{snippet_prefix} 1: Precision craftsmanship defines our standard.",
            f"{snippet_prefix} 2: The voice remains measured and assured.",
            f"{snippet_prefix} 3: We communicate with timeless clarity.",
            f"{snippet_prefix} 4: Every message balances heritage and confidence.",
            f"{snippet_prefix} 5: Our copy values detail, restraint, and elegance.",
            f"{snippet_prefix} 6: We speak to discerning audiences with care.",
            f"{snippet_prefix} 7: Consistency across channels is part of the craft.",
        ],
    }


ON_BRAND_TEXT = (
    "Precision craftsmanship and timeless elegance define the way we speak, "
    "balancing confidence with restraint in every detail."
)
OFF_BRAND_TEXT = (
    "This product is super cool and easy to use, with casual vibes and a "
    "playful tone that feels light and everyday."
)


class TestConsistencyStage2:
    def test_missing_genome_blocks_scoring_and_writes_no_history(self, api_client, db_conn):
        before = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        response = api_client.post("/api/consistency/score", json={"text": ON_BRAND_TEXT})
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["error"] == "genome_not_initialized"
        after = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        assert after == before

    def test_blank_text_is_rejected(self, api_client):
        response = api_client.post("/api/consistency/score", json={"text": "   "})
        assert response.status_code == 422

    def test_extra_target_fields_are_rejected(self, api_client):
        response = api_client.post("/api/consistency/score", json={"text": ON_BRAND_TEXT, "brand_id": "rolex"})
        assert response.status_code == 422

    def test_legacy_route_rejects_competitor_fields(self, api_client):
        response = api_client.post("/api/check-consistency", json={"text": ON_BRAND_TEXT, "brand_id": "rolex"})
        assert response.status_code == 422

    def test_legacy_route_matches_canonical_semantics(self, api_client, db_conn):
        api_client.post(
            "/api/genome/init",
            json=genome_payload(
                designation="StageTwoBrand",
                mission="Deliver precise, enduring, and disciplined brand language.",
                snippet_prefix="StageTwo",
            ),
        )
        canonical = api_client.post("/api/consistency/score", json={"text": ON_BRAND_TEXT})
        legacy = api_client.post("/api/check-consistency", json={"text": ON_BRAND_TEXT})

        assert canonical.status_code == 200
        assert legacy.status_code == 200
        canonical_body = canonical.json()
        legacy_body = legacy.json()

        assert legacy_body["score_overall"] == pytest.approx(canonical_body["score_overall"], abs=0.1)
        assert legacy_body["feature_breakdown"] == canonical_body["feature_breakdown"]
        assert legacy_body["diagnostic_breakdown"] == canonical_body["diagnostic_breakdown"]

        history_rows = db_conn.execute("SELECT COUNT(*) FROM analysis_history WHERE event_type = 'consistency'").fetchone()[0]
        assert history_rows >= 2

    def test_successful_score_returns_expected_contract(self, api_client, db_conn):
        init_response = api_client.post(
            "/api/genome/init",
            json=genome_payload(
                designation="StageTwoBrand",
                mission="Deliver precise, enduring, and disciplined brand language.",
                snippet_prefix="StageTwo",
            ),
        )
        assert init_response.status_code == 200

        before = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        response = api_client.post("/api/consistency/score", json={"text": ON_BRAND_TEXT})
        assert response.status_code == 200
        body = response.json()

        assert "score_overall" in body
        assert "feature_breakdown" in body
        assert "diagnostic_breakdown" in body
        assert "brand_name_mentions" in body
        assert "timestamp" in body
        assert math.isfinite(body["score_overall"])
        assert 0.0 <= body["score_overall"] <= 100.0
        assert isinstance(body["feature_breakdown"], dict)
        assert set(body["feature_breakdown"].keys()) >= {"tone", "sentiment", "readability", "keywords", "embedding_similarity"}
        assert isinstance(body["diagnostic_breakdown"], list)
        assert body["brand_name_mentions"]["designation"] == "StageTwoBrand"
        assert body["brand_name_mentions"]["count"] == 0

        after = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        assert after == before + 1

    def test_history_row_is_structured_and_canonical(self, api_client, db_conn):
        api_client.post(
            "/api/genome/init",
            json=genome_payload(
                designation="StageTwoBrand",
                mission="Deliver precise, enduring, and disciplined brand language.",
                snippet_prefix="StageTwo",
            ),
        )
        response = api_client.post("/api/consistency/score", json={"text": ON_BRAND_TEXT})
        assert response.status_code == 200
        score = response.json()

        row = db_conn.execute(
            "SELECT brand_id, event_type, input_text, pre_score, post_score, diagnostics_json, extra_json FROM analysis_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row["brand_id"] == 0
        assert row["event_type"] == "consistency"
        assert row["input_text"] == ON_BRAND_TEXT
        assert row["post_score"] is None
        assert math.isclose(float(row["pre_score"]), float(score["score_overall"]), rel_tol=0.0, abs_tol=0.2)
        assert json.loads(row["diagnostics_json"]) == score["diagnostic_breakdown"]

        extra = json.loads(row["extra_json"])
        assert "feature_breakdown" in extra
        assert "brand_name_mentions" in extra
        assert extra["brand_name_mentions"]["designation"] == "StageTwoBrand"
        assert extra["genome_version"] == score["genome_version"]

    def test_reinitialized_genome_is_used_on_next_request(self, api_client):
        first = genome_payload(
            designation="AlphaGenome",
            mission="We speak with precise, formal, and restrained confidence.",
            snippet_prefix="Alpha",
        )
        second = genome_payload(
            designation="BetaGenome",
            mission="We sound warmer, more conversational, and lightly playful.",
            snippet_prefix="Beta",
        )
        assert api_client.post("/api/genome/init", json=first).status_code == 200
        score_a = api_client.post("/api/consistency/score", json={"text": ON_BRAND_TEXT}).json()
        assert api_client.post("/api/genome/init", json=second).status_code == 200
        score_b = api_client.post("/api/consistency/score", json={"text": ON_BRAND_TEXT}).json()

        assert score_a["brand_name_mentions"]["designation"] == "AlphaGenome"
        assert score_b["brand_name_mentions"]["designation"] == "BetaGenome"
        assert score_a["score_overall"] != score_b["score_overall"]

    def test_brand_mentions_are_neutral_to_score(self, api_client):
        api_client.post(
            "/api/genome/init",
            json=genome_payload(
                designation="NeutralBrand",
                mission="Deliver precise and disciplined copy.",
                snippet_prefix="Neutral",
            ),
        )

        def constant_sentiment(_text):
            return 0.61

        def constant_formality(_text):
            return 0.74

        def constant_readability(_text):
            return (52.0, 14.0)

        def constant_vocab_metrics(_text):
            return {"vocab_diversity": 0.55, "avg_sentence_length": 14.0, "punctuation_density": 0.06}

        def constant_embedding(_text, model_name="all-MiniLM-L6-v2"):
            return ([0.01] * 384, model_name)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("src.scoring.consistency.extract_sentiment", constant_sentiment)
        monkeypatch.setattr("src.scoring.consistency.extract_formality", constant_formality)
        monkeypatch.setattr("src.scoring.consistency.extract_readability", constant_readability)
        monkeypatch.setattr("src.scoring.consistency.extract_vocab_metrics", constant_vocab_metrics)
        monkeypatch.setattr("src.scoring.consistency.get_embedding", constant_embedding)

        base_text = "Precision craftsmanship and timeless elegance shape the message with restraint."
        with_name = "Precision craftsmanship and timeless elegance shape the message with NeutralBrand restraint."

        try:
            base = api_client.post("/api/consistency/score", json={"text": base_text}).json()
            marked = api_client.post("/api/consistency/score", json={"text": with_name}).json()
        finally:
            monkeypatch.undo()

        assert marked["brand_name_mentions"]["count"] == 1
        assert base["brand_name_mentions"]["count"] == 0
        assert marked["score_overall"] == pytest.approx(base["score_overall"], abs=0.2)

    def test_core_scoring_service_is_callable_independently(self, api_client, db_conn):
        api_client.post(
            "/api/genome/init",
            json=genome_payload(
                designation="StandaloneBrand",
                mission="Deliver precise, enduring, and disciplined brand language.",
                snippet_prefix="Standalone",
            ),
        )
        genome = load_active_user_genome(db_conn)
        assert genome is not None

        rich = score_against_user_genome(ON_BRAND_TEXT, genome)
        legacy = compute_consistency_score(ON_BRAND_TEXT, genome)

        assert rich["score_overall"] >= 0.0
        assert set(rich["feature_breakdown"].keys()) >= {"tone", "sentiment", "readability", "keywords", "embedding_similarity"}
        assert set(legacy.keys()) == {"overall_score", "tone_pct", "vocab_overlap_pct", "sentiment_alignment_pct", "readability_match_pct"}
        assert 0.0 <= legacy["overall_score"] <= 100.0

    def test_off_brand_scores_differ_from_on_brand(self, api_client):
        api_client.post(
            "/api/genome/init",
            json=genome_payload(
                designation="DifferenceBrand",
                mission="Deliver precise, enduring, and disciplined brand language.",
                snippet_prefix="Difference",
            ),
        )
        on_brand = api_client.post("/api/consistency/score", json={"text": ON_BRAND_TEXT}).json()
        off_brand = api_client.post("/api/consistency/score", json={"text": OFF_BRAND_TEXT}).json()

        assert on_brand["score_overall"] != off_brand["score_overall"]
        assert on_brand["diagnostic_breakdown"] != off_brand["diagnostic_breakdown"]

    def test_legacy_route_writes_exactly_one_history_row(self, api_client, db_conn):
        api_client.post(
            "/api/genome/init",
            json=genome_payload(
                designation="HistoryBrand",
                mission="Deliver precise, enduring, and disciplined brand language.",
                snippet_prefix="History",
            ),
        )
        before = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        response = api_client.post("/api/check-consistency", json={"text": ON_BRAND_TEXT})
        assert response.status_code == 200
        after = db_conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        assert after == before + 1
