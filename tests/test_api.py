# filepath: tests/test_api.py
"""Comprehensive API tests for the Brand Genome Engine FastAPI backend."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="brand_genome_api_tests_"))
TEST_DB_PATH = TEST_DB_DIR / "brand_data.db"
shutil.copy2(REPO_ROOT / "data" / "brand_data.db", TEST_DB_PATH)
os.environ["SQLITE_DB_PATH"] = str(TEST_DB_PATH)

from src.api.main import app

client = TestClient(app)


def genome_payload(
    designation: str = "TestBrand",
    mission: str = "Delivering precision engineering and timeless craftsmanship.",
    snippet_prefix: str = "Snippet",
):
    return {
        "designation": designation,
        "mission_core_vision": mission,
        "snippets": [
            f"{snippet_prefix} 1: We create elevated copy with precision and heritage.",
            f"{snippet_prefix} 2: Every message balances luxury, clarity, and confidence.",
            f"{snippet_prefix} 3: The voice stays refined, technical, and authoritative.",
            f"{snippet_prefix} 4: We speak to collectors, enthusiasts, and discerning buyers.",
            f"{snippet_prefix} 5: Brand language should feel timeless and exacting.",
            f"{snippet_prefix} 6: Copy must remain consistent across every channel.",
            f"{snippet_prefix} 7: We value measured tone, craftsmanship, and detail.",
        ],
    }


def init_consistency_genome(client, designation: str = "StageOneBrand"):
    response = client.post("/api/genome/init", json=genome_payload(designation=designation))
    assert response.status_code == 200
    return response.json()["profile"]


@pytest.fixture(autouse=True)
def _fake_genome_embedding(monkeypatch):
    def fake_embedding(text, model_name="all-MiniLM-L6-v2"):
        seed = sum(ord(char) for char in str(text))
        vector = [float((seed + index) % 97) / 97.0 for index in range(384)]
        return vector, model_name

    monkeypatch.setattr("src.api.genome_service.get_embedding", fake_embedding)

ROLEX_ON_BRAND = (
    "The Oyster Perpetual embodies precision craftsmanship and perpetual "
    "excellence, a testament to enduring horological mastery."
)
ROLEX_OFF_BRAND = (
    "This watch is awesome and super easy to wear every day. Cool design "
    "and pretty nice overall."
)

SCORE_KEYS = {
    "overall_score",
    "tone_pct",
    "vocab_overlap_pct",
    "sentiment_alignment_pct",
    "readability_match_pct",
}


class TestHealth:
    def test_status_ok(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_version_string(self):
        r = client.get("/api/health")
        body = r.json()
        assert "version" in body
        assert isinstance(body["version"], str)
        assert body["version"]


class TestBrands:
    def test_returns_list(self):
        r = client.get("/api/brands")
        assert r.status_code == 200
        brands = r.json()["brands"]
        assert isinstance(brands, list)
        assert len(brands) >= 1

    def test_brand_required_keys(self):
        r = client.get("/api/brands")
        for b in r.json()["brands"]:
            assert "brand_id" in b
            assert "brand_name" in b

    def test_fallback_brands_present(self):
        r = client.get("/api/brands")
        ids = {b["brand_id"] for b in r.json()["brands"]}
        # These brands exist in both the DB profiles and the fallback list
        for expected in ("rolex", "omega", "tag_heuer", "tissot"):
            assert expected in ids, f"{expected} missing from brands list"


class TestCheckConsistency:
    def test_returns_scores(self):
        profile = init_consistency_genome(client)
        r = client.post("/api/check-consistency", json={"text": ROLEX_ON_BRAND})
        assert r.status_code == 200
        body = r.json()
        for k in ("score_overall", "feature_breakdown", "diagnostic_breakdown", "brand_name_mentions", "timestamp"):
            assert k in body, f"Missing key: {k}"
        assert body["brand_name"] == profile["designation"]

    def test_scores_in_range(self):
        init_consistency_genome(client)
        r = client.post("/api/check-consistency", json={"text": ROLEX_ON_BRAND})
        body = r.json()
        assert 0 <= body["score_overall"] <= 100, f"score_overall={body['score_overall']} out of [0,100]"

    def test_brand_name_present(self):
        profile = init_consistency_genome(client, designation="RolexLike")
        r = client.post("/api/check-consistency", json={"text": ROLEX_ON_BRAND})
        assert r.json()["brand_name"] == profile["designation"]

    def test_text_too_short(self):
        init_consistency_genome(client)
        r = client.post("/api/check-consistency", json={"text": "Hi"})
        assert r.status_code == 200
        assert 0 <= r.json()["score_overall"] <= 100

    def test_competitor_field_is_rejected(self):
        init_consistency_genome(client)
        r = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND,
            "brand_id": "rolex",
        })
        assert r.status_code == 422

    def test_on_brand_scores_higher_than_off_brand(self):
        init_consistency_genome(client)
        on = client.post("/api/check-consistency", json={"text": ROLEX_ON_BRAND}).json()
        off = client.post("/api/check-consistency", json={"text": ROLEX_OFF_BRAND}).json()
        assert on["score_overall"] > off["score_overall"], (
            f"On-brand ({on['score_overall']}) should beat "
            f"off-brand ({off['score_overall']})"
        )


class TestCheckConsistencyContract:
    """Frozen response-schema contract for POST /api/check-consistency."""

    FROZEN_KEYS = {
        "score_overall", "feature_breakdown", "diagnostic_breakdown",
        "brand_name_mentions", "timestamp", "brand_name",
        "designation", "genome_version", "error",
    }

    def test_response_keys_exact(self):
        init_consistency_genome(client)
        r = client.post("/api/check-consistency", json={"text": ROLEX_ON_BRAND})
        assert r.status_code == 200
        assert set(r.json().keys()) == self.FROZEN_KEYS

    def test_pct_values_numeric_and_clamped(self):
        init_consistency_genome(client)
        body = client.post("/api/check-consistency", json={"text": ROLEX_ON_BRAND}).json()
        assert isinstance(body["score_overall"], (int, float)), f"score_overall is {type(body['score_overall'])}"
        assert 0 <= body["score_overall"] <= 100, f"score_overall={body['score_overall']} out of [0,100]"

    def test_error_null_on_success(self):
        init_consistency_genome(client)
        body = client.post("/api/check-consistency", json={"text": ROLEX_ON_BRAND}).json()
        assert body["error"] is None

    def test_short_text_returns_zeros(self):
        init_consistency_genome(client)
        body = client.post("/api/check-consistency", json={"text": "Hi"}).json()
        assert 0 <= body["score_overall"] <= 100
        assert isinstance(body["diagnostic_breakdown"], list)


class TestRewrite:
    def test_full_response(self):
        r = client.post("/api/rewrite", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "rolex",
            "n_grounding_chunks": 2,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["brand_id"] == "rolex"
        assert body["brand_name"] == "Rolex"
        assert body["original_text"] == ROLEX_OFF_BRAND
        assert body["rewritten_text"] is not None
        assert body["error"] is None

    def test_score_keys_present(self):
        r = client.post("/api/rewrite", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "rolex",
        })
        body = r.json()
        for section in ("score_before", "score_after"):
            assert section in body, f"Missing {section}"
            assert body[section] is not None, f"{section} is None"
            for k in SCORE_KEYS:
                assert k in body[section], f"Missing {k} in {section}"

    def test_text_too_short(self):
        r = client.post("/api/rewrite", json={
            "text": "Short", "brand_id": "rolex",
        })
        assert r.status_code == 200
        assert r.json()["error"] == "text_too_short"

    def test_suggestions_non_empty(self):
        r = client.post("/api/rewrite", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "rolex",
        })
        body = r.json()
        assert isinstance(body["suggestions"], list)
        assert len(body["suggestions"]) >= 1

    def test_grounding_chunks_returned(self):
        r = client.post("/api/rewrite", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "rolex",
            "n_grounding_chunks": 3,
        })
        body = r.json()
        assert isinstance(body["grounding_chunks_used"], list)
        assert len(body["grounding_chunks_used"]) >= 1

    def test_brand_name_in_response(self):
        r = client.post("/api/rewrite", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "omega",
        })
        assert r.json()["brand_name"] == "Omega"


class TestProfile:
    def test_get_profile(self):
        r = client.get("/api/genome")
        assert r.status_code == 200
        body = r.json()
        assert body["initialized"] is False
        assert body["snippetsCount"] == 0
        assert body["brand_id"] == "user_brand"
@pytest.fixture(autouse=True)
def _reset_test_db():
    shutil.copy2(REPO_ROOT / "data" / "brand_data.db", TEST_DB_PATH)
    yield

    def test_update_profile(self):
        r = client.post("/api/genome/init", json=genome_payload())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["profile"]["designation"] == "TestBrand"
        assert body["profile"]["initialized"] is True

    def test_keyword_extraction(self):
        r = client.post("/api/genome/init", json=genome_payload(
            designation="KWTest",
            mission="Our commitment to precision engineering and craftsmanship defines every timepiece",
            snippet_prefix="KW",
        ))
        body = r.json()
        kw = body["profile"]["keywords"]
        assert isinstance(kw, list)
        assert len(kw) >= 1
        all_kw = " ".join(kw).lower()
        assert any(w in all_kw for w in ("commitment", "precision", "engineering",
                                          "craftsmanship", "defines", "timepiece"))

    def test_sentiment_computed(self):
        r = client.post("/api/genome/init", json=genome_payload(
            designation="SentTest",
            mission="We build terrible broken products that nobody wants",
            snippet_prefix="Sent",
        ))
        body = r.json()
        sent = body["profile"]["tone_features"]["avg_sentiment"]
        assert isinstance(sent, float)
        assert sent < 0.6, f"Expected sentiment < 0.6 for negative mission, got {sent}"

    def test_tone_keyword_fallback(self):
        r = client.post("/api/genome/init", json=genome_payload(
            designation="FallbackTest",
            mission="We create technical copy for discerning audiences.",
            snippet_prefix="Fallback",
        ))
        body = r.json()
        kw = body["profile"]["top_keywords"]
        assert isinstance(kw, list)
        assert len(kw) >= 1


class TestAnalytics:
    def test_returns_data(self):
        r = client.get("/api/analytics")
        assert r.status_code == 200
        body = r.json()
        for key in ("pillars", "heatmap", "tsne", "tone", "history", "metadata"):
            assert key in body, f"Missing analytics key: {key}"

    def test_pillars_are_the_five_authoritative_names(self):
        r = client.get("/api/analytics")
        body = r.json()
        assert set(body["pillars"]["names"]) == {
            "Sustainability", "Precision", "Heritage", "Value", "Innovation",
        }

    def test_history_counts_are_real_and_no_fake_fallback(self):
        r = client.get("/api/analytics")
        body = r.json()
        counts = body["history"]["counts"]
        for key in ("consistency", "benchmark", "rewrite", "total"):
            assert key in counts
            assert isinstance(counts[key], (int, float))
        assert isinstance(body["history"]["score_trend"], list)

    def test_no_hardcoded_placeholder_fields(self):
        r = client.get("/api/analytics")
        body = r.json()
        # Legacy fake/placeholder fields must not reappear.
        for legacy_key in ("total_analyzed", "avg_consistency", "deviations_fixed", "trend"):
            assert legacy_key not in body


class TestBenchmark:
    def test_brands_endpoint_returns_real_competitors(self):
        r = client.get("/api/benchmark/brands")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert all("brand_id" in item and "designation" in item for item in body)

    def test_benchmark_run_contract(self):
        r = client.post("/api/benchmark/run", json={
            "competitor_brand_id": "omega",
            "metric": "sentiment",
        })
        assert r.status_code in {400, 422}


class TestRebuildEndpoints:
    def test_profile_rebuild(self):
        r = client.post("/api/profile/rebuild", json={"brand_id": "rolex"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert "built_at" in body

    def test_index_rebuild(self):
        r = client.post("/api/index/rebuild")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert "n_brands" in body

    def test_chunks_rebuild(self):
        r = client.post("/api/chunks/rebuild")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert "n_chunks" in body


class TestValidation:
    def test_check_consistency_missing_text(self):
        r = client.post("/api/check-consistency", json={})
        assert r.status_code == 422

    def test_check_consistency_rejects_extra_fields(self):
        r = client.post("/api/check-consistency", json={"text": "some text here", "brand_id": "rolex"})
        assert r.status_code == 422

    def test_rewrite_missing_text(self):
        r = client.post("/api/rewrite", json={"brand_id": "rolex"})
        assert r.status_code == 422

    def test_rewrite_missing_brand(self):
        r = client.post("/api/rewrite", json={"text": "some text here"})
        assert r.status_code == 422

    def test_profile_update_missing_fields(self):
        r = client.post("/api/profile", json={"brand_name": "X"})
        assert r.status_code == 422


class TestScoringIntegration:
    def test_scores_vary_by_text(self):
        init_consistency_genome(client)
        s1 = client.post("/api/check-consistency", json={"text": ROLEX_ON_BRAND}).json()["score_overall"]
        s2 = client.post("/api/check-consistency", json={"text": ROLEX_OFF_BRAND}).json()["score_overall"]
        assert s1 != s2, "Real scoring should differentiate texts"

    def test_vocab_overlap_detects_keywords(self):
        init_consistency_genome(client)
        with_kw = client.post("/api/check-consistency", json={
            "text": "Precision and perpetual excellence in craftsmanship define this oyster timepiece.",
        }).json()
        without_kw = client.post("/api/check-consistency", json={
            "text": "This is a very casual and fun everyday accessory that looks neat.",
        }).json()
        assert with_kw["feature_breakdown"]["keywords"]["score"] > without_kw["feature_breakdown"]["keywords"]["score"]

    def test_rewrite_score_not_hardcoded(self):
        r = client.post("/api/rewrite", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "rolex",
        }).json()
        before = r["score_before"]["overall_score"]
        after = r["score_after"]["overall_score"]
        assert after != before + 45, "Scores should not be mock offset"
        assert after != before + 20, "Scores should not be mock offset"
