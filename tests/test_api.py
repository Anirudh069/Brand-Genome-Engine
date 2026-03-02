# filepath: tests/test_api.py
"""Comprehensive API tests for the Brand Genome Engine FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

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
        r = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "rolex",
        })
        assert r.status_code == 200
        body = r.json()
        for k in SCORE_KEYS:
            assert k in body, f"Missing key: {k}"

    def test_scores_in_range(self):
        r = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "rolex",
        })
        body = r.json()
        for k in SCORE_KEYS:
            assert 0 <= body[k] <= 100, f"{k}={body[k]} out of [0,100]"

    def test_brand_name_present(self):
        r = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "rolex",
        })
        assert r.json()["brand_name"] == "Rolex"

    def test_text_too_short(self):
        r = client.post("/api/check-consistency", json={
            "text": "Hi", "brand_id": "rolex",
        })
        assert r.status_code == 200
        assert r.json()["error"] == "text_too_short"

    def test_unknown_brand_returns_404(self):
        r = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "unknown_brand_xyz",
        })
        assert r.status_code == 404
        body = r.json()
        assert body["detail"]["error"] == "profile_missing"

    def test_on_brand_scores_higher_than_off_brand(self):
        on = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "rolex",
        }).json()
        off = client.post("/api/check-consistency", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "rolex",
        }).json()
        assert on["overall_score"] > off["overall_score"], (
            f"On-brand ({on['overall_score']}) should beat "
            f"off-brand ({off['overall_score']})"
        )


class TestCheckConsistencyContract:
    """Frozen response-schema contract for POST /api/check-consistency."""

    FROZEN_KEYS = {
        "brand_id", "brand_name",
        "overall_score", "tone_pct", "vocab_overlap_pct",
        "sentiment_alignment_pct", "readability_match_pct",
        "error",
    }

    def test_response_keys_exact(self):
        r = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "rolex",
        })
        assert r.status_code == 200
        assert set(r.json().keys()) == self.FROZEN_KEYS

    def test_pct_values_numeric_and_clamped(self):
        body = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "rolex",
        }).json()
        for k in SCORE_KEYS:
            v = body[k]
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"
            assert 0 <= v <= 100, f"{k}={v} out of [0,100]"

    def test_error_null_on_success(self):
        body = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "rolex",
        }).json()
        assert body["error"] is None

    def test_short_text_returns_zeros(self):
        body = client.post("/api/check-consistency", json={
            "text": "Hi", "brand_id": "rolex",
        }).json()
        assert body["error"] == "text_too_short"
        for k in SCORE_KEYS:
            assert body[k] == 0


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
        r = client.get("/api/profile")
        assert r.status_code == 200
        body = r.json()
        assert "name" in body
        assert "mission" in body
        assert "tone" in body

    def test_update_profile(self):
        r = client.post("/api/profile", json={
            "brand_name": "TestBrand",
            "mission": "Delivering precision engineering for the modern era",
            "tone": "Sophisticated",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["profile"]["name"] == "TestBrand"

    def test_keyword_extraction(self):
        r = client.post("/api/profile", json={
            "brand_name": "KWTest",
            "mission": "Our commitment to precision engineering and craftsmanship defines every timepiece",
            "tone": "Sophisticated",
        })
        body = r.json()
        kw = body["profile"]["top_keywords"]
        assert isinstance(kw, list)
        assert len(kw) >= 1
        all_kw = " ".join(kw).lower()
        assert any(w in all_kw for w in ("commitment", "precision", "engineering",
                                          "craftsmanship", "defines", "timepiece"))

    def test_sentiment_computed(self):
        r = client.post("/api/profile", json={
            "brand_name": "SentTest",
            "mission": "We build terrible broken products that nobody wants",
            "tone": "Sophisticated",
        })
        body = r.json()
        sent = body["profile"]["avg_sentiment"]
        assert isinstance(sent, float)
        assert sent < 0.85, f"Expected sentiment < 0.85 for negative mission, got {sent}"

    def test_tone_keyword_fallback(self):
        r = client.post("/api/profile", json={
            "brand_name": "FallbackTest",
            "mission": "",
            "tone": "Technical",
        })
        body = r.json()
        kw = body["profile"]["top_keywords"]
        assert isinstance(kw, list)
        assert len(kw) >= 1


class TestAnalytics:
    def test_returns_data(self):
        r = client.get("/api/analytics")
        assert r.status_code == 200
        body = r.json()
        for key in ("total_analyzed", "avg_consistency", "deviations_fixed", "trend"):
            assert key in body, f"Missing analytics key: {key}"

    def test_numeric_values(self):
        r = client.get("/api/analytics")
        body = r.json()
        assert isinstance(body["total_analyzed"], (int, float))
        assert isinstance(body["avg_consistency"], (int, float))
        assert isinstance(body["deviations_fixed"], (int, float))
        assert isinstance(body["trend"], list)
        assert len(body["trend"]) >= 1


class TestBenchmark:
    def test_structure(self):
        r = client.post("/api/benchmark", json={
            "my_brand": "Rolex", "competitor": "omega",
            "metric": "Sentiment Distribution",
        })
        assert r.status_code == 200
        body = r.json()
        assert "my_brand" in body
        assert "competitor" in body
        assert "radar_data" in body

    def test_brand_keys(self):
        r = client.post("/api/benchmark", json={
            "my_brand": "Rolex", "competitor": "omega",
            "metric": "Sentiment Distribution",
        })
        body = r.json()
        for section in ("my_brand", "competitor"):
            assert "name" in body[section]
            assert "value" in body[section]
            assert "label" in body[section]

    def test_radar_data(self):
        r = client.post("/api/benchmark", json={
            "my_brand": "Rolex", "competitor": "omega",
            "metric": "Keyword Overlap",
        })
        body = r.json()
        rd = body["radar_data"]
        assert isinstance(rd, list)
        assert len(rd) >= 3
        for item in rd:
            assert "subject" in item
            assert "A" in item
            assert "B" in item

    def test_different_brands_produce_different_scores(self):
        r1 = client.post("/api/benchmark", json={
            "my_brand": "Rolex", "competitor": "tissot",
            "metric": "Sentiment Distribution",
        })
        body = r1.json()
        assert body["my_brand"]["value"] != body["competitor"]["value"]

    def test_label_categories(self):
        r = client.post("/api/benchmark", json={
            "my_brand": "Rolex", "competitor": "omega",
            "metric": "Readability Level",
        })
        body = r.json()
        valid_labels = {"High Alignment", "Moderate Alignment", "Low Alignment"}
        assert body["my_brand"]["label"] in valid_labels
        assert body["competitor"]["label"] in valid_labels


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
        r = client.post("/api/check-consistency", json={"brand_id": "rolex"})
        assert r.status_code == 422

    def test_check_consistency_missing_brand(self):
        r = client.post("/api/check-consistency", json={"text": "some text here"})
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
        s1 = client.post("/api/check-consistency", json={
            "text": ROLEX_ON_BRAND, "brand_id": "rolex",
        }).json()["overall_score"]
        s2 = client.post("/api/check-consistency", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "rolex",
        }).json()["overall_score"]
        assert s1 != s2, "Real scoring should differentiate texts"

    def test_vocab_overlap_detects_keywords(self):
        with_kw = client.post("/api/check-consistency", json={
            "text": "Precision and perpetual excellence in craftsmanship define this oyster timepiece.",
            "brand_id": "rolex",
        }).json()
        without_kw = client.post("/api/check-consistency", json={
            "text": "This is a very casual and fun everyday accessory that looks neat.",
            "brand_id": "rolex",
        }).json()
        assert with_kw["vocab_overlap_pct"] > without_kw["vocab_overlap_pct"]

    def test_rewrite_score_not_hardcoded(self):
        r = client.post("/api/rewrite", json={
            "text": ROLEX_OFF_BRAND, "brand_id": "rolex",
        }).json()
        before = r["score_before"]["overall_score"]
        after = r["score_after"]["overall_score"]
        assert after != before + 45, "Scores should not be mock offset"
        assert after != before + 20, "Scores should not be mock offset"
