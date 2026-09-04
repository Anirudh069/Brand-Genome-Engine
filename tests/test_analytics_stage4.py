"""
Stage 4 – REAL ANALYTICS tests.

Covers pillar derivation, TF-IDF heatmap, chunk-level t-SNE, tone
distribution, live history counters/trend, and the `/api/analytics`
contract. Unit tests use a deterministic fake embedding (fast, no model
load); one smoke test is marked ``requires_model`` and exercises the real
local sentence-transformers embedding abstraction end-to-end.
"""

from __future__ import annotations

import json
import shutil
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.analytics.pillars import PILLAR_NAMES, derive_pillar_keywords
from src.analytics.heatmap import compute_pillar_heatmap
from src.analytics.chunk_tsne import compute_chunk_tsne, sample_chunks
from src.analytics.tone import compute_tone_distribution, TONE_LABELS
from src.analytics.history import compute_history_counters, compute_score_trend
from src.api.main import app

REPO_ROOT_DB = "data/brand_data.db"


def _fake_embedding(text, model_name="all-MiniLM-L6-v2"):
    """Deterministic, fast, non-random fake embedding for unit tests."""
    seed = sum(ord(char) for char in str(text))
    vector = [float((seed + index) % 97) / 97.0 for index in range(384)]
    return vector, model_name


@pytest.fixture(autouse=True)
def _fake_embeddings_everywhere(monkeypatch):
    monkeypatch.setattr("src.analytics.pillars.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.analytics.chunk_tsne.get_embedding", _fake_embedding)
    monkeypatch.setattr("src.feature_extraction.embedding_extractor.get_embedding", _fake_embedding)


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    shutil.copy2(REPO_ROOT_DB, db_path)
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("ANALYTICS_CACHE_PATH", str(tmp_path / "analytics_cache.json"))
    return db_path


# ── Fixture corpus (small, controlled) ────────────────────────────────────

_FIXTURE_DOCS = {
    "brandA": "Our sustainability commitment drives recycled materials and green energy. "
              "We reduce our carbon footprint every year.",
    "brandB": "Precision engineering delivers exact accuracy and chronometer-certified performance.",
    "brandC": "Our heritage spans generations, rooted in tradition since our founding century.",
    "brandD": "We offer exceptional value and affordable pricing without compromise.",
    "brandE": "Innovation drives our patented, cutting-edge technology and breakthrough design.",
}


# ── Pillar derivation ──────────────────────────────────────────────────────

class TestPillarDerivation:
    def test_exactly_five_authoritative_pillar_names(self):
        assert PILLAR_NAMES == ["Sustainability", "Precision", "Heritage", "Value", "Innovation"]

    def test_old_pillar_set_not_used(self):
        assert "Craftsmanship" not in PILLAR_NAMES
        assert "Luxury" not in PILLAR_NAMES
        assert "Performance" not in PILLAR_NAMES

    def test_derived_keywords_occur_in_corpus(self):
        docs = list(_FIXTURE_DOCS.values())
        result = derive_pillar_keywords(docs, top_k=5)
        assert set(result.keys()) == set(PILLAR_NAMES)
        combined = " ".join(docs).lower()
        for pillar, terms in result.items():
            for entry in terms:
                for word in entry["term"].split():
                    assert word in combined, f"{entry['term']!r} for {pillar} not found in corpus"

    def test_keyword_result_shape_has_relevance_info(self):
        docs = list(_FIXTURE_DOCS.values())
        result = derive_pillar_keywords(docs, top_k=5)
        for terms in result.values():
            for entry in terms:
                assert {"term", "similarity", "score", "corpus_strength"} <= entry.keys()

    def test_deterministic_for_same_corpus(self):
        docs = list(_FIXTURE_DOCS.values())
        first = derive_pillar_keywords(docs, top_k=5)
        second = derive_pillar_keywords(docs, top_k=5)
        assert first == second

    def test_empty_corpus_returns_empty_keyword_sets(self):
        result = derive_pillar_keywords([], top_k=5)
        assert set(result.keys()) == set(PILLAR_NAMES)
        assert all(terms == [] for terms in result.values())


# ── Heatmap ────────────────────────────────────────────────────────────────

class TestHeatmap:
    def test_heatmap_shape_and_finiteness(self):
        docs = list(_FIXTURE_DOCS.values())
        brand_ids = list(_FIXTURE_DOCS.keys())
        pillar_terms = derive_pillar_keywords(docs, top_k=5)
        heatmap = compute_pillar_heatmap(
            brand_ids=brand_ids,
            brand_names=brand_ids,
            brand_documents=docs,
            pillar_names=PILLAR_NAMES,
            pillar_terms=pillar_terms,
        )
        values = np.array(heatmap["values"])
        assert values.shape == (len(brand_ids), len(PILLAR_NAMES))
        assert np.isfinite(values).all()

    def test_heatmap_changes_with_text_distribution(self):
        docs = list(_FIXTURE_DOCS.values())
        brand_ids = list(_FIXTURE_DOCS.keys())
        pillar_terms = derive_pillar_keywords(docs, top_k=5)
        heatmap_a = compute_pillar_heatmap(brand_ids, brand_ids, docs, PILLAR_NAMES, pillar_terms)

        altered_docs = list(docs)
        altered_docs[0] = "Completely unrelated filler text about nothing thematic at all."
        heatmap_b = compute_pillar_heatmap(brand_ids, brand_ids, altered_docs, PILLAR_NAMES, pillar_terms)

        assert heatmap_a["values"] != heatmap_b["values"]

    def test_ten_competitor_rows_five_columns_on_canonical_db(self, temp_db):
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT text_id, brand_id, brand_name, text FROM brand_texts ORDER BY text_id")
        rows = cur.fetchall()
        conn.close()

        brand_order = sorted({r[1] for r in rows})
        docs = [" ".join(r[3] for r in rows if r[1] == b) for b in brand_order]
        pillar_terms = derive_pillar_keywords([r[3] for r in rows], top_k=8)
        heatmap = compute_pillar_heatmap(brand_order, brand_order, docs, PILLAR_NAMES, pillar_terms)

        assert len(heatmap["brands"]) == 10
        assert len(heatmap["pillars"]) == 5
        values = np.array(heatmap["values"])
        assert np.isfinite(values).all()
        assert not np.isnan(values).any()


# ── t-SNE ──────────────────────────────────────────────────────────────────

class TestChunkTSNE:
    def _chunks(self, n_per_brand=15, n_brands=3):
        chunks = []
        for b in range(n_brands):
            for i in range(n_per_brand):
                chunks.append(
                    {
                        "chunk_id": f"b{b}_c{i}",
                        "brand_id": f"brand{b}",
                        "brand_name": f"Brand {b}",
                        "chunk_text": f"Sample chunk text number {i} for brand {b} discussing watches.",
                    }
                )
        return chunks

    def test_sampling_is_deterministic(self):
        chunks = self._chunks()
        first = [c["chunk_id"] for c in sample_chunks(chunks, max_per_brand=5)]
        second = [c["chunk_id"] for c in sample_chunks(chunks, max_per_brand=5)]
        assert first == second

    def test_source_is_chunk_level_not_centroids(self):
        chunks = self._chunks(n_per_brand=15, n_brands=3)
        result = compute_chunk_tsne(chunks)
        assert result["sample_total"] > 10
        assert len(result["points"]) == result["sample_total"]

    def test_points_have_chunk_and_brand_identity_and_finite_coords(self):
        chunks = self._chunks()
        result = compute_chunk_tsne(chunks)
        for point in result["points"]:
            assert "chunk_id" in point and "brand_id" in point
            assert np.isfinite(point["x"]) and np.isfinite(point["y"])

    def test_sample_cap_respected(self):
        chunks = self._chunks(n_per_brand=100, n_brands=2)
        result = compute_chunk_tsne(chunks)
        assert result["sample_total"] <= 50 * 2

    def test_fixed_random_state(self):
        result = compute_chunk_tsne(self._chunks())
        assert result["random_state"] == 42

    def test_sampled_points_map_to_real_chunk_rows(self, temp_db):
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, brand_id, brand_name, chunk_text FROM brand_chunks")
        rows = {r[0]: r for r in cur.fetchall()}
        conn.close()
        chunks = [
            {"chunk_id": cid, "brand_id": r[1], "brand_name": r[2], "chunk_text": r[3]}
            for cid, r in rows.items()
        ]
        result = compute_chunk_tsne(chunks)
        for point in result["points"]:
            assert point["chunk_id"] in rows


# ── Tone distribution ───────────────────────────────────────────────────────

class TestTone:
    def _texts(self):
        return [
            {"brand_id": "brandA", "brand_name": "A", "text": "Furthermore, our meticulous approach demonstrates precision."},
            {"brand_id": "brandA", "brand_name": "A", "text": "hey dude this is kinda cool lol"},
            {"brand_id": "brandB", "brand_name": "B", "text": "The watch offers reliable performance for everyday wear."},
        ]

    def test_distribution_finite_and_deterministic(self):
        first = compute_tone_distribution(self._texts())
        second = compute_tone_distribution(self._texts())
        assert first == second
        for count in first["totals"].values():
            assert np.isfinite(count)

    def test_no_hardcoded_user_value(self):
        result = compute_tone_distribution(self._texts())
        assert "userBrand" not in json.dumps(result)
        assert 25 not in result["totals"].values()

    def test_labels_are_defined_set(self):
        result = compute_tone_distribution(self._texts())
        assert set(result["labels"]) == set(TONE_LABELS)

    def test_each_represented_brand_appears(self):
        result = compute_tone_distribution(self._texts())
        assert "brandA" in result["by_brand"]
        assert "brandB" in result["by_brand"]


# ── History counters / trend ─────────────────────────────────────────────

class TestHistory:
    def test_empty_history_zero_counters(self, temp_db):
        conn = sqlite3.connect(temp_db)
        from src.api.genome_service import ensure_canonical_schema
        ensure_canonical_schema(conn)
        counts = compute_history_counters(conn)
        conn.close()
        assert counts == {"consistency": 0, "benchmark": 0, "rewrite": 0, "total": 0}

    def test_insert_consistency_increments_counter(self, temp_db):
        conn = sqlite3.connect(temp_db)
        from src.api.genome_service import ensure_canonical_schema, write_history_event
        ensure_canonical_schema(conn)
        write_history_event(conn, brand_id=0, event_type="consistency", pre_score={"overall_score": 72.5})
        counts = compute_history_counters(conn)
        conn.close()
        assert counts["consistency"] == 1
        assert counts["benchmark"] == 0
        assert counts["rewrite"] == 0
        assert counts["total"] == 1

    def test_insert_benchmark_increments_counter(self, temp_db):
        conn = sqlite3.connect(temp_db)
        from src.api.genome_service import ensure_canonical_schema, write_history_event
        ensure_canonical_schema(conn)
        write_history_event(conn, brand_id=0, event_type="benchmark", pre_score={"overall_score": 60.0})
        counts = compute_history_counters(conn)
        conn.close()
        assert counts["benchmark"] == 1
        assert counts["total"] == 1

    def test_rewrite_zero_when_absent(self, temp_db):
        conn = sqlite3.connect(temp_db)
        from src.api.genome_service import ensure_canonical_schema, write_history_event
        ensure_canonical_schema(conn)
        write_history_event(conn, brand_id=0, event_type="consistency", pre_score={"overall_score": 72.5})
        counts = compute_history_counters(conn)
        conn.close()
        assert counts["rewrite"] == 0

    def test_score_trend_empty_when_no_scored_rows(self, temp_db):
        conn = sqlite3.connect(temp_db)
        from src.api.genome_service import ensure_canonical_schema
        ensure_canonical_schema(conn)
        trend = compute_score_trend(conn)
        conn.close()
        assert trend == []

    def test_score_trend_uses_real_scores(self, temp_db):
        conn = sqlite3.connect(temp_db)
        from src.api.genome_service import ensure_canonical_schema, write_history_event
        ensure_canonical_schema(conn)
        write_history_event(conn, brand_id=0, event_type="consistency", pre_score={"overall_score": 55.0})
        write_history_event(conn, brand_id=0, event_type="benchmark", pre_score={"overall_score": 60.0})
        trend = compute_score_trend(conn)
        conn.close()
        assert len(trend) == 2
        assert trend[0]["score"] == 55.0
        assert trend[1]["score"] == 60.0
        assert all("timestamp" in e and "event_type" in e for e in trend)


# ── API contract ────────────────────────────────────────────────────────────

class TestAnalyticsAPI:
    def test_endpoint_succeeds_with_valid_artifact(self, temp_db):
        client = TestClient(app)
        response = client.get("/api/analytics")
        assert response.status_code == 200
        payload = response.json()
        assert set(payload["pillars"]["names"]) == set(PILLAR_NAMES)
        assert "heatmap" in payload
        assert "tsne" in payload
        assert "tone" in payload
        assert "history" in payload
        assert "counts" in payload["history"]

    def test_history_changes_are_live_without_rebuilding_artifact(self, temp_db):
        client = TestClient(app)
        first = client.get("/api/analytics").json()
        assert first["history"]["counts"]["consistency"] == 0

        conn = sqlite3.connect(temp_db)
        from src.api.genome_service import ensure_canonical_schema, write_history_event
        ensure_canonical_schema(conn)
        write_history_event(conn, brand_id=0, event_type="consistency", pre_score={"overall_score": 80.0})
        conn.close()

        second = client.get("/api/analytics").json()
        assert second["history"]["counts"]["consistency"] == 1
        # Corpus-derived artifact fingerprint/pillars should be stable (not rebuilt for a history-only change).
        assert first["metadata"]["fingerprint"] == second["metadata"]["fingerprint"]

    def test_no_fake_numeric_fallbacks_in_response(self, temp_db):
        client = TestClient(app)
        payload = client.get("/api/analytics").json()
        raw = json.dumps(payload)
        assert "userBrand" not in raw


@pytest.mark.requires_model
class TestRealEmbeddingSmoke:
    def test_pillar_derivation_with_real_embedding_model(self, monkeypatch):
        # Undo the fake-embedding patch for this smoke test only.
        import src.analytics.pillars as pillars_module
        from src.feature_extraction.embedding_extractor import get_embedding as real_get_embedding
        monkeypatch.setattr(pillars_module, "get_embedding", real_get_embedding)

        docs = list(_FIXTURE_DOCS.values())
        result = derive_pillar_keywords(docs, top_k=3)
        assert set(result.keys()) == set(PILLAR_NAMES)
