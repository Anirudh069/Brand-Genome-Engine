"""
tests/test_rewrite_stage6.py – Stage 6: RAG Rewrite + OpenAI integration.

Covers:
    * canonical scorer reuse (pre == post == same function as Consistency API)
    * cost-ordered local preconditions (provider never called unless all pass)
    * Stage 5 semantic user_brand RAG retrieval only (no legacy/lexical/competitor leakage)
    * provider abstraction (mocked — no network calls in this file)
    * exactly one rewrite analysis_history row per successful request
    * response contract

Uses only temp copies of the canonical DB and temp RAG artifact directories.
Never touches data/brand_data.db. The provider is always dependency-injected
(mocked) here — the one real OpenAI smoke test is run separately/manually.
"""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.genome_service import USER_BRAND_ALIAS, ensure_canonical_schema, initialize_user_genome
from src.retrieval import rag_builder, rag_service
from src.retrieval.rag_builder import build_index
from src.retrieval.rag_service import RagError
from src.rewrite import rewrite_service
from src.rewrite.openai_provider import RewriteProviderError
from src.rewrite.rewrite_service import RewriteError, rewrite_copy
from src.scoring import consistency
from src.scoring.consistency import score_against_user_genome

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

ON_BRAND_TEXT = (
    "Precision and craftsmanship guide our enduring commitment to refined, "
    "measured excellence in every timepiece."
)
OFF_BRAND_TEXT = "this watch is super cool and awesome lol, totally rad and fun."


def _fake_embedding(text, model_name="all-MiniLM-L6-v2"):
    seed = sum(ord(char) for char in str(text))
    vector = [float((seed + index) % 97) / 97.0 for index in range(384)]
    return vector, model_name


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch):
    monkeypatch.setattr("src.api.genome_service.get_embedding", _fake_embedding)
    monkeypatch.setattr(rag_builder, "get_embedding", _fake_embedding)
    monkeypatch.setattr(rag_service, "get_embedding", _fake_embedding)
    monkeypatch.setattr(consistency, "get_embedding", _fake_embedding)


class FakeProvider:
    """Dependency-injected mock provider — never makes a network call."""

    name = "openai"
    model = "gpt-5.6-luna-test"

    def __init__(self, response_text="A refined line of enduring craftsmanship.", raise_error=None):
        self.response_text = response_text
        self.raise_error = raise_error
        self.calls: list[dict] = []

    def rewrite(self, *, instructions, input_text, max_output_tokens=400):
        self.calls.append(
            {"instructions": instructions, "input_text": input_text, "max_output_tokens": max_output_tokens}
        )
        if self.raise_error is not None:
            raise self.raise_error
        return self.response_text


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    shutil.copy2(CANONICAL_DB, db_path)
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    return db_path


@pytest.fixture()
def rag_dir(tmp_path, monkeypatch):
    out_dir = tmp_path / "rag"
    monkeypatch.setenv("RAG_INDEX_DIR", str(out_dir))
    return out_dir


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_canonical_schema(conn)
    return conn


def _init_genome(db_path, mission=_MISSION, snippets=None, designation=_DESIGNATION):
    conn = _connect(db_path)
    try:
        return initialize_user_genome(conn, designation, mission, snippets or list(_SNIPPETS))
    finally:
        conn.close()


def _build_rag_index(db_path, out_dir):
    build_index(str(db_path), str(out_dir))


def _ready(db_path, out_dir):
    """Full local precondition chain: genome initialized + fresh RAG index."""
    _init_genome(db_path)
    _build_rag_index(db_path, out_dir)


def _history_rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM analysis_history ORDER BY id").fetchall()]
    finally:
        conn.close()


def _user_chunk_ids(db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT chunk_id FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,)).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def _competitor_chunk_ids(db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT chunk_id FROM brand_chunks WHERE brand_id != ?", (USER_BRAND_ALIAS,)).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


# ── Scoring: canonical scorer reused, no side effects ─────────────────────


class TestCanonicalScorer:
    def test_pure_scorer_writes_no_history(self, temp_db):
        _init_genome(temp_db)
        conn = _connect(temp_db)
        genome = rewrite_service._require_user_genome(conn)
        before = len(_history_rows(temp_db))
        score_against_user_genome(ON_BRAND_TEXT, genome)
        conn.close()
        assert len(_history_rows(temp_db)) == before

    def test_rewrite_prescore_equals_canonical_scorer(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        conn = _connect(temp_db)
        genome = rewrite_service._require_user_genome(conn)
        direct = score_against_user_genome(OFF_BRAND_TEXT, genome)
        conn.close()

        conn = _connect(temp_db)
        provider = FakeProvider()
        result = rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=provider)
        conn.close()

        assert result["score_before"] == pytest.approx(direct["score_overall"], abs=1e-6)

    def test_rewrite_postscore_uses_same_scorer(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        provider = FakeProvider(response_text="A precise, measured line of enduring craftsmanship.")
        conn = _connect(temp_db)
        result = rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=provider)
        genome = rewrite_service._require_user_genome(conn)
        conn.close()

        direct_after = score_against_user_genome(provider.response_text, genome)
        assert result["score_after"] == pytest.approx(direct_after["score_overall"], abs=1e-6)

    def test_rewrite_causes_zero_consistency_history_rows(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        conn = _connect(temp_db)
        rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=FakeProvider())
        conn.close()
        rows = _history_rows(temp_db)
        assert sum(1 for r in rows if r["event_type"] == "consistency") == 0
        assert sum(1 for r in rows if r["event_type"] == "rewrite") == 1


# ── Preconditions (cost-ordered; provider must not be called on failure) ──


class TestPreconditions:
    def test_blank_text_rejected(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        conn = _connect(temp_db)
        provider = FakeProvider()
        with pytest.raises(RewriteError) as exc:
            rewrite_copy(conn, "   ", top_k=5, provider=provider)
        conn.close()
        assert exc.value.status_code == 400
        assert provider.calls == []

    def test_missing_genome_returns_400_and_no_provider_call(self, temp_db):
        conn = _connect(temp_db)
        provider = FakeProvider()
        with pytest.raises(RewriteError) as exc:
            rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=provider)
        conn.close()
        assert exc.value.status_code == 400
        assert exc.value.detail["error"] == "genome_not_initialized"
        assert provider.calls == []

    def test_missing_index_returns_503_and_no_provider_call(self, temp_db, rag_dir):
        _init_genome(temp_db)  # genome + chunks exist, but no RAG artifact built at rag_dir
        conn = _connect(temp_db)
        provider = FakeProvider()
        with pytest.raises(RewriteError) as exc:
            rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=provider)
        conn.close()
        assert exc.value.status_code == 503
        assert exc.value.detail["error"] == "index_missing"
        assert provider.calls == []

    def test_stale_index_returns_503_and_no_provider_call(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        _init_genome(temp_db, mission="A completely different mission changes the corpus fingerprint.")
        conn = _connect(temp_db)
        provider = FakeProvider()
        with pytest.raises(RewriteError) as exc:
            rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=provider)
        conn.close()
        assert exc.value.status_code == 503
        assert exc.value.detail["error"] == "index_stale"
        assert provider.calls == []

    def test_missing_user_brand_index_maps_to_user_grounding_not_indexed(self, temp_db, rag_dir, monkeypatch):
        _ready(temp_db, rag_dir)

        def _raise_unknown_brand(*args, **kwargs):
            raise RagError(404, {"error": "unknown_brand", "brand_id": USER_BRAND_ALIAS, "message": "not indexed"})

        monkeypatch.setattr(rewrite_service, "retrieve_chunks", _raise_unknown_brand)
        conn = _connect(temp_db)
        provider = FakeProvider()
        with pytest.raises(RewriteError) as exc:
            rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=provider)
        conn.close()
        assert exc.value.status_code == 503
        assert exc.value.detail["error"] == "user_grounding_not_indexed"
        assert provider.calls == []

    def test_invalid_top_k_rejected_by_api(self, temp_db, rag_dir, monkeypatch):
        _ready(temp_db, rag_dir)
        monkeypatch.setenv("SQLITE_DB_PATH", str(temp_db))
        monkeypatch.setenv("RAG_INDEX_DIR", str(rag_dir))
        from src.api.main import app

        client = TestClient(app)
        r = client.post("/api/rewrite", json={"text": OFF_BRAND_TEXT, "top_k": 0})
        assert r.status_code == 422


# ── RAG: user_brand-only semantic retrieval, real chunk IDs ───────────────


class TestRagRetrieval:
    def test_top_k_defaults_to_5_and_all_chunks_are_user_brand(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        user_ids = _user_chunk_ids(temp_db)
        competitor_ids = _competitor_chunk_ids(temp_db)
        conn = _connect(temp_db)
        result = rewrite_copy(conn, OFF_BRAND_TEXT, top_k=None, provider=FakeProvider())
        conn.close()

        chunks = result["grounding_chunks"]
        assert len(chunks) == 5
        for chunk in chunks:
            assert chunk["chunk_id"] in user_ids
            assert chunk["chunk_id"] not in competitor_ids
            assert math.isfinite(chunk["score"])
            assert chunk["chunk_text"]

    def test_legacy_grounding_helper_no_longer_exists(self):
        import src.api.main as main_module

        assert not hasattr(main_module, "retrieve_grounding_chunks")


# ── Provider: mocked, called exactly once, prompt content checks ─────────


class TestProvider:
    def test_provider_called_exactly_once_with_expected_prompt_content(self, temp_db, rag_dir, monkeypatch):
        _ready(temp_db, rag_dir)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
        conn = _connect(temp_db)
        provider = FakeProvider()
        result = rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=provider)
        conn.close()

        assert len(provider.calls) == 1
        call = provider.calls[0]
        prompt = call["input_text"]
        assert "EDIT PLAN" in prompt
        assert OFF_BRAND_TEXT in prompt
        assert _DESIGNATION in prompt
        for chunk in result["grounding_chunks"]:
            assert chunk["chunk_id"] in prompt
        assert "embedding" not in prompt.lower()
        assert "sk-test-not-a-real-key" not in prompt
        assert "sk-test-not-a-real-key" not in call["instructions"]

    def test_empty_rewrite_rejected(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        conn = _connect(temp_db)
        with pytest.raises(RewriteError) as exc:
            rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=FakeProvider(response_text="   "))
        conn.close()
        assert exc.value.status_code == 502
        assert exc.value.detail["error"] == "rewrite_provider_invalid_response"

    def test_provider_exception_surfaces_as_clean_error(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        conn = _connect(temp_db)
        provider_error = RewriteProviderError(503, {"error": "rewrite_provider_unavailable", "message": "no key"})
        with pytest.raises(RewriteError) as exc:
            rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=FakeProvider(raise_error=provider_error))
        conn.close()
        assert exc.value.status_code == 503
        assert exc.value.detail["error"] == "rewrite_provider_unavailable"

    def test_no_repeated_provider_call_regardless_of_score_direction(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        conn = _connect(temp_db)
        provider = FakeProvider(response_text="lol totally cool and awesome, super rad!!")
        rewrite_copy(conn, ON_BRAND_TEXT, top_k=5, provider=provider)
        conn.close()
        assert len(provider.calls) == 1


# ── History: exactly one rewrite row per successful request ──────────────


class TestHistory:
    def test_successful_rewrite_creates_exactly_one_row(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        before = len(_history_rows(temp_db))
        conn = _connect(temp_db)
        provider = FakeProvider()
        result = rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=provider)
        conn.close()

        rows = _history_rows(temp_db)
        assert len(rows) == before + 1
        row = rows[-1]
        assert row["event_type"] == "rewrite"
        assert row["brand_id"] == 0
        assert json.loads(row["pre_score"]) == pytest.approx(result["score_before"])
        assert json.loads(row["post_score"]) == pytest.approx(result["score_after"])

        extra = json.loads(row["extra_json"])
        assert extra["rewritten_text"] == result["rewritten_text"]
        assert extra["provider"] == "openai"
        assert extra["model"] == provider.model
        assert set(extra["retrieved_chunk_ids"]) == {c["chunk_id"] for c in result["grounding_chunks"]}

    def test_failed_provider_call_creates_no_history_row(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        before = len(_history_rows(temp_db))
        conn = _connect(temp_db)
        provider_error = RewriteProviderError(502, {"error": "rewrite_provider_error", "message": "boom"})
        with pytest.raises(RewriteError):
            rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=FakeProvider(raise_error=provider_error))
        conn.close()
        assert len(_history_rows(temp_db)) == before


# ── Response contract ──────────────────────────────────────────────────────


class TestResponseContract:
    def test_response_shape(self, temp_db, rag_dir):
        _ready(temp_db, rag_dir)
        conn = _connect(temp_db)
        result = rewrite_copy(conn, OFF_BRAND_TEXT, top_k=5, provider=FakeProvider())
        conn.close()

        assert result["original_text"] == OFF_BRAND_TEXT
        assert isinstance(result["rewritten_text"], str) and result["rewritten_text"]
        assert 0 <= result["score_before"] <= 100
        assert 0 <= result["score_after"] <= 100
        assert result["score_delta"] == pytest.approx(result["score_after"] - result["score_before"], abs=1e-6)
        assert "tone" in result["feature_breakdown_before"]
        assert "tone" in result["feature_breakdown_after"]
        assert isinstance(result["diagnostic_breakdown_before"], list)
        assert isinstance(result["diagnostic_breakdown_after"], list)
        assert result["drift_report"] == result["diagnostic_breakdown_before"]
        assert isinstance(result["edit_plan"], dict) and result["edit_plan"].get("goals")
        assert len(result["grounding_chunks"]) == 5
        assert result["provider"]["name"] == "openai"
        assert result["provider"]["model"]
        assert result["timestamp"]

    def test_score_after_may_be_lower_than_before(self, temp_db, rag_dir):
        """Do not force score-chasing/improvement — an honest decrease must be reportable."""
        _ready(temp_db, rag_dir)
        conn = _connect(temp_db)
        result = rewrite_copy(
            conn, ON_BRAND_TEXT, top_k=5, provider=FakeProvider(response_text="lol so cool and awesome dude")
        )
        conn.close()
        assert result["score_after"] < result["score_before"]
