"""
tests/test_rag_stage5.py – Stage 5 chunk-level RAG index/retrieval tests.

Uses a small temp SQLite fixture (schema mirrors the canonical brand_chunks
table) and a deterministic keyword-based fake embedding so that similarity
ordering is fully predictable, without loading the real MiniLM model in
most tests. One opt-in ``@pytest.mark.requires_model`` test exercises the
real sentence-transformers model end-to-end.
"""

from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.retrieval import rag_builder, rag_service
from src.retrieval.rag_builder import RagBuildError, build_index, compute_fingerprint, fetch_all_chunks, validate_corpus
from src.retrieval.rag_service import RagError, retrieve_chunks

EMBEDDING_DIM = rag_builder.EMBEDDING_DIM
_KEYWORD_DIMS = {"alpha": 0, "beta": 1, "gamma": 2}


def _keyword_embedding(text, model_name="all-MiniLM-L6-v2"):
    """Deterministic bag-of-keywords fake embedding for fully predictable cosine ranking."""
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    words = text.lower().split()
    hit = False
    for word in words:
        if word in _KEYWORD_DIMS:
            vec[_KEYWORD_DIMS[word]] += 1.0
            hit = True
    if not hit:
        vec[350] = 1.0
    return vec.tolist(), model_name


@pytest.fixture(autouse=True)
def _fake_embedding(monkeypatch):
    monkeypatch.setattr(rag_builder, "get_embedding", _keyword_embedding)
    monkeypatch.setattr(rag_service, "get_embedding", _keyword_embedding)


def _make_db(db_path, rows):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE brand_chunks (
            chunk_id TEXT PRIMARY KEY,
            text_id TEXT NOT NULL,
            brand_id TEXT NOT NULL,
            brand_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            char_count INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.executemany(
        "INSERT INTO brand_chunks (chunk_id, text_id, brand_id, brand_name, source_type, chunk_text, char_count) "
        "VALUES (:chunk_id, :text_id, :brand_id, :brand_name, :source_type, :chunk_text, :char_count)",
        rows,
    )
    conn.commit()
    conn.close()


def _row(chunk_id, brand_id, brand_name, text):
    return {
        "chunk_id": chunk_id,
        "text_id": f"{chunk_id}_src",
        "brand_id": brand_id,
        "brand_name": brand_name,
        "source_type": "website_copy",
        "chunk_text": text,
        "char_count": len(text),
    }


def _small_corpus():
    """3 chunks in brandA, 2 in brandB — enough to test brand scoping/ranking."""
    return [
        _row("a1", "brandA", "Brand A", "alpha alpha alpha content"),
        _row("a2", "brandA", "Brand A", "beta beta content only"),
        _row("a3", "brandA", "Brand A", "alpha beta mixed content"),
        _row("b1", "brandB", "Brand B", "gamma content only"),
        _row("b2", "brandB", "Brand B", "alpha content in brand b"),
    ]


def _wide_corpus():
    """8 chunks in one brand so top_k=5 default has enough candidates."""
    rows = []
    for i in range(8):
        rows.append(_row(f"w{i}", "wide_brand", "Wide Brand", f"alpha chunk number {i} filler words here"))
    return rows


# ── Builder: ordering / fingerprint / validation ───────────────────────────

def test_fetch_all_chunks_deterministic_ordering(tmp_path):
    db_path = tmp_path / "brand_data.db"
    rows = _small_corpus()
    # Insert in reverse order to prove ORDER BY chunk_id is applied, not insertion order.
    _make_db(db_path, list(reversed(rows)))
    chunks = fetch_all_chunks(str(db_path))
    assert [c["chunk_id"] for c in chunks] == sorted(r["chunk_id"] for r in rows)


def test_duplicate_chunk_id_rejected():
    rows = _small_corpus()
    rows.append(dict(rows[0]))
    with pytest.raises(RagBuildError, match="duplicate chunk_id"):
        validate_corpus(rows)


def test_blank_chunk_text_rejected():
    rows = _small_corpus()
    rows[0]["chunk_text"] = "   "
    with pytest.raises(RagBuildError, match="blank chunk_text"):
        validate_corpus(rows)


def test_empty_corpus_rejected():
    with pytest.raises(RagBuildError, match="empty"):
        validate_corpus([])


def test_fingerprint_deterministic_and_sensitive_to_text_change():
    rows = _small_corpus()
    fp1 = compute_fingerprint(rows)
    fp2 = compute_fingerprint(list(rows))  # same content, new list object
    assert fp1 == fp2

    changed = [dict(r) for r in rows]
    changed[0]["chunk_text"] = changed[0]["chunk_text"] + " EXTRA"
    fp3 = compute_fingerprint(changed)
    assert fp3 != fp1


# ── Builder: full index build ──────────────────────────────────────────────

@pytest.fixture()
def built_index(tmp_path):
    db_path = tmp_path / "brand_data.db"
    out_dir = tmp_path / "rag_artifact"
    _make_db(db_path, _small_corpus())
    manifest = build_index(str(db_path), str(out_dir))
    return db_path, out_dir, manifest


def test_build_produces_one_vector_per_chunk_and_matching_metadata(built_index):
    db_path, out_dir, manifest = built_index
    assert manifest["embedding_dim"] == EMBEDDING_DIM
    assert manifest["chunk_count"] == 5
    assert manifest["brands"]["brandA"]["count"] == 3
    assert manifest["brands"]["brandB"]["count"] == 2

    metadata = json.loads((out_dir / "metadata.json").read_text())
    for brand_id, info in manifest["brands"].items():
        assert len(metadata[brand_id]) == info["count"]
        index = rag_builder.load_brand_index(str(out_dir), brand_id, info["index_file"])
        assert index.ntotal == info["count"]
        assert index.d == EMBEDDING_DIM


def test_build_rejects_wrong_embedding_dimension(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    out_dir = tmp_path / "rag_artifact"
    _make_db(db_path, _small_corpus())

    def bad_embedding(text, model_name="all-MiniLM-L6-v2"):
        return [0.1, 0.2, 0.3], model_name

    monkeypatch.setattr(rag_builder, "get_embedding", bad_embedding)
    with pytest.raises(RagBuildError, match="dimension"):
        build_index(str(db_path), str(out_dir))
    assert not out_dir.exists()


def test_build_rejects_nonfinite_vector(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    out_dir = tmp_path / "rag_artifact"
    _make_db(db_path, _small_corpus())

    def nan_embedding(text, model_name="all-MiniLM-L6-v2"):
        vec = [float("nan")] * EMBEDDING_DIM
        return vec, model_name

    monkeypatch.setattr(rag_builder, "get_embedding", nan_embedding)
    with pytest.raises(RagBuildError, match="non-finite"):
        build_index(str(db_path), str(out_dir))
    assert not out_dir.exists()


def test_build_idempotent_same_fingerprint_and_counts(tmp_path):
    db_path = tmp_path / "brand_data.db"
    out_dir = tmp_path / "rag_artifact"
    _make_db(db_path, _small_corpus())

    manifest1 = build_index(str(db_path), str(out_dir))
    manifest2 = build_index(str(db_path), str(out_dir))

    assert manifest1["fingerprint"] == manifest2["fingerprint"]
    assert manifest1["chunk_count"] == manifest2["chunk_count"]
    assert manifest1["brands"] == manifest2["brands"]


def test_atomic_build_does_not_replace_valid_index_on_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    out_dir = tmp_path / "rag_artifact"
    _make_db(db_path, _small_corpus())

    good_manifest = build_index(str(db_path), str(out_dir))

    def bad_embedding(text, model_name="all-MiniLM-L6-v2"):
        return [float("nan")] * EMBEDDING_DIM, model_name

    monkeypatch.setattr(rag_builder, "get_embedding", bad_embedding)
    with pytest.raises(RagBuildError):
        build_index(str(db_path), str(out_dir))

    reloaded = json.loads((out_dir / "manifest.json").read_text())
    assert reloaded["fingerprint"] == good_manifest["fingerprint"]


# ── Retrieval service ───────────────────────────────────────────────────────

def test_retrieve_rejects_blank_query(built_index):
    _db_path, out_dir, _ = built_index
    with pytest.raises(RagError) as exc_info:
        retrieve_chunks("   ", "brandA", artifact_dir=str(out_dir), db_path=str(_db_path))
    assert exc_info.value.status_code == 400


def test_retrieve_rejects_unknown_brand(built_index):
    db_path, out_dir, _ = built_index
    with pytest.raises(RagError) as exc_info:
        retrieve_chunks("alpha", "brandZ", artifact_dir=str(out_dir), db_path=str(db_path))
    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("bad_k", [0, -1, 11, 100])
def test_retrieve_rejects_invalid_top_k(built_index, bad_k):
    db_path, out_dir, _ = built_index
    with pytest.raises(RagError) as exc_info:
        retrieve_chunks("alpha", "brandA", top_k=bad_k, artifact_dir=str(out_dir), db_path=str(db_path))
    assert exc_info.value.status_code == 400


def test_retrieve_default_top_k_is_5(built_index):
    db_path, out_dir, _ = built_index
    result = retrieve_chunks("alpha", "brandA", artifact_dir=str(out_dir), db_path=str(db_path))
    assert result["top_k"] == 5
    # only 3 chunks exist in brandA, so available count is returned safely
    assert len(result["results"]) == 3


def test_retrieve_result_count_equals_k_when_enough_chunks(tmp_path):
    db_path = tmp_path / "brand_data.db"
    out_dir = tmp_path / "rag_artifact"
    _make_db(db_path, _wide_corpus())
    build_index(str(db_path), str(out_dir))

    result = retrieve_chunks("alpha", "wide_brand", top_k=5, artifact_dir=str(out_dir), db_path=str(db_path))
    assert len(result["results"]) == 5


def test_retrieve_results_scoped_ranked_and_well_formed(built_index):
    db_path, out_dir, _ = built_index
    result = retrieve_chunks("alpha", "brandA", top_k=3, artifact_dir=str(out_dir), db_path=str(db_path))
    results = result["results"]

    assert all(item["brand_id"] == "brandA" for item in results)
    chunk_ids = [item["chunk_id"] for item in results]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert [item["rank"] for item in results] == list(range(1, len(results) + 1))
    scores = [item["score"] for item in results]
    assert all(math.isfinite(s) for s in scores)
    assert scores == sorted(scores, reverse=True)

    # a1 is pure-alpha -> must rank first for an alpha query
    assert results[0]["chunk_id"] == "a1"


def test_retrieve_chunk_text_matches_canonical_db(built_index):
    db_path, out_dir, _ = built_index
    result = retrieve_chunks("alpha", "brandA", artifact_dir=str(out_dir), db_path=str(db_path))

    conn = sqlite3.connect(db_path)
    try:
        db_texts = dict(conn.execute("SELECT chunk_id, chunk_text FROM brand_chunks").fetchall())
    finally:
        conn.close()

    for item in result["results"]:
        assert item["chunk_text"] == db_texts[item["chunk_id"]]


def test_retrieve_deterministic_ranking_for_same_query(built_index):
    db_path, out_dir, _ = built_index
    r1 = retrieve_chunks("alpha", "brandA", artifact_dir=str(out_dir), db_path=str(db_path))
    r2 = retrieve_chunks("alpha", "brandA", artifact_dir=str(out_dir), db_path=str(db_path))
    assert [i["chunk_id"] for i in r1["results"]] == [i["chunk_id"] for i in r2["results"]]
    assert [i["score"] for i in r1["results"]] == [i["score"] for i in r2["results"]]


def test_retrieve_different_query_produces_different_ranking(built_index):
    db_path, out_dir, _ = built_index
    alpha_order = [i["chunk_id"] for i in retrieve_chunks("alpha", "brandA", artifact_dir=str(out_dir), db_path=str(db_path))["results"]]
    beta_order = [i["chunk_id"] for i in retrieve_chunks("beta", "brandA", artifact_dir=str(out_dir), db_path=str(db_path))["results"]]
    assert alpha_order != beta_order
    assert alpha_order[0] == "a1"
    assert beta_order[0] == "a2"


def test_retrieve_never_leaks_cross_brand_chunks(built_index):
    db_path, out_dir, _ = built_index
    # "alpha" vocabulary exists in both brandA (a1, a3) and brandB (b2).
    result_b = retrieve_chunks("alpha", "brandB", artifact_dir=str(out_dir), db_path=str(db_path))
    assert {i["brand_id"] for i in result_b["results"]} == {"brandB"}
    assert result_b["results"][0]["chunk_id"] == "b2"

    result_a = retrieve_chunks("alpha", "brandA", artifact_dir=str(out_dir), db_path=str(db_path))
    assert {i["brand_id"] for i in result_a["results"]} == {"brandA"}


def test_retrieve_missing_artifact_handled_honestly(tmp_path):
    db_path = tmp_path / "brand_data.db"
    _make_db(db_path, _small_corpus())
    with pytest.raises(RagError) as exc_info:
        retrieve_chunks("alpha", "brandA", artifact_dir=str(tmp_path / "does_not_exist"), db_path=str(db_path))
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "index_missing"


def test_retrieve_stale_fingerprint_blocks_retrieval(built_index):
    db_path, out_dir, _ = built_index
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO brand_chunks (chunk_id, text_id, brand_id, brand_name, source_type, chunk_text, char_count) "
        "VALUES ('a4', 'a4_src', 'brandA', 'Brand A', 'website_copy', 'new alpha content', 17)"
    )
    conn.commit()
    conn.close()

    with pytest.raises(RagError) as exc_info:
        retrieve_chunks("alpha", "brandA", artifact_dir=str(out_dir), db_path=str(db_path))
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "index_stale"


# ── Real model integration (opt-in) ─────────────────────────────────────────

@pytest.mark.requires_model
def test_real_model_build_and_retrieve(tmp_path, monkeypatch):
    # Undo the module-level fake embedding fixture for this test only.
    from src.feature_extraction.embedding_extractor import get_embedding as real_get_embedding

    monkeypatch.setattr(rag_builder, "get_embedding", real_get_embedding)
    monkeypatch.setattr(rag_service, "get_embedding", real_get_embedding)

    db_path = tmp_path / "brand_data.db"
    out_dir = tmp_path / "rag_artifact"
    _make_db(
        db_path,
        [
            _row("r1", "rolex", "Rolex", "Precision and chronometer accuracy define every Rolex movement."),
            _row("r2", "rolex", "Rolex", "Heritage and tradition since 1905 shaped the brand's identity."),
            _row("o1", "omega", "Omega", "Innovation in materials and engineering drives Omega watchmaking."),
        ],
    )
    manifest = build_index(str(db_path), str(out_dir))
    assert manifest["embedding_dim"] == 384

    result = retrieve_chunks(
        "chronometer precision accuracy",
        "rolex",
        top_k=2,
        artifact_dir=str(out_dir),
        db_path=str(db_path),
    )
    assert result["results"]
    assert all(math.isfinite(item["score"]) for item in result["results"])
    assert result["results"][0]["chunk_id"] == "r1"


# ── API ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    out_dir = tmp_path / "rag_artifact"
    _make_db(db_path, _small_corpus())
    build_index(str(db_path), str(out_dir))

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("RAG_INDEX_DIR", str(out_dir))

    from src.api.main import app

    return TestClient(app), db_path, out_dir


def test_api_retrieve_success_default_k(api_client):
    client, _db_path, _out_dir = api_client
    resp = client.post("/api/rag/retrieve", json={"text": "alpha", "brand_id": "brandA"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_k"] == 5
    assert body["brand_id"] == "brandA"
    assert all(r["brand_id"] == "brandA" for r in body["results"])


def test_api_retrieve_custom_k(api_client):
    client, _db_path, _out_dir = api_client
    resp = client.post("/api/rag/retrieve", json={"text": "alpha", "brand_id": "brandA", "top_k": 2})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_api_retrieve_unknown_brand(api_client):
    client, _db_path, _out_dir = api_client
    resp = client.post("/api/rag/retrieve", json={"text": "alpha", "brand_id": "nope"})
    assert resp.status_code == 404


def test_api_retrieve_blank_text(api_client):
    client, _db_path, _out_dir = api_client
    resp = client.post("/api/rag/retrieve", json={"text": "   ", "brand_id": "brandA"})
    assert resp.status_code == 400


def test_api_retrieve_invalid_top_k(api_client):
    client, _db_path, _out_dir = api_client
    resp = client.post("/api/rag/retrieve", json={"text": "alpha", "brand_id": "brandA", "top_k": 0})
    assert resp.status_code == 400


def test_api_retrieve_missing_index(tmp_path, monkeypatch):
    db_path = tmp_path / "brand_data.db"
    _make_db(db_path, _small_corpus())
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "no_index_here"))

    from src.api.main import app

    client = TestClient(app)
    resp = client.post("/api/rag/retrieve", json={"text": "alpha", "brand_id": "brandA"})
    assert resp.status_code == 503


def test_api_retrieve_stale_index(api_client):
    client, db_path, _out_dir = api_client
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO brand_chunks (chunk_id, text_id, brand_id, brand_name, source_type, chunk_text, char_count) "
        "VALUES ('a4', 'a4_src', 'brandA', 'Brand A', 'website_copy', 'new alpha content', 17)"
    )
    conn.commit()
    conn.close()

    resp = client.post("/api/rag/retrieve", json={"text": "alpha", "brand_id": "brandA"})
    assert resp.status_code == 503


def test_api_retrieve_chunk_text_matches_db(api_client):
    client, db_path, _out_dir = api_client
    resp = client.post("/api/rag/retrieve", json={"text": "alpha", "brand_id": "brandA"})
    body = resp.json()

    conn = sqlite3.connect(db_path)
    try:
        db_texts = dict(conn.execute("SELECT chunk_id, chunk_text FROM brand_chunks").fetchall())
    finally:
        conn.close()

    for item in body["results"]:
        assert item["chunk_text"] == db_texts[item["chunk_id"]]
