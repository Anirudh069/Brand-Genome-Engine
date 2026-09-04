"""
tests/test_user_rag_stage5_1.py – Stage 5.1: truthful user-brand RAG corpus.

Verifies that POST-equivalent genome initialization (src.api.genome_service.
initialize_user_genome) materializes exactly 8 real user source texts
(mission + 7 snippets) into brand_texts/brand_texts_raw, chunks them with the
same deterministic packer used for competitors, and that the Stage 5 RAG
builder/retriever naturally include the user brand without any special
casing — all while leaving the competitor corpus (657 chunks / 10 brands)
completely untouched.

Uses temp copies of the canonical DB (never the checked-in data/brand_data.db)
and mocked embeddings for all but one opt-in real-model smoke test.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from src.api.genome_service import (
    USER_BRAND_ALIAS,
    USER_BRAND_DB_ID,
    ensure_canonical_schema,
    initialize_user_genome,
    write_history_event,
)
from src.retrieval import rag_builder, rag_service
from src.retrieval.rag_builder import build_index
from src.retrieval.rag_service import RagError, retrieve_chunks

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


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch):
    monkeypatch.setattr("src.api.genome_service.get_embedding", _fake_embedding)
    monkeypatch.setattr(rag_builder, "get_embedding", _fake_embedding)
    monkeypatch.setattr(rag_service, "get_embedding", _fake_embedding)


@pytest.fixture()
def temp_db(tmp_path):
    db_path = tmp_path / "brand_data.db"
    shutil.copy2(CANONICAL_DB, db_path)
    return db_path


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


def _competitor_chunk_ids(db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT chunk_id FROM brand_chunks WHERE brand_id != ? ORDER BY chunk_id", (USER_BRAND_ALIAS,)
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


# ── User corpus materialization ────────────────────────────────────────────

def test_genome_init_creates_exactly_8_user_source_rows(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        n_texts = conn.execute(
            "SELECT COUNT(*) FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchone()[0]
        n_raw = conn.execute(
            "SELECT COUNT(*) FROM brand_texts_raw WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchone()[0]
        assert n_texts == 8
        assert n_raw == 8
    finally:
        conn.close()


def test_mission_stored_ex_and_seven_snippets(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        mission_rows = conn.execute(
            "SELECT text_id, text FROM brand_texts WHERE brand_id = ? AND source_type = 'genome_mission'",
            (USER_BRAND_ALIAS,),
        ).fetchall()
        snippet_rows = conn.execute(
            "SELECT text_id, text FROM brand_texts WHERE brand_id = ? AND source_type = 'genome_snippet'",
            (USER_BRAND_ALIAS,),
        ).fetchall()
        assert len(mission_rows) == 1
        assert mission_rows[0][0] == "user_brand__mission"
        assert mission_rows[0][1] == _MISSION
        assert len(snippet_rows) == 7
        assert {r[0] for r in snippet_rows} == {f"user_brand__snippet_{i:03d}" for i in range(1, 8)}
    finally:
        conn.close()


def test_user_brand_identity_correct(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        brand_row = conn.execute("SELECT designation FROM brands WHERE id = ?", (USER_BRAND_DB_ID,)).fetchone()
        assert brand_row[0] == _DESIGNATION
        brand_ids = conn.execute(
            "SELECT DISTINCT brand_id FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchall()
        assert brand_ids == [(USER_BRAND_ALIAS,)]
    finally:
        conn.close()


def test_source_ids_deterministic_across_reinit(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    ids1 = {r[0] for r in conn.execute(
        "SELECT text_id FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
    ).fetchall()}
    conn.close()

    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    ids2 = {r[0] for r in conn.execute(
        "SELECT text_id FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
    ).fetchall()}
    conn.close()

    assert ids1 == ids2


def test_source_text_faithfully_preserved(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        for i, snippet in enumerate(_SNIPPETS, start=1):
            row = conn.execute(
                "SELECT text FROM brand_texts WHERE text_id = ?", (f"user_brand__snippet_{i:03d}",)
            ).fetchone()
            assert row[0] == snippet
    finally:
        conn.close()


def test_every_user_source_produces_at_least_one_chunk(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        text_ids = {r[0] for r in conn.execute(
            "SELECT text_id FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchall()}
        chunked_text_ids = {r[0] for r in conn.execute(
            "SELECT DISTINCT text_id FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchall()}
        assert text_ids == chunked_text_ids
    finally:
        conn.close()


def test_chunk_ids_deterministic(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    ids1 = sorted(r[0] for r in conn.execute(
        "SELECT chunk_id FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,)
    ).fetchall())
    conn.close()

    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    ids2 = sorted(r[0] for r in conn.execute(
        "SELECT chunk_id FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,)
    ).fetchall())
    conn.close()

    assert ids1 == ids2


def test_every_user_chunk_links_to_a_user_source_text(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        source_ids = {r[0] for r in conn.execute(
            "SELECT text_id FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchall()}
        chunk_rows = conn.execute(
            "SELECT chunk_id, text_id FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchall()
        assert chunk_rows
        for chunk_id, text_id in chunk_rows:
            assert text_id in source_ids
    finally:
        conn.close()


def test_no_user_chunk_belongs_to_a_competitor_and_competitor_untouched(temp_db):
    before_ids = _competitor_chunk_ids(temp_db)
    conn = sqlite3.connect(temp_db)
    before_texts = conn.execute(
        "SELECT text_id, chunk_text FROM brand_chunks WHERE chunk_id = 'rolex_001__chunk_000'"
    ).fetchone()
    conn.close()

    _init_genome(temp_db)

    after_ids = _competitor_chunk_ids(temp_db)
    conn = sqlite3.connect(temp_db)
    after_texts = conn.execute(
        "SELECT text_id, chunk_text FROM brand_chunks WHERE chunk_id = 'rolex_001__chunk_000'"
    ).fetchone()
    n_competitor_texts = conn.execute(
        "SELECT COUNT(*) FROM brand_texts WHERE brand_id != ?", (USER_BRAND_ALIAS,)
    ).fetchone()[0]
    n_competitor_chunks = conn.execute(
        "SELECT COUNT(*) FROM brand_chunks WHERE brand_id != ?", (USER_BRAND_ALIAS,)
    ).fetchone()[0]
    user_chunk_brands = {r[0] for r in conn.execute(
        "SELECT DISTINCT brand_id FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,)
    ).fetchall()}
    conn.close()

    assert before_ids == after_ids
    assert before_texts == after_texts
    assert n_competitor_texts == 450
    assert n_competitor_chunks == 657
    assert user_chunk_brands == {USER_BRAND_ALIAS}


def test_user_chunk_hard_max_400_chars(temp_db):
    _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        lengths = [r[0] for r in conn.execute(
            "SELECT char_count FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchall()]
        assert lengths
        assert all(length <= 400 for length in lengths)
    finally:
        conn.close()


def test_reinit_replaces_old_user_source_and_chunk_rows(temp_db):
    _init_genome(temp_db, mission="Original mission statement text.")
    conn = sqlite3.connect(temp_db)
    old_mission = conn.execute(
        "SELECT text FROM brand_texts WHERE text_id = 'user_brand__mission'"
    ).fetchone()[0]
    conn.close()
    assert old_mission == "Original mission statement text."

    _init_genome(temp_db, mission="Updated mission statement text, now different.")
    conn = sqlite3.connect(temp_db)
    try:
        rows = conn.execute(
            "SELECT text FROM brand_texts WHERE text_id = 'user_brand__mission'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "Updated mission statement text, now different."

        n_texts = conn.execute(
            "SELECT COUNT(*) FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchone()[0]
        n_chunks_docs = conn.execute(
            "SELECT COUNT(DISTINCT text_id) FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchone()[0]
        assert n_texts == 8
        assert n_chunks_docs == 8
    finally:
        conn.close()


def test_reinit_does_not_accumulate_duplicate_rows(temp_db):
    for _ in range(3):
        _init_genome(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        n_texts = conn.execute(
            "SELECT COUNT(*) FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchone()[0]
        n_raw = conn.execute(
            "SELECT COUNT(*) FROM brand_texts_raw WHERE brand_id = ?", (USER_BRAND_ALIAS,)
        ).fetchone()[0]
        assert n_texts == 8
        assert n_raw == 8
    finally:
        conn.close()


def test_analysis_history_survives_reinit(temp_db):
    conn = _connect(temp_db)
    write_history_event(conn, brand_id=USER_BRAND_DB_ID, event_type="consistency", input_text="hello")
    conn.close()

    conn = sqlite3.connect(temp_db)
    before_count = conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
    conn.close()
    assert before_count >= 1

    _init_genome(temp_db)

    conn = sqlite3.connect(temp_db)
    after_count = conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
    conn.close()
    assert after_count == before_count


# ── Index / retrieval ───────────────────────────────────────────────────────

def test_index_builder_includes_user_brand_with_correct_counts(temp_db, tmp_path):
    _init_genome(temp_db)
    out_dir = tmp_path / "rag"
    manifest = build_index(str(temp_db), str(out_dir))

    assert USER_BRAND_ALIAS in manifest["brands"]
    assert manifest["brands"][USER_BRAND_ALIAS]["count"] == 8

    conn = sqlite3.connect(temp_db)
    competitor_counts = dict(conn.execute(
        "SELECT brand_id, COUNT(*) FROM brand_chunks WHERE brand_id != ? GROUP BY brand_id", (USER_BRAND_ALIAS,)
    ).fetchall())
    conn.close()
    for brand_id, count in competitor_counts.items():
        assert manifest["brands"][brand_id]["count"] == count


def test_user_index_ntotal_matches_metadata_count(temp_db, tmp_path):
    _init_genome(temp_db)
    out_dir = tmp_path / "rag"
    manifest = build_index(str(temp_db), str(out_dir))

    info = manifest["brands"][USER_BRAND_ALIAS]
    index = rag_builder.load_brand_index(str(out_dir), USER_BRAND_ALIAS, info["index_file"])
    assert index.ntotal == info["count"]
    assert index.d == 384


def test_fingerprint_changes_after_genome_change(temp_db, tmp_path):
    _init_genome(temp_db, mission="First mission version for fingerprinting.")
    out_dir = tmp_path / "rag"
    manifest1 = build_index(str(temp_db), str(out_dir))

    _init_genome(temp_db, mission="Second, different mission version.")
    manifest2 = build_index(str(temp_db), str(out_dir))

    assert manifest1["fingerprint"] != manifest2["fingerprint"]


def test_stale_index_after_reinit_then_valid_after_rebuild(temp_db, tmp_path):
    _init_genome(temp_db, mission="Mission before reinit.")
    out_dir = tmp_path / "rag"
    build_index(str(temp_db), str(out_dir))

    # Sanity: retrieval works against the fresh index.
    result = retrieve_chunks("precision", USER_BRAND_ALIAS, artifact_dir=str(out_dir), db_path=str(temp_db))
    assert result["results"]

    # Genome reinit changes user chunks -> index becomes stale.
    _init_genome(temp_db, mission="Mission after reinit, changed content.")
    with pytest.raises(RagError) as exc_info:
        retrieve_chunks("precision", USER_BRAND_ALIAS, artifact_dir=str(out_dir), db_path=str(temp_db))
    assert exc_info.value.detail["error"] == "index_stale"

    # Rebuilding restores validity.
    build_index(str(temp_db), str(out_dir))
    result2 = retrieve_chunks("precision", USER_BRAND_ALIAS, artifact_dir=str(out_dir), db_path=str(temp_db))
    assert result2["results"]


def test_user_retrieval_returns_only_user_chunks_and_top_k_5(temp_db, tmp_path):
    _init_genome(temp_db)
    out_dir = tmp_path / "rag"
    build_index(str(temp_db), str(out_dir))

    result = retrieve_chunks("precision and craftsmanship", USER_BRAND_ALIAS, top_k=5, artifact_dir=str(out_dir), db_path=str(temp_db))
    assert len(result["results"]) == 5
    assert all(item["brand_id"] == USER_BRAND_ALIAS for item in result["results"])
    chunk_ids = {item["chunk_id"] for item in result["results"]}
    assert all(cid.startswith("user_brand__") for cid in chunk_ids)


def test_user_retrieval_chunk_text_matches_source_provenance(temp_db, tmp_path):
    _init_genome(temp_db)
    out_dir = tmp_path / "rag"
    build_index(str(temp_db), str(out_dir))

    result = retrieve_chunks("precision and craftsmanship", USER_BRAND_ALIAS, artifact_dir=str(out_dir), db_path=str(temp_db))

    conn = sqlite3.connect(temp_db)
    source_texts = dict(conn.execute(
        "SELECT text_id, text FROM brand_texts WHERE brand_id = ?", (USER_BRAND_ALIAS,)
    ).fetchall())
    conn.close()

    for item in result["results"]:
        assert item["chunk_text"] in source_texts[item["text_id"]]


def test_zero_competitor_leakage_in_user_query_and_vice_versa(temp_db, tmp_path):
    _init_genome(temp_db)
    out_dir = tmp_path / "rag"
    build_index(str(temp_db), str(out_dir))

    user_result = retrieve_chunks("precision", USER_BRAND_ALIAS, artifact_dir=str(out_dir), db_path=str(temp_db))
    assert {r["brand_id"] for r in user_result["results"]} == {USER_BRAND_ALIAS}

    rolex_result = retrieve_chunks("precision", "rolex", artifact_dir=str(out_dir), db_path=str(temp_db))
    assert {r["brand_id"] for r in rolex_result["results"]} == {"rolex"}


# ── Real-model integration smoke (opt-in) ───────────────────────────────────

@pytest.mark.requires_model
def test_real_model_user_corpus_smoke(temp_db, tmp_path, monkeypatch):
    from src.feature_extraction.embedding_extractor import get_embedding as real_get_embedding

    monkeypatch.setattr("src.api.genome_service.get_embedding", real_get_embedding)
    monkeypatch.setattr(rag_builder, "get_embedding", real_get_embedding)
    monkeypatch.setattr(rag_service, "get_embedding", real_get_embedding)

    _init_genome(temp_db)
    out_dir = tmp_path / "rag"
    manifest = build_index(str(temp_db), str(out_dir))
    assert manifest["brands"][USER_BRAND_ALIAS]["count"] == 8

    result = retrieve_chunks("precision and craftsmanship", USER_BRAND_ALIAS, top_k=5, artifact_dir=str(out_dir), db_path=str(temp_db))
    assert len(result["results"]) == 5
    assert all(r["brand_id"] == USER_BRAND_ALIAS for r in result["results"])
