"""
Tests for scripts/build_brand_chunks.py

Covers deterministic chunk generation, hard-constraint validation,
transaction safety, and idempotence. Uses temporary SQLite DBs / pure
in-memory fixtures only — never touches the real canonical database.
"""

from __future__ import annotations

import sqlite3

import pytest

from scripts.build_brand_chunks import (
    HARD_MAX_CHARS,
    MAX_SENTENCES,
    OversizedTokenError,
    build_candidates,
    load_source_rows,
    pack_text_into_chunks,
    replace_chunks,
    run,
    split_long_sentence,
    validate_candidates,
)

BRAND_CHUNKS_SCHEMA = """
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

BRAND_TEXTS_SCHEMA = """
CREATE TABLE "brand_texts" (
    "text_id"   TEXT,
    "brand_id"  TEXT,
    "brand_name" TEXT,
    "source_type" TEXT,
    "text"      TEXT
, created_at TEXT)
"""


def _row(text_id, brand_id, brand_name, source_type, text):
    return {
        "text_id": text_id,
        "brand_id": brand_id,
        "brand_name": brand_name,
        "source_type": source_type,
        "text": text,
    }


def make_db(tmp_path, rows, db_name="test.db"):
    db_path = str(tmp_path / db_name)
    conn = sqlite3.connect(db_path)
    conn.execute(BRAND_TEXTS_SCHEMA)
    conn.execute(BRAND_CHUNKS_SCHEMA)
    conn.executemany(
        "INSERT INTO brand_texts (text_id, brand_id, brand_name, source_type, text) "
        "VALUES (?, ?, ?, ?, ?)",
        [(r["text_id"], r["brand_id"], r["brand_name"], r["source_type"], r["text"]) for r in rows],
    )
    conn.commit()
    conn.close()
    return db_path


def make_brand_rows(brand_id, brand_name, n_texts, sentence_template=None):
    """Generate n_texts synthetic multi-sentence source rows for a brand."""
    rows = []
    for i in range(n_texts):
        text = (
            f"Sentence one about {brand_id} craftsmanship number {i}. "
            f"Sentence two describing heritage and precision for item {i}. "
            f"Sentence three on design philosophy and materials used {i}."
        )
        rows.append(_row(f"{brand_id}_{i:03d}", brand_id, brand_name, "website_copy", text))
    return rows


# ── 1. Deterministic chunk generation ──────────────────────────────────────

def test_deterministic_chunk_generation():
    rows = make_brand_rows("rolex", "Rolex", 5)
    result1 = build_candidates(rows)
    result2 = build_candidates(rows)
    texts1 = [(c["chunk_id"], c["chunk_text"]) for c in result1.candidates]
    texts2 = [(c["chunk_id"], c["chunk_text"]) for c in result2.candidates]
    assert texts1 == texts2
    assert len(result1.candidates) > 0


# ── 2. Stable deterministic chunk IDs ──────────────────────────────────────

def test_chunk_ids_are_stable_and_use_expected_format():
    rows = [_row("rolex_004", "rolex", "Rolex", "website_copy",
                  "First sentence here. Second sentence follows. Third one too.")]
    result = build_candidates(rows)
    for c in result.candidates:
        assert c["chunk_id"].startswith("rolex_004__chunk_")
    ids = [c["chunk_id"] for c in result.candidates]
    assert ids == sorted(ids)  # index-ordered
    assert ids[0] == "rolex_004__chunk_000"


# ── 3. Hard maximum <= 400 characters ──────────────────────────────────────

def test_hard_maximum_never_exceeded():
    long_sentence = "word" + (" filler") * 200 + "."  # long single "sentence"
    rows = [_row("t1", "b1", "Brand", "website_copy", long_sentence)]
    result = build_candidates(rows)
    assert result.errors == []
    for c in result.candidates:
        assert c["char_count"] <= HARD_MAX_CHARS
        assert len(c["chunk_text"]) <= HARD_MAX_CHARS


# ── 4. Sentence packing respects <= 4 sentences ────────────────────────────

def test_sentence_packing_respects_max_sentences():
    text = " ".join([f"Short sentence number {i}." for i in range(10)])
    rows = [_row("t1", "b1", "Brand", "website_copy", text)]
    result = build_candidates(rows)
    for c in result.candidates:
        n_sentences = c["chunk_text"].count(".") + c["chunk_text"].count("!") + c["chunk_text"].count("?")
        assert n_sentences <= MAX_SENTENCES


# ── 5 & 6. Oversized single sentence splits at word boundaries, no word loss ─

def test_oversized_sentence_splits_at_word_boundaries_without_losing_words():
    words = [f"word{i}" for i in range(150)]
    sentence = " ".join(words) + "."
    pieces = split_long_sentence(sentence, target_max=375, hard_max=400)

    assert all(len(p) <= 400 for p in pieces)
    # Rejoin and compare token sets (order preserved, nothing dropped)
    rejoined_words = " ".join(pieces).split(" ")
    original_words = sentence.split(" ")
    assert rejoined_words == original_words


def test_single_token_over_hard_max_raises():
    huge_token = "x" * 401
    with pytest.raises(OversizedTokenError):
        split_long_sentence(f"prefix {huge_token} suffix.", target_max=375, hard_max=400)


def test_oversized_token_source_reported_and_skipped_not_crashing_whole_run():
    huge_token = "y" * 500
    rows = [
        _row("bad_001", "b1", "Brand", "website_copy", f"Broken {huge_token} sentence."),
        _row("good_001", "b1", "Brand", "website_copy", "A perfectly normal short sentence."),
    ]
    result = build_candidates(rows)
    assert any(e["text_id"] == "bad_001" for e in result.errors)
    assert "bad_001" in result.zero_chunk_text_ids
    # the other source text is unaffected
    assert any(c["text_id"] == "good_001" for c in result.candidates)


# ── 7. Short source text is preserved ──────────────────────────────────────

def test_short_source_text_preserved_as_single_chunk():
    rows = [_row("t1", "b1", "Brand", "social_media", "Tiny.")]
    result = build_candidates(rows)
    assert len(result.candidates) == 1
    assert result.candidates[0]["chunk_text"] == "Tiny."
    assert any("short" in w.lower() for w in result.warnings)


# ── 8. Short final fragment handled without fabrication ────────────────────

def test_short_final_fragment_merged_when_possible():
    # Two sentences: a large one + a tiny trailing one that would be < 80
    # chars alone, but combining fits under HARD_MAX and <=4 sentences.
    big = "A" * 300 + "."
    tiny = "Ok."
    text = f"{big} {tiny}"
    rows = [_row("t1", "b1", "Brand", "website_copy", text)]
    result = build_candidates(rows)
    # Should be merged into a single chunk rather than left as a fabricated/padded fragment
    assert len(result.candidates) == 1
    assert result.candidates[0]["chunk_text"].endswith("Ok.")
    assert "AAAA" in result.candidates[0]["chunk_text"]


def test_short_final_fragment_preserved_when_merge_not_possible():
    # Four max-length-ish sentences already at MAX_SENTENCES, plus a short
    # trailing one that can't merge without breaking the sentence-count cap.
    s = "This is sentence content that is reasonably long for packing purposes today."
    text = f"{s} {s} {s} {s} Ok."
    rows = [_row("t1", "b1", "Brand", "website_copy", text)]
    result = build_candidates(rows)
    chunk_texts = [c["chunk_text"] for c in result.candidates]
    assert any(t.strip() == "Ok." for t in chunk_texts)
    assert any("short final chunk" in w for w in result.warnings)


# ── 9 & 10. text_id / brand_id preservation ────────────────────────────────

def test_text_id_and_brand_id_preserved():
    rows = [_row("abc_123", "cartier", "Cartier", "website_copy", "One sentence here.")]
    result = build_candidates(rows)
    assert result.candidates[0]["text_id"] == "abc_123"
    assert result.candidates[0]["brand_id"] == "cartier"
    assert result.candidates[0]["brand_name"] == "Cartier"
    assert result.candidates[0]["source_type"] == "website_copy"


# ── 11. No merging across separate source text IDs ─────────────────────────

def test_no_merging_across_different_text_ids():
    rows = [
        _row("t1", "b1", "Brand", "social_media", "Hi."),
        _row("t2", "b1", "Brand", "social_media", "Bye."),
    ]
    result = build_candidates(rows)
    assert len(result.candidates) == 2
    text_ids = {c["text_id"] for c in result.candidates}
    assert text_ids == {"t1", "t2"}
    for c in result.candidates:
        # each chunk_text belongs to exactly one source text's content
        assert c["chunk_text"] in ("Hi.", "Bye.")


# ── 12. char_count matches final text ──────────────────────────────────────

def test_char_count_matches_chunk_text_length():
    rows = make_brand_rows("omega", "Omega", 3)
    result = build_candidates(rows)
    for c in result.candidates:
        assert c["char_count"] == len(c["chunk_text"])


# ── 13. Blank text does not create a blank chunk ────────────────────────────

def test_blank_text_produces_no_chunks():
    rows = [
        _row("t1", "b1", "Brand", "website_copy", ""),
        _row("t2", "b1", "Brand", "website_copy", "   "),
        _row("t3", "b1", "Brand", "website_copy", None),
    ]
    result = build_candidates(rows)
    assert result.candidates == []
    assert result.zero_chunk_text_ids == []  # blank source -> not "unexpected zero"
    for c in result.candidates:
        assert c["chunk_text"].strip() != ""


# ── 14. Idempotent rebuild behavior ─────────────────────────────────────────

def test_idempotent_rebuild(tmp_path):
    rows = make_brand_rows("hublot", "Hublot", 4)
    db_path = make_db(tmp_path, rows)

    report1 = run(db_path, mode="replace", min_chunks_per_brand=1)
    assert report1["pass"]
    assert report1.get("written")

    conn = sqlite3.connect(db_path)
    first_rows = conn.execute(
        "SELECT chunk_id, text_id, brand_id, chunk_text FROM brand_chunks ORDER BY chunk_id"
    ).fetchall()
    conn.close()

    report2 = run(db_path, mode="replace", min_chunks_per_brand=1)
    assert report2["pass"]
    assert report2.get("written")

    conn = sqlite3.connect(db_path)
    second_rows = conn.execute(
        "SELECT chunk_id, text_id, brand_id, chunk_text FROM brand_chunks ORDER BY chunk_id"
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM brand_chunks").fetchone()[0]
    conn.close()

    assert first_rows == second_rows
    assert total == len(first_rows)  # no accumulation/duplication


# ── 15. Validation refuses a candidate set where a brand has <50 chunks ────

def test_validation_rejects_brand_below_minimum():
    rows = make_brand_rows("tissot", "Tissot", 2)  # far too few source rows for 50 chunks
    result = build_candidates(rows)
    failures = validate_candidates(result, rows, min_chunks_per_brand=50)
    assert any("tissot" in f and "50" in f for f in failures)


def test_validation_passes_with_relaxed_minimum():
    rows = make_brand_rows("tissot", "Tissot", 5)
    result = build_candidates(rows)
    failures = validate_candidates(result, rows, min_chunks_per_brand=1)
    assert failures == []


# ── 16. Failed candidate validation does NOT destructively clear brand_chunks ─

def test_failed_validation_does_not_touch_existing_brand_chunks(tmp_path):
    rows = make_brand_rows("iwc", "IWC", 2)  # will fail the >=50/brand rule
    db_path = make_db(tmp_path, rows)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO brand_chunks (chunk_id, text_id, brand_id, brand_name, source_type, chunk_text, char_count) "
        "VALUES ('preexisting__chunk_000', 'iwc_000', 'iwc', 'IWC', 'website_copy', 'Pre-existing chunk.', 19)"
    )
    conn.commit()
    conn.close()

    report = run(db_path, mode="replace", min_chunks_per_brand=50)
    assert not report["pass"]
    assert not report.get("written")

    conn = sqlite3.connect(db_path)
    remaining = conn.execute("SELECT chunk_id FROM brand_chunks").fetchall()
    conn.close()
    assert remaining == [("preexisting__chunk_000",)]


# ── 17. Transaction rolls back on simulated insertion failure ──────────────

def test_replace_chunks_rolls_back_on_integrity_error(tmp_path):
    db_path = str(tmp_path / "rollback.db")
    conn = sqlite3.connect(db_path)
    conn.execute(BRAND_CHUNKS_SCHEMA)
    conn.execute(
        "INSERT INTO brand_chunks (chunk_id, text_id, brand_id, brand_name, source_type, chunk_text, char_count) "
        "VALUES ('existing__chunk_000', 't0', 'b0', 'Brand0', 'website_copy', 'Existing.', 9)"
    )
    conn.commit()

    # Duplicate chunk_id in the candidate batch -> sqlite3.IntegrityError mid-insert
    bad_candidates = [
        {
            "chunk_id": "dup__chunk_000",
            "text_id": "t1",
            "brand_id": "b1",
            "brand_name": "Brand1",
            "source_type": "website_copy",
            "chunk_text": "Chunk A.",
            "char_count": 8,
        },
        {
            "chunk_id": "dup__chunk_000",  # collision -> triggers IntegrityError
            "text_id": "t1",
            "brand_id": "b1",
            "brand_name": "Brand1",
            "source_type": "website_copy",
            "chunk_text": "Chunk B.",
            "char_count": 8,
        },
    ]

    with pytest.raises(sqlite3.IntegrityError):
        replace_chunks(conn, bad_candidates)

    remaining = conn.execute("SELECT chunk_id FROM brand_chunks").fetchall()
    conn.close()
    # Original row must still be present -> DELETE was rolled back too.
    assert remaining == [("existing__chunk_000",)]


# ── Additional: load_source_rows / dry-run does not write ──────────────────

def test_dry_run_does_not_write_to_db(tmp_path):
    rows = make_brand_rows("patek_phillipe", "Patek Phillipe", 3)
    db_path = make_db(tmp_path, rows)

    report = run(db_path, mode="dry-run", min_chunks_per_brand=1)
    assert report["total_candidates"] > 0

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM brand_chunks").fetchone()[0]
    conn.close()
    assert count == 0


def test_load_source_rows_matches_inserted_rows(tmp_path):
    rows = make_brand_rows("breitling", "Breitling", 3)
    db_path = make_db(tmp_path, rows)
    conn = sqlite3.connect(db_path)
    loaded = load_source_rows(conn)
    conn.close()
    assert len(loaded) == len(rows)
    assert {r["text_id"] for r in loaded} == {r["text_id"] for r in rows}
