# src/scoring/grounding.py
# Retrieval of grounding chunks for the rewrite prompt.
#
# LANE NOTE: Person B owns RAG retrieval (FAISS over brand_chunks embeddings).
# This module defines the interface Person C's edit plan needs and ships a
# lexical fallback so the pipeline runs before the FAISS index exists. Pass
# Person B's retrieval function as `retriever` and it is used instead:
#
#     retrieve_grounding_chunks(text, "rolex", retriever=b.retrieve_top_k)
#
# The retriever contract is:  retriever(query_text, brand_id, k) -> [str, ...]

import math
import sqlite3
from collections import Counter

from src.profiles.brand_profile_builder import _content_words, SQLITE_DB_PATH

DEFAULT_K = 3
MIN_CHUNK_CHARS = 60


def _load_chunks(brand_id, db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT chunk_text FROM brand_chunks WHERE brand_id = ?", (brand_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [r[0] for r in rows if r[0] and len(r[0]) >= MIN_CHUNK_CHARS]


def _lexical_score(chunk, query_words, brand_keywords):
    """
    Rank a chunk as a rewrite example.

        0.6 * topical overlap with the input text
      + 0.4 * density of the brand's characteristic vocabulary

    Topical overlap keeps the example relevant to what the writer is writing
    about; brand-vocabulary density keeps it a good specimen of the voice.
    Weighting relevance higher than exemplarity matters because an example
    about a different product teaches the LLM the wrong content.
    """
    chunk_words = set(_content_words(chunk))
    if not chunk_words:
        return 0.0

    overlap = len(chunk_words & query_words) / math.sqrt(len(chunk_words))
    kw_density = (len(chunk_words & set(brand_keywords)) / len(chunk_words)) * 4.0

    return 0.6 * overlap + 0.4 * kw_density


def retrieve_grounding_chunks(query_text, brand_id, k=DEFAULT_K,
                              brand_profile=None, db_path=SQLITE_DB_PATH,
                              retriever=None):
    """
    Return up to `k` brand chunks to ground an LLM rewrite.

    If `retriever` is supplied (Person B's FAISS retrieval), it is used and its
    output returned verbatim. Otherwise a lexical ranking over brand_chunks is
    applied. Returns [] when the brand has no chunks — callers must handle an
    empty list rather than assuming grounding is always available.
    """
    if retriever is not None:
        try:
            return list(retriever(query_text, brand_id, k))[:k]
        except Exception:
            # A failing retriever must not take the rewrite endpoint down;
            # fall through to the lexical path.
            pass

    chunks = _load_chunks(brand_id, db_path)
    if not chunks:
        return []

    query_words = set(_content_words(query_text))
    brand_keywords = (brand_profile or {}).get("top_keywords") or []

    ranked = sorted(
        chunks,
        key=lambda c: _lexical_score(c, query_words, brand_keywords),
        reverse=True,
    )

    # De-duplicate near-identical chunks so the LLM sees three different
    # examples rather than the same sentence three ways.
    selected = []
    seen = []
    for chunk in ranked:
        words = set(_content_words(chunk))
        if any(len(words & prev) / max(1, len(words | prev)) > 0.6 for prev in seen):
            continue
        selected.append(chunk)
        seen.append(words)
        if len(selected) >= k:
            break

    return selected
