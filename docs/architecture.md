# Architecture — Brand Genome Engine (Final / Stage 9)

## Data flow

```
Genome Setup (designation + mission + exactly 7 snippets)
    -> SQLite user corpus (brand_texts/brand_texts_raw, brand_id='user_brand')
        + persisted feature/embedding profile (brand_profile, brand_id=0)
    -> Consistency scoring (src/scoring/consistency.py)

Competitor raw texts (brand_texts_raw)
    -> brand_texts (cleaned)
    -> brand_chunks (<=400 char chunks, scripts/build_brand_chunks.py)
    -> brand_profiles (per-competitor aggregate features, scripts/build_brand_profiles.py)
    -> Analytics (five pillars, TF-IDF terms, chunk t-SNE, tone distribution)

brand_chunks (all 11 "brands": 10 competitors + user_brand)
    -> MiniLM (all-MiniLM-L6-v2, 384-d, L2-normalized)
    -> one FAISS IndexFlatIP per brand_id (src/retrieval/rag_builder.py)
    -> data/processed/rag/ (manifest.json, metadata.json, indexes/<brand_id>.faiss)

Rewrite:
  input text
    -> pre-score (score_against_user_genome, same scorer as Consistency Check)
    -> edit plan (generate_edit_plan, flattened user genome features)
    -> RAG retrieval scoped to user_brand only (src/retrieval/rag_service.py)
    -> OpenAI (or FallbackRewriteProvider if no key) -> rewritten text
    -> post-score (same scorer)
    -> one analysis_history row (event_type='rewrite', pre/post score, retrieved chunk ids)
```

## Canonical tables (see [docs/database.md](database.md) for full detail)

`brands`, `brand_texts_raw`, `brand_texts`, `brand_chunks`, `brand_profiles`
(competitor profiles), `brand_profile` (active user genome), `analysis_history`.

## Derived / generated artifacts (gitignored, reproducible)

| Artifact | Built by | Consumed by |
|---|---|---|
| `data/processed/analytics_cache.json` | `scripts/build_analytics_cache.py` / `src/analytics/cache.py` | `GET /api/analytics` |
| `data/processed/rag/` (manifest + per-brand FAISS indexes + metadata) | `scripts/build_rag_index.py` / `src/retrieval/rag_builder.py` | `POST /api/rag/retrieve`, `POST /api/rewrite` |

Both are fingerprinted against the live DB (`brand_texts`+`brand_chunks` content
hash). A mismatched fingerprint is reported as stale/rejected rather than
silently served, and both can be rebuilt through `POST /api/rebuild/*` or the
CLI scripts above — never manually edited.

## Legacy / non-production components

`src/benchmarking/retrieval.py`, `scripts/build_embeddings_index.py`, and
`scripts/query_competitors.py` implement an older **brand-level centroid**
FAISS index (`embeddings/metadata.json`, 5 brands only). They are explicitly
labelled LEGACY in their module docstrings, are not imported by
`src/api/main.py`, and are not part of any canonical `/api/*` route. They are
retained only as a standalone CLI research tool (with its own test,
`tests/test_query_competitors_script.py`). The live product's competitor
semantic retrieval is the chunk-level RAG stack described above, which covers
all 10 competitors.

## Layers

- **Frontend**: React 18 + Vite, pages Genome Setup / Consistency Check /
  Benchmarking / Analytics / Rewrite / Dev Tools (`frontend/src/pages/`).
- **API**: FastAPI (`src/api/main.py`), lifespan-based startup, single SQLite
  connection per request via `get_db_connection()`.
- **NLP**: `src/feature_extraction/` (sentiment, formality, readability,
  vocabulary, topics, embeddings).
- **Scoring**: `src/scoring/consistency.py` (canonical, no side effects,
  used identically by Consistency Check and Rewrite pre/post scoring).
- **Benchmarking**: `src/benchmarking/market_benchmark.py` (DB-only, 3
  metrics: tone, sentiment, readability).
- **Analytics**: `src/analytics/` (pillars, chunk_tsne, tone, heatmap,
  history, cache).
- **Retrieval (RAG)**: `src/retrieval/` (rag_builder, rag_service).
- **Rewrite**: `src/rewrite/` (openai_provider, rewrite_service).
- **Rebuild**: `src/rebuild/rebuild_service.py` (profile/chunks/index,
  idempotent, safe to call repeatedly).
