# Database — Brand Genome Engine (Final / Stage 9)

Canonical DB: **`data/brand_data.db`** (SQLite, the only physical `.db` file
in the repo — enforced by `scripts/validate_db.py` / `validate_chunk_handoff.py`
/ `verify_phase4.py`, and by `!data/brand_data.db` in `.gitignore`).

## Tables

| Table | Rows (baseline) | Purpose |
|---|---|---|
| `brands` | 1 (`id=0`) | Runtime **user** identity: designation, mission/core vision. |
| `brand_profile` | 0 (until genome init) | Active **user** genome: keywords, tone/NLP features, aggregate MiniLM embedding. `brand_id=0` maps to `brands.id`. |
| `brand_texts_raw` | 450 competitor rows | Raw ingested competitor ad copy (+ segment/country/category/year provenance columns). |
| `brand_texts` | 450 competitor rows | Cleaned competitor texts derived 1:1 from `brand_texts_raw`. Also holds the user's 8 source rows (`user_brand__mission`, `user_brand__snippet_001..007`) once genome is initialised. |
| `brand_chunks` | 657 competitor rows | `<=400` char chunks of `brand_texts`, one row per chunk, `chunk_id` unique, FK-like link via `text_id`. Also holds user_brand chunks once genome is initialised (no `MIN_CHUNKS_PER_BRAND` floor for the user). |
| `brand_profiles` | 10 | **Competitor** aggregate profiles (one per competitor brand): keywords, mean sentiment/formality/Flesch, aggregate embedding. |
| `analysis_history` | 0 (baseline) | Append-only log of consistency / benchmark / rewrite events (`event_type`, `pre_score`, `post_score`, `diagnostics_json`, `extra_json`). |

## The critical identity distinction

- **`brand_profiles`** (plural) = **competitor** profiles (10 rows, one per
  luxury-watch brand). Read-only reference data for Benchmarking/Analytics.
- **`brand_profile`** (singular) = the **active user genome** (at most one
  row per app instance in this single-user PoC). Written by
  `POST /api/genome/init`, read by Consistency/Rewrite/Analytics.
- **`brands.id = 0`** is the relational identity of the current user for
  joins with `brand_profile`/`analysis_history`.
- **`user_brand`** is the *text/chunk/RAG* identifier used as `brand_id` in
  `brand_texts`, `brand_texts_raw`, and `brand_chunks` (string, not integer,
  matching the competitor brand_id convention in those three tables) and as
  the brand scope passed to `POST /api/rag/retrieve` / used internally by
  Rewrite.

No runtime code conflates these — see `src/api/genome_service.py`
(`ensure_canonical_schema`, `initialize_user_genome`) and
`src/rewrite/rewrite_service.py` for the concrete read/write paths.

## Provenance & lineage

```
brand_texts_raw --(1:1 clean)--> brand_texts --(chunk, <=400 chars)--> brand_chunks
                                       |
                                       +--(aggregate features, competitors only)--> brand_profiles
brand_chunks --(MiniLM embed)--> data/processed/rag/*.faiss (per brand_id)
```

## Expected baseline (clean checked-in DB, no user genome yet)

```
integrity_check = ok
brand_texts        = 450   (45 per competitor x 10)
brand_texts_raw     = 450
brand_chunks        = 657   (51-87 per competitor, all >= 50)
brand_profiles      = 10
brand_profile       = 0    (expected — user genome not yet initialised)
analysis_history    = 0    (expected — no runtime activity yet)
```

Verified by `scripts/validate_db.py`, `scripts/validate_chunk_handoff.py`,
and `scripts/verify_phase4.py`.
