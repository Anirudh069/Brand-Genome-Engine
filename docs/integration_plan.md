# Integration Plan — Dormant Files into Brand Genome Engine

**Date:** 2025-03-02  
**Commit baseline:** `359f6c3` (main)  
**Status:** Plan only — no code changes

---

## 1. Executive Summary

Three dormant files exist in the repository that were never wired into the active codebase:

| File | Purpose | Why it matters |
|------|---------|----------------|
| `scripts/brand_profile_builder (1).py` | Builds per-brand statistical profiles from raw `brand_texts` in SQLite | Without it, `/api/profile` falls back to hardcoded `_FALLBACK_PROFILES`; real data never reaches the scorer |
| `src/api/consistency_scorer (1).py` | Statistically rigorous scorer (Gaussian similarity, Jaccard vocab, cosine embedding tone) | Current `scoring.py` uses simpler absolute-difference math; this is the spec-compliant replacement |
| `tests/test_scoring (1).py` | 15 unit tests for the consistency scorer | Needed to gate quality of the new scorer before it replaces the current one |

The companion spec `docs/scoring_spec (1).md` defines the mathematical contract these files implement.

Integrating them will close the loop: **raw text → statistical profiles → spec-compliant scoring → tested pipeline**.

---

## 2. Current Architecture (As-Is)

```
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│  brand_data.db  │      │  features.parquet     │      │  FAISS index        │
│  (root, 10      │      │  (75 rows, 5 brands,  │      │  (5 brand centroids │
│   brands,       │      │   no-underscore IDs)  │      │   no-underscore IDs)│
│   150 texts)    │      └──────────┬───────────┘      └─────────┬───────────┘
└────────┬────────┘                 │                             │
         │                          │                             │
         │ (NOT CONNECTED)          │ pandas read                 │ load_index()
         ▼                          ▼                             ▼
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│ brand_profile_  │      │   main.py            │      │ retrieval.py        │
│ builder (1).py  │      │   v2.0.0             │◄─────│ query()             │
│ (DORMANT)       │      │                      │      └─────────────────────┘
└─────────────────┘      │  _FALLBACK_PROFILES  │
                         │  (5 hardcoded brands) │
                         │                      │
                         │  scoring.py (active)  │
                         │  simple diff math     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  React Frontend      │
                         │  (underscore IDs)    │
                         └──────────────────────┘
```

### Key pain points

1. **Two SQLite databases** — root `brand_data.db` has the actual texts (150 rows, 10 brands); `data/brand_data.db` has an empty `brand_profiles` table. `main.py` defaults to `data/brand_data.db`.
2. **Brand ID format mismatch** — parquet/FAISS use `tagheuer`, `patekphillipe`; DB/frontend/fallbacks use `tag_heuer`, `patek_phillipe`.
3. **No `brand_profiles` data** — the profile builder was never run; the API falls back to hardcoded dicts.
4. **Two competing scorers** — `scoring.py` (active, simple) vs `consistency_scorer (1).py` (dormant, spec-compliant).
5. **Test import path wrong** — `test_scoring (1).py` imports `src.scoring.consistency_scorer`; actual module path would be `src.api.consistency_scorer`.

---

## 3. Target Architecture (To-Be)

```
┌─────────────────┐
│  brand_data.db  │  (single canonical DB at project root)
│  10 brands,     │
│  150 texts      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  brand_profile_builder.py   │  (renamed, no space)
│  reads brand_texts          │
│  writes brand_profiles      │
│  uses canonical ID format   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│  brand_profiles │      │  features.parquet     │      │  FAISS index        │
│  table (SQLite) │      │  (re-extracted for    │      │  (rebuilt, all 10   │
│  10 brands      │      │   all 10 brands)      │      │   brands)           │
└────────┬────────┘      └──────────┬───────────┘      └─────────┬───────────┘
         │                          │                             │
         ▼                          ▼                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  main.py v3.0.0                                                         │
│  • Loads profiles from DB (brand_profiles table), no more fallbacks     │
│  • Imports consistency_scorer.py (renamed, spec-compliant)              │
│  • ID normaliser layer: underscore ↔ no-underscore                      │
│  • All 10 brands available                                              │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  React Frontend      │
│  (unchanged)         │
└──────────────────────┘
```

---

## 4. Data Alignment Plan

### 4.1 Canonical brand ID format

Choose **underscore** (`tag_heuer`, `patek_phillipe`) as canonical — it matches the frontend, the SQLite texts, and the existing API fallbacks.

| Layer | Current format | Action |
|-------|---------------|--------|
| `brand_data.db` brand_texts | `tag_heuer` ✓ | No change |
| `features.parquet` | `tagheuer` ✗ | Re-extract with underscore IDs, OR add normaliser |
| FAISS `metadata.json` | `tagheuer` ✗ | Rebuild index after parquet fix |
| `_FALLBACK_PROFILES` | `tag_heuer` ✓ | Will be replaced by DB profiles |
| Frontend constants | `tag_heuer` ✓ | No change |

**Recommended:** Add a `normalise_brand_id(raw_id) → str` utility that canonicalises any variant to underscore format. Apply it at every data boundary (parquet read, index load, API input).

### 4.2 Single database

- **Keep** root `brand_data.db` as the single source.
- **Add** `brand_profiles` table to it (the builder script already does this).
- **Update** `SQLITE_DB_PATH` default in `main.py` from `data/brand_data.db` → `brand_data.db`.
- **Delete** or gitignore `data/brand_data.db` to remove confusion.

### 4.3 Complete the parquet / FAISS for all 10 brands

- Run `scripts/run_feature_extraction.py` on all 10 brands (currently only 5 have rows in parquet).
- Rebuild FAISS index via `scripts/build_embeddings_index.py`.

---

## 5. File-by-File Integration Steps

### 5.1 `scripts/brand_profile_builder (1).py` → `scripts/brand_profile_builder.py`

**What it does:** Reads all rows from `brand_texts`, groups by `brand_id`, computes per-brand aggregates (mean/std of sentiment, flesch, formality, vocab_richness; top keywords; tone label), writes JSON profile into `brand_profiles` table.

**Changes needed:**

| # | Change | Reason |
|---|--------|--------|
| 1 | Rename file to remove ` (1)` | Spaces in Python filenames break imports and shell scripts |
| 2 | Update `DB_PATH` default to `brand_data.db` (root) | Points to the DB that actually has `brand_texts` |
| 3 | Normalise brand IDs to underscore format | Canonical ID consistency |
| 4 | Add `mean_embedding` computation (384-d) | `consistency_scorer` expects it for cosine tone scoring |
| 5 | Align profile JSON keys with the scorer that will consume them | If we adopt `consistency_scorer`, keep `mean_sentiment`/`std_sentiment` format; if we keep `scoring.py`, map to `avg_sentiment`/`avg_formality` |
| 6 | Add CLI `--db-path` argument | Flexibility for CI/Docker |
| 7 | Add `if __name__ == "__main__"` guard | Already exists — verify |
| 8 | Add logging | Replace bare `print()` calls |

**Dependency:** Must run *after* `run_feature_extraction.py` populates `brand_texts` (already done for 10 brands in root DB).

### 5.2 `src/api/consistency_scorer (1).py` → `src/api/consistency_scorer.py`

**What it does:** Implements `ConsistencyScorer` class with `score(text, brand_profile) → ScoreResult` dataclass. Sub-scores: Jaccard vocabulary overlap, Gaussian sentiment similarity, inverse-distance readability, cosine embedding tone.

**Changes needed:**

| # | Change | Reason |
|---|--------|--------|
| 1 | Rename file to remove ` (1)` | Import compatibility |
| 2 | Replace internal proxy functions (`_sentiment_proxy`, `_flesch_score`) with real extractors from `src.feature_extraction` | Eliminate code duplication; proxies are stubs |
| 3 | Add `normalise_brand_id()` call at entry point | ID consistency |
| 4 | Wire embedding extractor for `_cosine_tone_score` | Currently uses a placeholder; connect to `EmbeddingExtractor` singleton |
| 5 | Ensure `score()` return dict is JSON-serialisable | `ScoreResult` dataclass needs `.to_dict()` or API must convert |
| 6 | Add `generate_edit_plan()` (or keep the one from `scoring.py`) | Frontend expects `edit_plan` in response |
| 7 | Update `main.py` imports: `from src.api.consistency_scorer import ConsistencyScorer` | Replace `from src.api.scoring import score_consistency, generate_edit_plan` |
| 8 | Preserve backward-compatible response shape | `{"score": float, "details": {...}, "suggestions": [...]}` — the frontend depends on this |

**Decision required:** Two integration strategies:

- **Strategy A (Replace):** Delete `scoring.py`, make `consistency_scorer.py` the sole scorer, adapt `main.py` to its interface. *Pro:* cleaner, spec-compliant. *Con:* bigger diff, more risk.
- **Strategy B (Wrap):** Keep `scoring.py` as the API-facing adapter, have it internally delegate to `ConsistencyScorer`. *Pro:* minimal `main.py` changes. *Con:* extra indirection.

**Recommendation:** Strategy A — the scorer is the core of the system; it should be the spec-compliant version. Adapt `main.py` response serialisation to convert `ScoreResult` → dict matching current API contract.

### 5.3 `tests/test_scoring (1).py` → `tests/test_scoring.py`

**What it does:** 15 tests covering `ScoreResult` dataclass, custom exceptions (`BrandProfileNotFoundError`, `EmbeddingDimensionError`, `FeatureExtractionError`), Jaccard vocabulary score, Gaussian sentiment score, cosine tone with embeddings, overall score weighted composition.

**Changes needed:**

| # | Change | Reason |
|---|--------|--------|
| 1 | Rename file to remove ` (1)` | pytest collection |
| 2 | Fix import path: `from src.scoring.consistency_scorer import ...` → `from src.api.consistency_scorer import ...` | `src/scoring/` doesn't exist; module lives in `src/api/` |
| 3 | Add `requires_model` marker to tests that invoke the embedding model | Consistency with existing test strategy (skip without `--include-model-tests`) |
| 4 | Update any profile key references if builder output format changes | Test fixtures must match actual profile dict shape |
| 5 | Verify fixture profiles include `mean_embedding` (384-d numpy array) | Cosine tone tests need it |
| 6 | Add integration test: build profile → score → verify round-trip | End-to-end confidence |

---

## 6. Integration Order (Step-by-Step Checklist)

Execute in this exact order to maintain a green test suite at every step:

### Phase 1 — Preparation (no behavior change)

- [ ] **1.1** Create `src/utils/brand_id.py` with `normalise_brand_id(raw: str) → str` (lowercases, replaces hyphens/spaces with underscores, known aliases like `tagheuer` → `tag_heuer`)
- [ ] **1.2** Add unit tests for `normalise_brand_id` in `tests/test_brand_id.py`
- [ ] **1.3** Update `SQLITE_DB_PATH` default in `main.py` from `data/brand_data.db` to `brand_data.db`
- [ ] **1.4** Delete or `.gitignore` `data/brand_data.db` (empty/misleading)
- [ ] **1.5** Run existing test suite — expect 420 pass, 79 skip

### Phase 2 — Profile Builder

- [ ] **2.1** Rename `scripts/brand_profile_builder (1).py` → `scripts/brand_profile_builder.py`
- [ ] **2.2** Refactor: replace internal proxy functions with imports from `src.feature_extraction.*`
- [ ] **2.3** Add `normalise_brand_id()` to profile output
- [ ] **2.4** Add `mean_embedding` computation (average 384-d vectors per brand)
- [ ] **2.5** Add `--db-path` CLI argument
- [ ] **2.6** Run the builder: `python scripts/brand_profile_builder.py`
- [ ] **2.7** Verify: `sqlite3 brand_data.db "SELECT brand_id, length(profile_json) FROM brand_profiles"` → 10 rows
- [ ] **2.8** Run test suite — still green

### Phase 3 — Consistency Scorer

- [ ] **3.1** Rename `src/api/consistency_scorer (1).py` → `src/api/consistency_scorer.py`
- [ ] **3.2** Replace proxy functions with real extractor imports
- [ ] **3.3** Wire `EmbeddingExtractor` for cosine tone scoring
- [ ] **3.4** Add `normalise_brand_id()` at scorer entry
- [ ] **3.5** Add `.to_dict()` to `ScoreResult` dataclass (JSON-serialisable output)
- [ ] **3.6** Add/preserve `generate_edit_plan()` function
- [ ] **3.7** Update `main.py`:
  - Change import from `src.api.scoring` → `src.api.consistency_scorer`
  - Update `/api/check-consistency` to use `ConsistencyScorer.score()` + `.to_dict()`
  - Load profiles from `brand_profiles` table instead of `_FALLBACK_PROFILES`
  - Keep `_FALLBACK_PROFILES` as last-resort fallback only
- [ ] **3.8** Verify API response shape is unchanged (run `tests/test_api.py`)

### Phase 4 — Scorer Tests

- [ ] **4.1** Rename `tests/test_scoring (1).py` → `tests/test_scoring.py`
- [ ] **4.2** Fix import path → `src.api.consistency_scorer`
- [ ] **4.3** Add `requires_model` markers where needed
- [ ] **4.4** Update fixture profiles to match builder output format
- [ ] **4.5** Run `pytest tests/test_scoring.py -v` — all 15 pass
- [ ] **4.6** Run full suite — all pass

### Phase 5 — Data Completeness

- [ ] **5.1** Re-run `scripts/run_feature_extraction.py` to cover all 10 brands (currently only 5 in parquet)
- [ ] **5.2** Rebuild FAISS index: `python scripts/build_embeddings_index.py`
- [ ] **5.3** Re-run profile builder to pick up new data
- [ ] **5.4** Smoke-test: `curl -X POST localhost:8000/api/check-consistency -d '{"text":"...", "brand":"rolex"}'`
- [ ] **5.5** Smoke-test with all 10 brands
- [ ] **5.6** Run full test suite one final time

### Phase 6 — Cleanup

- [ ] **6.1** Delete `src/api/scoring.py` (replaced by `consistency_scorer.py`)
- [ ] **6.2** Delete dormant originals: `scripts/brand_profile_builder (1).py`, `src/api/consistency_scorer (1).py`, `tests/test_scoring (1).py`
- [ ] **6.3** Update `docs/scoring_spec (1).md` → `docs/scoring_spec.md` (rename, remove `(1)`)
- [ ] **6.4** Update `README.md` with new architecture / run instructions
- [ ] **6.5** Git commit and push

---

## 7. Breaking Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **API response shape change** breaks frontend | 🔴 High | Write a response-shape assertion test FIRST (`test_api.py` already covers this); run it after every scorer change |
| **Brand ID mismatch** causes empty profiles / 404s | 🔴 High | `normalise_brand_id()` utility applied at every boundary; add assertion in profile loader: `assert len(profiles) == 10` |
| **Embedding model not available** in CI | 🟡 Medium | `requires_model` marker + `--include-model-tests` flag (already implemented) |
| **Profile builder computes different stats** than scorer expects | 🟡 Medium | Add a contract test: load a built profile, pass to scorer, assert no `KeyError` |
| **Parquet only has 5 brands** | 🟡 Medium | Phase 5 re-extraction; but until then, profile builder uses `brand_texts` directly (all 10 brands), so profiles are complete |
| **Two DB files cause confusion** | 🟢 Low | Phase 1.4 removes `data/brand_data.db`; single source of truth |
| **File rename breaks git history** | 🟢 Low | Use `git mv` for renames to preserve history |

---

## 8. Testing Plan

### Unit tests (fast, no model)
- `test_brand_id.py` — normaliser edge cases (NEW)
- `test_scoring.py` — scorer math with mocked extractors (ADAPTED from dormant file)
- `test_api.py` — API contract tests (EXISTING, 40 tests)

### Integration tests (need model, gated by `--include-model-tests`)
- `test_scoring.py` — cosine tone with real embeddings (subset of 15 tests)
- `test_embedding_extractor.py` — real model load (EXISTING)
- NEW: `test_profile_round_trip.py` — build profile → score text → verify structure

### Smoke tests (manual or CI script)
- Start server, hit every endpoint with `curl`/`httpx`
- Compare response shapes to frontend expectations
- Verify all 10 brands return non-fallback profiles

### Regression gate
- Full suite must pass (`pytest` → 420+ pass, 79 skip, 0 fail) at every phase boundary

---

## 9. Profile Key Mapping Reference

For clarity, here is the mapping between the two profile formats:

| Dormant format (scorer/builder) | Active format (scoring.py/fallbacks) | Parquet column |
|---------------------------------|--------------------------------------|----------------|
| `mean_sentiment` | `avg_sentiment` | `sentiment` |
| `std_sentiment` | *(not present)* | — |
| `mean_flesch` | `avg_readability_flesch` | `readability_flesch` |
| `std_flesch` | *(not present)* | — |
| `mean_formality` | `avg_formality` | `formality` |
| `vocab_richness` | `vocabulary_richness` | `vocab_diversity` |
| `top_keywords` | `top_keywords` | `top_topics` |
| `tone_label` | *(not present)* | — |
| `mean_embedding` (384-d) | *(not present)* | `embedding` (384-d) |

**Decision:** Adopt the dormant format (richer, includes std deviations needed for Gaussian scoring). Update API serialisation layer to translate if frontend expects different keys.

---

## 10. Estimated Effort

| Phase | Estimated time | Files touched |
|-------|---------------|---------------|
| Phase 1 — Preparation | 30 min | 3 new, 1 edit |
| Phase 2 — Profile Builder | 45 min | 1 rename + edit, 1 verify |
| Phase 3 — Consistency Scorer | 60 min | 1 rename + edit, 1 major edit (main.py) |
| Phase 4 — Scorer Tests | 30 min | 1 rename + edit |
| Phase 5 — Data Completeness | 20 min | scripts only (run existing) |
| Phase 6 — Cleanup | 15 min | deletes + renames |
| **Total** | **~3.5 hours** | |

---

## 11. Open Questions

1. **Should `_FALLBACK_PROFILES` be kept as a safety net or fully removed?** Recommendation: keep but log a warning when used.
2. **Should the parquet be re-extracted with underscore IDs, or should the normaliser handle it at read time?** Recommendation: normaliser at read time (less data churn).
3. **Should we add a `/api/rebuild-profiles` endpoint that runs the profile builder on demand?** The endpoint stub exists in `main.py` (`/api/rebuild`); we could wire it to the real builder.
4. **Should the scoring spec doc be versioned (v1 → v2) or just updated in place?** Recommendation: rename to `scoring_spec.md` and note the version at the top.

---

*This document is the deliverable for Phase B. No code changes have been made. Proceed to implementation only after review.*
