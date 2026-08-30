# Phase 4 — Full-Stack Integration & Rewrite Agent

**Version:** 1.0  
**Date:** 2026-03-03  
**Baseline commit:** `ff16735` (main)  
**Status:** Plan — no code changes

---

## 1. Objectives

Phase 4 merges the remaining work into a single execution plan:

- **Genome Setup** — User configures a brand profile from scratch
  via the UI; system extracts features and stores an ephemeral
  profile that drives all downstream scoring.
- **Consistency Scoring** — Score arbitrary copy against the
  *user-defined* brand (default) rather than a competitor dropdown.
- **Market Benchmarking** — Real head-to-head comparisons against
  competitor profiles from the DB, driven by `brand_chunks` data.
- **Data Analytics** — Replace all placeholder charts with live
  data: tone histogram, t-SNE scatter, messaging-pillars heatmap.
- **RAG Retrieval** — Retrieve grounding chunks from `brand_chunks`
  for both user brand (by similarity) and competitors (by brand_id).
- **Rewrite Agent** — LLM-powered rewrite loop that proposes
  on-brand rewrites and re-scores iteratively.

---

## 2. Target Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         FRONTEND                             │
│  Genome Setup ─→ ConsistencyCheck ─→ Benchmarking ─→ Analytics│
└──────┬──────────────┬──────────────────┬────────────┬────────┘
       │              │                  │            │
       ▼              ▼                  ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI (main.py)                        │
│                                                               │
│  POST /api/user-brand/init ──→ UserProfileBuilder             │
│  GET  /api/user-brand/status                                  │
│  POST /api/check-consistency ─→ ConsistencyScorer             │
│  POST /api/benchmark ─────────→ BenchmarkEngine               │
│  GET  /api/analytics/* ───────→ AnalyticsEngine               │
│  POST /api/rewrite ───────────→ RewriteAgent                  │
└──────┬───────────────────────────────────────┬───────────────┘
       │                                       │
       ▼                                       ▼
┌────────────────────┐              ┌────────────────────────┐
│  Ephemeral Store   │              │  data/brand_data.db    │
│  (in-memory dict)  │              │  brand_profiles (10)   │
│  _user_brand_state │              │  brand_chunks (150)    │
│  reset on restart  │              │  brand_texts (150)     │
└────────────────────┘              └────────────────────────┘
                                             │
                                             ▼
                                   ┌────────────────────────┐
                                   │  FAISS Index           │
                                   │  brand_profile_index   │
                                   │  (5 brand centroids)   │
                                   └────────────────────────┘
```

### Data Flow Summary

1. **Genome Setup** → User submits name + mission + snippets →
   backend extracts features → stores ephemeral profile in
   `_user_brand_state`.
2. **Consistency Check** → Scores text against `_user_brand_state`
   (default "Your Brand") or a competitor profile from DB.
3. **Benchmarking** → Compares user profile vs one competitor
   using real `brand_chunks` features.
4. **Analytics** → Queries `brand_chunks` for tone/embedding data,
   computes t-SNE / histograms / heatmaps server-side.
5. **Rewrite** → Retrieves grounding chunks via RAG → builds
   prompt → calls LLM → re-scores rewritten text → returns
   before/after scores + suggestions.

---

## 3. Data Model

### 3.1 Ephemeral User Brand Profile

**Decision:** In-memory Python dict, keyed by a singleton
`"user"` key. Reset when the backend process restarts.
No SQLite write for user data.

**Rationale:** Single-user localhost demo. No persistence
requirement. Avoids DB migration complexity.

**Schema** (`_user_brand_state`):

```
Field                Type        Source
─────────────────────────────────────────────────
brand_id             str         "user_brand" (fixed)
brand_name           str         User input (Brand Designation)
mission              str         User input (Mission/Core Vision)
snippets             list[str]   User input (3–7 pasted snippets)
top_keywords         list[str]   Extracted from mission + snippets
mean_sentiment       float       Avg sentiment across snippets
std_sentiment        float       Std dev sentiment
mean_formality       float       Avg formality across snippets
std_formality        float       Std dev formality
mean_flesch          float       Avg Flesch reading ease
std_flesch           float       Std dev Flesch
avg_sentiment        float       Alias for mean_sentiment
avg_formality        float       Alias for mean_formality
avg_readability      float       Alias for mean_flesch
initialized          bool        True after first successful init
initialized_at       str         ISO timestamp
snippet_embeddings   list[vec]   384-dim vecs (optional, gated)
```

> The `mean_*` / `std_*` / `avg_*` fields match the format
> used by competitor `brand_profiles.profile_json`, so the
> canonical scorer works identically for both.

### 3.2 Snippet Count Validation

- **Minimum:** 3 snippets (need statistical variance).
- **Maximum:** 7 snippets (keep demo snappy).
- **Per-snippet min length:** 40 characters.
- **Per-snippet max length:** 2000 characters.
- Enforced in both frontend (UI validation) and backend
  (Pydantic model).

### 3.3 Embedding Gating

Runtime embedding computation is **optional** and gated
behind an environment variable:

```
ENABLE_RUNTIME_EMBEDDINGS=false   (default)
```

When `false`:
- t-SNE uses only competitor embeddings from the FAISS
  index; user brand is positioned by feature-distance
  interpolation (no `sentence-transformers` loaded).
- RAG retrieval for user brand uses keyword matching
  (TF-IDF over `brand_chunks`) instead of vector search.

When `true`:
- `sentence-transformers` is loaded lazily on first call.
- User snippets are embedded at init time.
- t-SNE includes real user embeddings.
- RAG uses vector similarity.

---

## 4. API Contract

### 4.1 POST /api/user-brand/init

Initializes or overwrites the ephemeral user brand profile.

**Request:**

```json
{
  "brand_name": "Meridian Watches",
  "mission": "Crafting timepieces that bridge ...",
  "snippets": [
    "Our heritage of precision...",
    "Every Meridian watch embodies...",
    "Designed for the modern explorer..."
  ]
}
```

**Validation:**
- `brand_name`: 2–60 characters, required.
- `mission`: 20–500 characters, required.
- `snippets`: list of 3–7 strings, each 40–2000 chars.

**Response (200):**

```json
{
  "status": "initialized",
  "brand_name": "Meridian Watches",
  "top_keywords": ["precision", "heritage", "explorer"],
  "mean_sentiment": 0.72,
  "mean_formality": 0.68,
  "mean_flesch": 42.5,
  "n_snippets": 3,
  "initialized_at": "2026-03-03T10:00:00Z"
}
```

**Errors:**
- 422: Validation failure (Pydantic).

### 4.2 GET /api/user-brand/status

**Response (200):**

```json
{
  "initialized": true,
  "brand_name": "Meridian Watches",
  "top_keywords": ["precision", "heritage", "explorer"],
  "mean_sentiment": 0.72,
  "initialized_at": "2026-03-03T10:00:00Z"
}
```

If not initialized:

```json
{
  "initialized": false,
  "brand_name": null,
  "message": "Run Genome Setup to create your brand profile."
}
```

### 4.3 POST /api/check-consistency

Scores text against user brand (default) or a competitor.

**Request:**

```json
{
  "brand_id": "user_brand",
  "text": "This timepiece embodies..."
}
```

- `brand_id = "user_brand"` → score against `_user_brand_state`.
- `brand_id = "rolex"` → score against DB competitor profile.

**Response (200):** (unchanged schema)

```json
{
  "brand_id": "user_brand",
  "brand_name": "Meridian Watches",
  "overall_score": 72.4,
  "tone_pct": 81.2,
  "vocab_overlap_pct": 45.0,
  "sentiment_alignment_pct": 88.1,
  "readability_match_pct": 62.3,
  "error": null
}
```

**Errors:**
- 404 `{"error": "profile_missing"}` if user brand not
  initialized and `brand_id = "user_brand"`.
- 400 `{"error": "genome_not_initialized"}` with friendly
  message if user hasn't run setup yet.

### 4.4 POST /api/benchmark

**Request:**

```json
{
  "competitor_id": "omega",
  "metric": "sentiment"
}
```

Valid metrics: `sentiment`, `keyword_overlap`,
`readability`, `formality`, `overall`.

**Response (200):**

```json
{
  "user_brand": {
    "name": "Meridian Watches",
    "value": 72.4,
    "label": "High Alignment"
  },
  "competitor": {
    "name": "Omega",
    "value": 65.1,
    "label": "Moderate Alignment"
  },
  "metric": "sentiment",
  "radar_data": [
    {"subject": "Vocab", "A": 45, "B": 60},
    {"subject": "Tone", "A": 81, "B": 55}
  ]
}
```

### 4.5 GET /api/analytics/tone-distribution

**Query:** `?competitor_id=omega` (optional)

**Response (200):**

```json
{
  "buckets": [
    "Very Casual", "Casual", "Neutral",
    "Formal", "Very Formal"
  ],
  "user_brand": [5, 12, 22, 45, 16],
  "market_avg": [10, 20, 35, 25, 10],
  "competitor": [8, 15, 30, 35, 12]
}
```

Computed from formality scores across all `brand_chunks`.
User brand values from user snippets. Market average from
all 10 competitor brands.

### 4.6 GET /api/analytics/tsne

**Query:** `?competitor_id=omega` (optional)

**Response (200):**

```json
{
  "points": [
    {"x": -40, "y": 30, "brand": "Rolex",
     "cluster": "luxury", "is_user": false},
    {"x": 5, "y": 15, "brand": "Your Brand",
     "cluster": "you", "is_user": true}
  ],
  "highlighted_competitor": "omega"
}
```

Computed via sklearn `TSNE(n_components=2)` over brand
centroid embeddings from FAISS index. User brand point
positioned by feature interpolation (or real embedding
if `ENABLE_RUNTIME_EMBEDDINGS=true`).

### 4.7 GET /api/analytics/pillars-heatmap

**Query:** `?competitor_ids=omega,tag_heuer,tissot`

**Response (200):**

```json
{
  "pillars": [
    "Sustainability", "Precision", "Heritage",
    "Value", "Innovation"
  ],
  "brands": [
    {
      "name": "Your Brand",
      "is_user": true,
      "weights": [0.1, 0.85, 0.9, 0.3, 0.7]
    },
    {
      "name": "Omega",
      "is_user": false,
      "weights": [0.3, 0.8, 0.7, 0.4, 0.7]
    }
  ]
}
```

Weights computed via keyword-group overlap: each pillar
has a canonical keyword list; weight = ratio of matching
keywords found in that brand's `brand_chunks` corpus.

### 4.8 POST /api/rewrite

RAG-grounded rewrite with iterative re-scoring.

**Request:**

```json
{
  "text": "This watch is cool and easy to wear.",
  "brand_id": "user_brand",
  "n_grounding_chunks": 3,
  "max_iterations": 2
}
```

**Response (200):**

```json
{
  "brand_id": "user_brand",
  "brand_name": "Meridian Watches",
  "original_text": "This watch is cool and easy to wear.",
  "rewritten_text": "This timepiece exemplifies...",
  "score_before": {
    "overall_score": 18.2,
    "tone_pct": 12.0,
    "vocab_overlap_pct": 5.0,
    "sentiment_alignment_pct": 35.0,
    "readability_match_pct": 22.0
  },
  "score_after": {
    "overall_score": 68.5,
    "tone_pct": 75.0,
    "vocab_overlap_pct": 42.0,
    "sentiment_alignment_pct": 80.0,
    "readability_match_pct": 65.0
  },
  "iterations_used": 1,
  "suggestions": [
    "Replaced casual vocabulary with brand terms.",
    "Elevated tone toward formal register."
  ],
  "grounding_chunks_used": [
    "Our heritage of precision engineering..."
  ],
  "error": null
}
```

**Error states:**
- `"error": "llm_unavailable"` — no API key configured;
  returns deterministic template rewrite.
- `"error": "llm_timeout"` — OpenAI call exceeded timeout.
- 404 if profile missing.

---

## 5. Execution Timeline

### Checkpoint 1 — Genome Setup + Scoring (Week 1)

**Goal:** User can create a brand profile and score text
against it end-to-end.

**Deliverables:**

- `POST /api/user-brand/init` implemented and tested
- `GET /api/user-brand/status` implemented
- `_user_brand_state` ephemeral store in `main.py`
- `UserBrandProfile` builder module extracts features
  from snippets (sentiment, formality, readability, keywords)
- `POST /api/check-consistency` supports `brand_id =
  "user_brand"` routing
- BrandSetup.jsx redesigned: snippet textarea (3–7),
  validation, success feedback
- ConsistencyCheck.jsx: "Your Brand" as default in dropdown;
  disable scoring with message if genome not initialized

**Acceptance tests:**

- `test_user_brand_init_happy_path` — 3 snippets → 200,
  profile has `mean_sentiment`, `top_keywords` populated.
- `test_user_brand_init_validation` — 1 snippet → 422.
- `test_check_consistency_user_brand` — init profile, then
  score → returns valid scores in [0, 100].
- `test_check_consistency_not_initialized` — skip init,
  score with `brand_id=user_brand` → 400 with friendly error.
- `test_user_brand_status` — returns `initialized: true`
  after init, `false` before.
- Frontend: Genome Setup form saves successfully; Consistency
  Check shows "Your Brand" default and disables if not init.

### Checkpoint 2 — Benchmarking + Analytics (Week 2)

**Goal:** All four pages show real, computed data.

**Deliverables:**

- `POST /api/benchmark` uses real `brand_chunks` features
  for comparison (not just profile-level scores)
- `GET /api/analytics/tone-distribution` computes formality
  histogram from `brand_chunks` + user snippets
- `GET /api/analytics/tsne` runs sklearn TSNE over brand
  embeddings; user brand positioned via interpolation
- `GET /api/analytics/pillars-heatmap` computes keyword-
  group overlap from `brand_chunks`
- Benchmarking.jsx: competitor dropdown populated from
  `/api/brands`; metric dropdown drives single-chart output
- Analytics.jsx: wired to three new endpoints; removes all
  hardcoded mock data

**Acceptance tests:**

- `test_benchmark_real_data` — compare user vs omega →
  radar_data has 5 dimensions, all values in [0, 100].
- `test_tone_distribution_shape` — response has `buckets`
  (5 items), `user_brand` (5 ints), `market_avg` (5 ints).
- `test_tsne_includes_user` — at least one point has
  `is_user: true`.
- `test_pillars_heatmap_shape` — 5 pillars, ≥2 brands,
  all weights in [0, 1].
- Frontend: tone histogram, t-SNE scatter, and heatmap
  render with real data; no hardcoded arrays remain.

### Checkpoint 3 — RAG + Rewrite Agent + Demo (Week 3)

**Goal:** Rewrite agent functional; demo script runs
end-to-end without errors.

**Deliverables:**

- RAG retrieval module: keyword-based (default) or
  vector-based (if `ENABLE_RUNTIME_EMBEDDINGS=true`)
  retrieval over `brand_chunks`
- `POST /api/rewrite` integrated with LLM (OpenAI) and
  fallback template rewrite
- Iterative re-scoring loop (up to `max_iterations`)
- `ENABLE_REWRITE_UI=true` flipped on in
  `ConsistencyCheck.jsx`
- RewritePanel, SuggestionsPanel, GroundingExamples
  components wired to real data
- Demo script document with step-by-step instructions

**Acceptance tests:**

- `test_rewrite_no_llm` — no API key → returns template
  rewrite, `score_after` populated, `error = null`.
- `test_rewrite_with_llm` — (integration, skip if no key)
  returns rewritten text, score_after > score_before.
- `test_rewrite_short_text` — < 10 chars → error response.
- `test_rewrite_grounding_chunks` — response includes
  `grounding_chunks_used` (non-empty list).
- `test_rag_keyword_fallback` — with embeddings disabled,
  retrieval returns chunks via keyword matching.
- Full demo smoke test: init brand → score → benchmark →
  analytics → rewrite → verify all pages render.

---

## 6. Work Distribution

### Person A — Backend: Genome Setup + Ephemeral Store

**Responsibilities:**

- Implement `_user_brand_state` ephemeral store in `main.py`
- Build `UserProfileBuilder` module
  (`src/scoring/user_profile.py`):
  - Accept name, mission, snippets
  - Extract features per snippet using existing extractors
  - Compute aggregate statistics (mean/std)
  - Extract keywords via TF-IDF or frequency analysis
- Implement `POST /api/user-brand/init`
- Implement `GET /api/user-brand/status`
- Update `POST /api/check-consistency` to route
  `brand_id = "user_brand"` to ephemeral store
- Pydantic validation models for all inputs

**Deliverables:**

- `src/scoring/user_profile.py` — profile builder
- Updated `src/api/main.py` — 3 new/modified endpoints
- `tests/test_user_brand.py` — ≥6 unit tests

**Dependencies:**

- Existing feature extractors (sentiment, formality,
  readability, vocabulary) — no changes needed
- Canonical scorer `src/scoring/consistency.py` — no
  changes needed (accepts any profile dict)

**Acceptance criteria:**

- Init with 3 valid snippets → 200 with all stats populated
- Init with 1 snippet → 422
- Status returns `initialized: true` after init
- Scoring with `user_brand` returns valid [0, 100] scores
- Scoring with `user_brand` before init → 400 with message

### Person B — Backend: Analytics + Benchmarking Endpoints

**Responsibilities:**

- Implement `GET /api/analytics/tone-distribution`
  - Query `brand_chunks`, compute formality per chunk
  - Bucket into 5 bins (Very Casual → Very Formal)
  - Compute market average across all brands
  - Include user brand distribution from snippets
- Implement `GET /api/analytics/tsne`
  - Load FAISS index / metadata
  - Run sklearn TSNE over brand centroids
  - Position user brand via feature interpolation
- Implement `GET /api/analytics/pillars-heatmap`
  - Define 5 pillar keyword lists
  - Compute keyword-group overlap per brand from chunks
- Refactor `POST /api/benchmark` to use chunk-level
  features for the selected metric

**Deliverables:**

- Updated `src/api/main.py` — 4 new/modified endpoints
- `src/analytics/engine.py` — analytics computation module
- `tests/test_analytics.py` — ≥8 unit tests

**Dependencies:**

- Person A's `_user_brand_state` (needed for user brand
  data in analytics) — can stub with test profile
- `brand_chunks` table (already populated, 15 per brand)
- FAISS index + metadata (already built, 5 brands)

**Acceptance criteria:**

- Tone distribution returns 5 buckets, counts are integers
- t-SNE returns ≥6 points, one with `is_user: true`
- Pillars heatmap returns 5 pillars, weights in [0, 1]
- Benchmark with valid competitor → radar_data populated

### Person C — Frontend: All Four Pages

**Responsibilities:**

- **BrandSetup.jsx** — Redesign:
  - Remove single "Mission" text area
  - Add snippet input area (paste box + counter showing
    3–7 required)
  - Character count validation per snippet (40–2000)
  - Call `POST /api/user-brand/init` on submit
  - Show success state with extracted keywords and stats
- **ConsistencyCheck.jsx** — Modify:
  - Change dropdown: "Your Brand" (default, value
    `user_brand`) + competitor brands from `/api/brands`
  - If genome not initialized, show disabled state with
    message: "Initialize your brand genome first"
  - Check `/api/user-brand/status` on mount
  - Wire `ENABLE_REWRITE_UI` flag (flip in Checkpoint 3)
- **Benchmarking.jsx** — Modify:
  - Populate competitor dropdown from `/api/brands`
    (remove hardcoded options, remove "Titan")
  - Map metric dropdown values to API metric keys
  - Display single metric chart + radar overlay
- **Analytics.jsx** — Rewrite:
  - Replace all hardcoded data arrays
  - Wire tone histogram to `/api/analytics/tone-distribution`
  - Wire t-SNE scatter to `/api/analytics/tsne`
  - Wire heatmap to `/api/analytics/pillars-heatmap`
  - Keep existing chart components and styling

**Deliverables:**

- Updated `BrandSetup.jsx`
- Updated `ConsistencyCheck.jsx`
- Updated `Benchmarking.jsx`
- Updated `Analytics.jsx`
- Updated `App.jsx` (pass user-brand status to children)

**Dependencies:**

- Person A's endpoints (init, status, check-consistency)
- Person B's endpoints (analytics/*, benchmark)
- Can develop against mock API responses initially

**Acceptance criteria:**

- Genome Setup: submit 3 snippets → keywords appear
- Consistency Check: "Your Brand" default, scoring works
- Consistency Check: disabled state if genome not initialized
- Benchmarking: dropdown from API, chart renders on submit
- Analytics: all 3 charts render with live data
- No hardcoded mock data arrays remain in any page

### Person D — RAG + Rewrite Agent + Integration Testing

**Responsibilities:**

- Build RAG retrieval module (`src/rag/retriever.py`):
  - Keyword-based retrieval (TF-IDF over `brand_chunks`)
    as default mode
  - Vector-based retrieval (FAISS) as optional mode gated
    by `ENABLE_RUNTIME_EMBEDDINGS`
  - For `user_brand`: retrieve chunks most similar to user
    snippets (keyword overlap)
  - For competitors: retrieve by `brand_id` from DB
- Build rewrite agent (`src/rag/rewrite_agent.py`):
  - Accept text + profile + grounding chunks
  - Construct LLM prompt (system + grounding examples +
    edit plan + original text)
  - Call OpenAI (or return template fallback)
  - Re-score rewritten text
  - Support iterative loop (up to `max_iterations`)
- Implement `POST /api/rewrite` endpoint
- Flip `ENABLE_REWRITE_UI=true` in Checkpoint 3
- Write integration tests and demo smoke script
- Write `docs/demo_script.md`

**Deliverables:**

- `src/rag/__init__.py`
- `src/rag/retriever.py` — RAG retrieval module
- `src/rag/rewrite_agent.py` — rewrite agent
- Updated `src/api/main.py` — rewrite endpoint refactored
- `tests/test_rag.py` — ≥6 tests
- `tests/test_rewrite_agent.py` — ≥4 tests
- `docs/demo_script.md` — step-by-step demo

**Dependencies:**

- Person A's ephemeral store (profile to rewrite against)
- Person B's analytics endpoints (for demo verification)
- Person C's UI (RewritePanel, SuggestionsPanel already
  exist; need `ENABLE_REWRITE_UI=true` flip)
- `brand_chunks` table (already populated)
- OpenAI API key (optional; fallback mode works without)

**Acceptance criteria:**

- Keyword retrieval returns ≥1 chunk for any known brand
- Rewrite without LLM key → template rewrite + scores
- Rewrite with LLM key → rewritten text, score_after >
  score_before (on average)
- Iterative loop terminates within `max_iterations`
- Full demo script runs without errors

---

## 7. RACI Summary

**R** = Responsible, **A** = Accountable,
**C** = Consulted, **I** = Informed

**Person A: Genome Setup + Ephemeral Store**

- R: User profile builder, init/status endpoints
- A: Scoring works against user brand
- C: Person C (frontend contract), Person D (profile format)
- I: Person B

**Person B: Analytics + Benchmarking**

- R: Three analytics endpoints, benchmark refactor
- A: All charts show real computed data
- C: Person A (user brand data), Person C (response shapes)
- I: Person D

**Person C: Frontend (All Pages)**

- R: All four page components, UI validation
- A: No placeholder data remains in the frontend
- C: Person A and B (API contracts)
- I: Person D

**Person D: RAG + Rewrite Agent + Integration**

- R: RAG module, rewrite agent, demo script
- A: End-to-end demo runs without errors
- C: Person A (profile), Person C (UI flag flip)
- I: Person B

---

## 8. Testing Plan

### Unit Tests (per checkpoint)

| Module | File | Count | Owner |
|--------|------|-------|-------|
| User profile builder | `test_user_brand.py` | ≥6 | A |
| Analytics engine | `test_analytics.py` | ≥8 | B |
| RAG retriever | `test_rag.py` | ≥6 | D |
| Rewrite agent | `test_rewrite_agent.py` | ≥4 | D |

### Integration Tests

| Test | Scope | Owner |
|------|-------|-------|
| Init → Score | A+C flow | A |
| Init → Benchmark | A+B flow | B |
| Init → Analytics | A+B flow | B |
| Init → Rewrite | A+D flow | D |
| Full demo smoke | All pages | D |

### Demo Smoke Test (automated)

```bash
# 1. Start backend
python3 -m uvicorn src.api.main:app --port 8000 &

# 2. Init user brand
curl -X POST localhost:8000/api/user-brand/init \
  -H "Content-Type: application/json" \
  -d '{"brand_name":"Demo Brand",
       "mission":"Precision timepieces for explorers",
       "snippets":["Our watches embody precision...",
                   "Crafted for endurance and style...",
                   "Heritage meets modern innovation..."]}'

# 3. Check status
curl localhost:8000/api/user-brand/status

# 4. Score text
curl -X POST localhost:8000/api/check-consistency \
  -H "Content-Type: application/json" \
  -d '{"brand_id":"user_brand",
       "text":"This watch is really cool and easy."}'

# 5. Benchmark
curl -X POST localhost:8000/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{"competitor_id":"omega","metric":"sentiment"}'

# 6. Analytics
curl "localhost:8000/api/analytics/tone-distribution"
curl "localhost:8000/api/analytics/tsne"
curl "localhost:8000/api/analytics/pillars-heatmap"

# 7. Rewrite
curl -X POST localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"brand_id":"user_brand",
       "text":"This watch is really cool and easy.",
       "n_grounding_chunks":3,"max_iterations":1}'
```

All responses must return HTTP 200 with valid JSON and
`error: null` (or no `error` key).

### Regression

Run full existing test suite after each checkpoint:

```bash
python3 -m pytest -q
```

Must remain at ≥441 passed, 0 failures.

---

## 9. Risks & Mitigations

**Risk 1: Runtime embedding segfault**

- *Impact:* Process crash on macOS with Python 3.9 +
  faiss-cpu + sentence-transformers co-loaded.
- *Mitigation:* Gate behind `ENABLE_RUNTIME_EMBEDDINGS=false`
  default. Keyword-based RAG and feature-interpolation
  t-SNE work without loading the embedding model.
  Document the flag in `.env.template`.

**Risk 2: OpenAI API key not configured**

- *Impact:* Rewrite agent cannot call LLM.
- *Mitigation:* Deterministic template fallback rewrite
  that still produces `score_before` / `score_after`.
  Error field set to `null` (not a failure).

**Risk 3: Ephemeral state lost on server restart**

- *Impact:* User loses brand profile.
- *Mitigation:* By design (single-user demo). Frontend
  shows clear "not initialized" state and directs user
  to Genome Setup. Could add localStorage persistence
  on frontend as future enhancement.

**Risk 4: FAISS index only covers 5 of 10 brands**

- *Impact:* t-SNE and vector-based RAG miss half the brands.
- *Mitigation:* Rebuild the FAISS index to include all 10
  brands (add to Checkpoint 2 acceptance criteria). Use
  `scripts/build_embeddings_index.py` with updated config.
  Keyword-based fallback covers all 10 brands regardless.

**Risk 5: Snippet quality too low for meaningful profiles**

- *Impact:* Garbage-in → garbage-out scores.
- *Mitigation:* Enforce minimum per-snippet length (40 chars)
  and minimum count (3). Show a quality indicator in the
  UI after init (e.g., "Profile confidence: Medium" based
  on snippet variance).

**Risk 6: Frontend-backend contract drift**

- *Impact:* UI breaks silently when API response shape
  changes.
- *Mitigation:* Freeze all response schemas in Pydantic
  `response_model` decorators. Add contract tests in
  `test_api.py` that assert exact key sets. Person C
  develops against documented schemas, not live API.

**Risk 7: Analytics computation too slow for demo**

- *Impact:* t-SNE or TF-IDF over 150 chunks takes >5s.
- *Mitigation:* Pre-compute and cache analytics on first
  request (lazy singleton). t-SNE over 10 brand centroids
  (not 150 chunks) is <100ms. Cache invalidates only on
  user brand init.

---

## 10. How to Run the Demo

### Start Backend

```bash
cd /path/to/Brand-Genome-Engine

# Optional: configure LLM (needed for rewrite only)
export OPENAI_API_KEY=sk-your-key-here

python3 -m uvicorn src.api.main:app \
  --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### Open in Browser

Navigate to **http://localhost:5173**

### Demo Script

**Step 1 — Genome Setup** (left sidebar → "Genome Setup")

1. Enter Brand Designation: `Meridian Watches`
2. Enter Mission: `Crafting precision timepieces that bridge
   heritage and modern exploration for discerning collectors.`
3. Paste 3–5 sample brand copy snippets into the snippets area.
4. Click **Initialize Profile**.
5. Verify: keywords extracted, stats shown, success message.

**Step 2 — Consistency Check** (sidebar → "Consistency Check")

1. Dropdown shows "Your Brand" (selected by default).
2. Paste off-brand text:
   `This watch is really cool and super easy to wear every
   day. Nice design.`
3. Click **Analyze Consistency**.
4. Observe low scores (tone, vocabulary especially).
5. Replace with on-brand text matching your snippets' style.
6. Click again. Observe higher scores.

**Step 3 — Market Benchmarking** (sidebar → "Market Benchmarking")

1. Select competitor: Omega.
2. Select metric: Sentiment Distribution.
3. Click **Run Simulation**.
4. Observe bar chart + radar chart comparing your brand vs Omega.

**Step 4 — Data Analytics** (sidebar → "Data Analytics")

1. Observe tone histogram: Your Brand vs Market Avg.
2. Observe t-SNE: Your Brand highlighted among competitors.
3. Observe heatmap: messaging pillars intensity across brands.

**Step 5 — Rewrite** (sidebar → "Consistency Check")

1. Paste off-brand text (same as Step 2).
2. Click **Ground & Rewrite** (visible after Checkpoint 3).
3. Observe: rewritten text, before/after scores, grounding
   examples, suggestions panel.
4. Verify score_after > score_before.

---

*End of Phase 4 Plan*
