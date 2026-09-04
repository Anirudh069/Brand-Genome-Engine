# API — Brand Genome Engine (Final / Stage 9)

Base URL: `http://localhost:8000` (see `CORS_ORIGIN` env for the allowed
frontend origin). All request/response bodies are JSON.

## Canonical endpoints

### `POST /api/genome/init`
Initialise/replace the user genome.
- **Request**: `{ designation: str, mission_core_vision: str, snippets: [str] }`
  — `snippets` must contain **exactly 7** non-blank entries.
- **Response**: the persisted genome summary (see `GET /api/genome`).
- **Errors**: `400` invalid payload (wrong snippet count, blank fields).

### `GET /api/genome`
Return the current user genome summary (designation, mission, snippet count,
tone/NLP features, genome_version) or `{"initialized": false}` if not yet set.

### `POST /api/consistency/score`
Score arbitrary text against the **user's own genome only**.
- **Request**: `{ text: str }`
- **Response**: `overall_score`, `tone_pct`, `vocab_overlap_pct`,
  `sentiment_alignment_pct`, `readability_match_pct`, `diagnostic_breakdown`.
  Writes one `analysis_history` row (`event_type="consistency"`).
- **Errors**: `400 genome_not_initialized` if no genome exists yet.

### `GET /api/benchmark/brands`
List the 10 competitor brands available for benchmarking (DB-only).

### `POST /api/benchmark/run`
Compare the user genome against one competitor on 3 metrics: tone, sentiment,
readability.
- **Request**: `{ competitor_brand_id: str }`
- **Response**: per-metric values for user vs competitor + chart-ready shape.
  Writes one `analysis_history` row (`event_type="benchmark"`,
  `pre_score=post_score=None` by design — benchmark events never appear in
  the Analytics score trend, only Consistency/Rewrite do).
- **Errors**: `400` unknown competitor, `400 genome_not_initialized`.

### `GET /api/analytics`
Real, DB-derived analytics: five fixed pillars + corpus-derived keywords,
TF-IDF terms, chunk-level t-SNE, tone distribution, and live history
counters/score trend. No hardcoded chart data — if the underlying artifact
cannot be built, an explicit `metadata.artifact_error` is returned instead of
fabricated numbers.

### `POST /api/rag/retrieve`
Strict brand-scoped semantic retrieval over the chunk-level RAG index.
- **Request**: `{ text: str, brand_id: str, top_k?: int (1-10, default 5) }`
- **Response**: retrieved chunks with similarity scores. Does **not** write
  `analysis_history`.
- **Errors**: `404 unknown_brand`, `503 index_missing`, `503 index_stale`.

### `POST /api/rewrite`
Rewrite text to better match the user's own brand voice, grounded in the
user's own RAG corpus (never a competitor's).
- **Request**: `{ text: str, top_k?: int (1-10) }` (`extra="forbid"` — no
  `brand_id` accepted).
- **Response**: `rewritten_text`, `pre_score`, `post_score`, `score_delta`,
  `retrieved_chunk_ids`, `retrieval_scores`, `edit_plan`, `provider`, `model`.
  Writes exactly one `analysis_history` row (`event_type="rewrite"`).
- **Errors**: `400 genome_not_initialized`, `400 user_genome_chunks_missing`,
  `503 index_missing` / `index_stale` / `user_grounding_not_indexed`.

### `POST /api/rebuild/profile`
Recompute the user's `brand_profile` (features + embedding) from the current
`brand_texts`/`brand_chunks`. Idempotent, safe to call repeatedly.

### `POST /api/rebuild/chunks`
Re-chunk the user's (or all) `brand_texts` into `brand_chunks`. Idempotent;
returns `207` with `status="partial"` if some sources failed.

### `POST /api/rebuild/index`
Rebuild the chunk-level FAISS RAG index for all brands from the current
`brand_chunks`. Idempotent, atomic swap on success.

## Deprecated compatibility aliases

These remain registered (delegating to the canonical handler above) purely
for backward compatibility; the frontend never calls them:

| Alias | Canonical equivalent |
|---|---|
| `POST /api/check-consistency` | `POST /api/consistency/score` |
| `POST /api/profile/rebuild` | `POST /api/rebuild/profile` |
| `POST /api/index/rebuild` | `POST /api/rebuild/index` |
| `POST /api/chunks/rebuild` | `POST /api/rebuild/chunks` |
| `GET /api/profile` | `GET /api/genome` |
| `POST /api/profile` | `POST /api/genome/init` |
| `POST /api/benchmark` | `POST /api/benchmark/run` |

## Misc

- `GET /api/health` — liveness probe (`{"status": "ok"}`).
- `GET /api/brands` — non-canonical convenience listing of competitor
  brand_id/brand_name pairs (falls back to a static list only if the DB is
  entirely unreachable; never fabricates scores).

Route registration is verified automatically by
`python -m scripts.verify_phase4` (no duplicate `(path, method)` pairs, all
11 canonical routes present).
