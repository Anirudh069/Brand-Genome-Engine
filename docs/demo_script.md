# Demo Script — Brand Genome Engine (5–8 minutes)

Use the fixture in [demo_fixture.md](demo_fixture.md). Backend running
(`uvicorn src.api.main:app --reload`), frontend running (`npm run dev`).

## 1. Problem & architecture (30s)
"Marketing teams lose brand voice consistency across channels. This tool
measures it with real NLP against a persisted brand genome, benchmarks
against real competitors, visualises it, and can rewrite off-brand copy
grounded in the brand's own words." Point to [architecture.md](architecture.md).

## 2. Genome Setup (60s)
Enter the fixture's designation, mission, and exactly 7 snippets. Submit.
Show the genome is now persisted (`GET /api/genome`), not ephemeral.

## 3. Consistency Check (60s)
Paste the off-brand example text. Show the composite score and the four
sub-scores (tone/vocab/sentiment/readability) with diagnostics — emphasize
this is a real weighted Gaussian/Jaccard computation, not a canned number.

## 4. Benchmarking (45s)
Pick a competitor (e.g. Rolex). Show the 3-metric chart (tone, sentiment,
readability) computed from the real `brand_profiles` row.

## 5. Analytics (90s)
Open Analytics. Explain:
- Five fixed pillars with corpus-derived keywords (TF-IDF + semantic
  similarity, not hand-picked).
- Chunk-level t-SNE scatter — a visualisation of chunk embeddings, not a
  metric.
- Tone distribution histogram.
- Live history counters (should already show 1 consistency + 1 benchmark
  event from steps 3–4).

## 6. Rebuild RAG index if needed (30s)
If the RAG index is stale (e.g. genome was just (re)initialised), call
`POST /api/rebuild/index` from Dev Tools and show it succeeds idempotently.

## 7. Rewrite (90s)
Paste the off-brand text into Rewrite. Show:
- The retrieved user snippets used for grounding (from the user's own 7
  snippets, never a competitor's).
- Before/after score using the *same* scorer as step 3.
- The rewritten text.

## 8. Analytics refresh (30s)
Return to Analytics — history counters should now read 1/1/1 (total 3),
and the score trend should include both the consistency and rewrite events.

## 9. Dev Tools / reproducibility (30s)
Show `POST /api/rebuild/profile` and `/chunks` are idempotent and safe, and
mention `python -m scripts.verify_phase4` as the one-command DoD check.

## Talking points to avoid overselling
- Corpus is modest (10 competitors, ~45 texts each) — an academic PoC, not
  a production dataset.
- FAISS uses exact search because the corpus is small; this would not scale
  to millions of chunks without an approximate index.
- LLM rewrite output varies; the score is not guaranteed to always improve.
- Single-user, local-only — no auth, no multi-tenant concurrency story.
