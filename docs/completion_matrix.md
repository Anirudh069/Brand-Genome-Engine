# Completion Matrix — Brand Genome Engine (Stage 9)

| Item | Status | Evidence |
|---|---|---|
| Canonical DB | PASS | `data/brand_data.db` only `.db` in repo; `PRAGMA integrity_check = ok` (verify_phase4, validate_db, validate_chunk_handoff) |
| Competitor corpus | PASS | brand_texts=450, brand_texts_raw=450, 45/brand (>=30) |
| Chunking | PASS | brand_chunks=657, 51-87/brand (>=50), 0 blank/orphan/dup, max 395 chars (<=400) |
| User Genome | PASS | `POST /api/genome/init` enforces exactly-7 snippets, persists to `brand_profile`; verified by `tests/test_genome_stage1.py`, `tests/test_user_rag_stage5_1.py` |
| Consistency | PASS | `src/scoring/consistency.py`, `tests/test_consistency_stage2.py`, real weighted Gaussian/Jaccard scorer, writes `analysis_history` |
| Benchmarking | PASS | `src/benchmarking/market_benchmark.py`, DB-only, 3 metrics (tone/sentiment/readability), `tests/test_benchmark_stage3.py` |
| Analytics Pillars | PASS | `src/analytics/pillars.py`, 5 fixed names + corpus-derived keywords, `tests/test_analytics_stage4.py` |
| Analytics t-SNE | PASS | `src/analytics/chunk_tsne.py`, chunk-level, `tests/test_analytics_stage4.py` |
| Tone | PASS | `src/analytics/tone.py`, heuristic formality-based histogram |
| History | PASS | `src/analytics/history.py`, live counters + score trend from `analysis_history` |
| Chunk Embeddings | PASS | `all-MiniLM-L6-v2`, 384-d, verified with `--include-model-tests` (63/63 passed) |
| FAISS | PASS | `IndexFlatIP` per brand_id, `src/retrieval/rag_builder.py`, manifest fingerprinted |
| Semantic Retrieval | PASS | `POST /api/rag/retrieve`, `tests/test_rag_stage5.py` (incl. real-model run, 55/55 passed) |
| User RAG Corpus | PASS | Genome init materializes 8 user source rows -> chunked into `brand_chunks` (`user_brand`) |
| Rewrite | PASS | `src/rewrite/rewrite_service.py`, `tests/test_rewrite_stage6.py`, same scorer pre/post |
| OpenAI | PASS (opt-in) | Real smoke test previously run manually (Stage 6, documented in repo memory); regression tests use dependency-injected fake provider to avoid repeat paid calls |
| Rebuild Profile | PASS | `tests/test_rebuild_stage7.py`, idempotent |
| Rebuild Chunks | PASS | `tests/test_rebuild_stage7.py`, idempotent, partial-status handling |
| Rebuild Index | PASS | `tests/test_rebuild_stage7.py`, idempotent, atomic swap |
| Frontend Integration | PASS | `npm run lint` clean, `npm run build` succeeds (738 KB / 223 KB gzip, one chunk-size warning only) |
| E2E | PASS | `tests/test_e2e_stage8.py` — full journey, history checkpoint 1/1/1 (total 3) |
| Tests | PASS | `python -m pytest tests/`: 615 passed, 82 skipped (opt-in real-model), 0 failures, 0 errors |
| Reproducibility | PASS | Clean-copy check (see below); artifacts rebuild from DB; app starts with no `.env`/OpenAI key (fallback rewrite provider) |
| Documentation | PASS | README, this file, architecture/database/api/methodology/demo_fixture/demo_script all current |

## Optional / non-blocking

- `src/benchmarking/retrieval.py` + `scripts/build_embeddings_index.py` +
  `scripts/query_competitors.py` (legacy 5-brand centroid CLI tool) —
  clearly labelled LEGACY, retained only for its own standalone test.
- Bundle-size warning (738 KB) — cosmetic, not a functional blocker.
- `sqlalchemy`, `beautifulsoup4`, `plotly`, `tqdm` removed from
  `requirements.txt` as genuinely unused.
