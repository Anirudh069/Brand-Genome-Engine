# Chunking Handoff Report

## Status

**READY FOR PERSON A DATA-QUALITY VALIDATION**

## Canonical database
- Path: `data/brand_data.db`
- Physical `.db` files in repo: 1 (verified via repo-wide search; none elsewhere)
- SQLite `PRAGMA integrity_check`: `ok`

## Source corpus
- Total `brand_texts`: 450
- Total `brand_texts_raw`: 450
- Brands: 10
- Per-brand `brand_texts` counts: audemars 45, breitling 45, cartier 45, hublot 45, iwc 45, omega 45, patek_phillipe 45, rolex 45, tag_heuer 45, tissot 45
- All brands exceed the Phase 4 minimum of 30 texts/brand

## Derived chunks
- Total chunks: 657
- Per-brand counts: audemars 53, breitling 87, cartier 81, hublot 77, iwc 57, omega 56, patek_phillipe 52, rolex 70, tag_heuer 51, tissot 73
- Min per-brand: 51 (tag_heuer) — Max per-brand: 87 (breitling)
- All brands exceed the Phase 4 minimum of 50 chunks/brand

## Chunk size
- min: 14
- median: 251
- mean: 234.9
- max: 395
- count < 80 chars: 49
- count > 400 chars: 0 (hard max satisfied)

## Traceability
- Orphan `text_id` references: 0
- Brand relationship mismatches (`chunk.brand_id != source.brand_id`): 0
- `brand_name` / `source_type` mismatches: 0
- Source rows producing zero chunks: 0 (all 450 non-blank source texts are represented)
- `char_count` mismatches (`char_count != len(chunk_text)`): 0
- Chunk ID collisions: 0 (657 rows, 657 distinct `chunk_id`)

## Warnings
- Short chunks (<80 chars): 49, all legitimate provenance-preserving cases (short source texts preserved whole, or trailing remainders that couldn't be safely merged without breaking sentence/word-boundary rules). See discrepancy explanation below for exactly which ones are and aren't captured by the builder's own warning log.
- Source duplicate-content groups (same brand, identical source text under different `text_id`): 11 groups, 24 affected rows. Example: `omega_006` / `omega_017` — identical Constellation collection text.
- Chunk duplicate-content groups (identical `chunk_text` from different `text_id`): 21 groups, 46 affected chunk rows — a direct consequence of the source-level duplicates above. Example: `cartier_028` / `cartier_034` / `cartier_039` all share the fragment `"30 meters/100 feet)."`.

None of these were altered, merged, or deleted — they are flagged for Person A's independent review only.

## 49-vs-48 discrepancy

**Definitive explanation:** these are two different counts measuring different things, and both are correct for what they measure.

- **49** is the count of chunk rows in `brand_chunks` with `char_count < 80` — a direct, independently-verified measurement of the actual data (`SELECT chunk_text FROM brand_chunks` filtered by `len(chunk_text) < 80` → 49 rows).
- **48** is the count of *warning log entries* the builder (`scripts/build_brand_chunks.py`) emits during chunking. The builder only appends a short-chunk warning in two specific code paths inside `pack_text_into_chunks`: (a) when an entire source text collapses to a single chunk under 80 chars, or (b) when a short trailing remainder chunk cannot be merged into the previous chunk. It does **not** inspect or warn about a short chunk that isn't the trailing chunk of a source text.

  Concretely, `iwc_045` produces 3 chunks: `chunk_000` (33 chars — the sentence "Purpose-designed for Spaceflight."), `chunk_001` (375 chars — a long sentence split at word boundaries), `chunk_002` (26 chars — the short final remainder, merge attempted and failed, so *this one* gets a warning). `chunk_000` is short (33 chars) but is not the trailing chunk — it was flushed as its own chunk only because the following sentence was oversized and triggered `flush()` before the split logic ran (`build_brand_chunks.py:146-148`). That flush path never runs the short-chunk check, so `chunk_000` never generates a warning.

  Net effect: 49 distinct DB rows are `<80` chars, but only 48 distinct source `text_id`s appear in the warning log, because `iwc_045` contributes 2 short chunks to the count while only contributing 1 warning message.

This is confirmed by cross-referencing all 49 short chunk rows against the 48 warning entries: every one of the 48 warned `text_id`s accounts for exactly one short chunk, except `iwc_045`, which contributes 2 short chunks under a single warning. No chunk is short for an unexplained reason — this is a known, benign gap in the warning log's coverage, not a data-quality defect. It does not affect the hard `<400`-char requirement or any integrity check.

## Tests
- `python -m pytest tests/test_build_brand_chunks.py -v`: **21/21 passed**
- `python -m scripts.validate_db --db-path data/brand_data.db`: **ALL CHECKS PASSED** (updated to Phase 4 per-brand thresholds instead of the stale fixed `150`/`150`/`150` counts)
- `python -m scripts.validate_chunk_handoff --db-path data/brand_data.db`: **PASS**, exit code 0 (3 warning categories, 0 hard failures)

## Scope
- Source tables (`brand_texts`, `brand_texts_raw`) were not modified by this audit — verified both by re-reading `scripts/build_brand_chunks.py` (only issues `SELECT` against `brand_texts`; only `DELETE`/`INSERT` against `brand_chunks`, transactionally, and only in `--replace` mode) and by `git status`/`git diff --stat`, which show only `data/brand_data.db` (binary diff) modified plus the two new script/test files.
- Embeddings were not generated.
- FAISS/RAG was not started.
- Profiles (`brand_profile`, `brand_profiles`) were not rebuilt or modified.
- Scoring, frontend, and analytics were not touched.
