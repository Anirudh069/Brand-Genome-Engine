# ML Methodology — Brand Genome Engine (Final / Stage 9)

## User genome

- Genome = designation + mission/core vision + exactly 7 short brand
  snippets. Feature aggregation runs each source through the same NLP
  extractors as competitors (sentiment, formality, Flesch readability,
  vocabulary/top-keywords), producing per-feature mean/std, plus a
  dense MiniLM embedding of the concatenated corpus.
- The genome is persisted (`brand_profile`, `brand_id=0`) and used
  identically by Consistency Check, Benchmarking, and Rewrite pre/post
  scoring — no scoring logic is duplicated per feature.

## Consistency scoring (`src/scoring/consistency.py`, see [scoring_spec.md](../docs/scoring_spec.md))

Four Gaussian/Jaccard sub-scores, weighted into one composite:

```
overall = (0.30*tone + 0.25*sentiment + 0.25*vocab_overlap + 0.20*readability) * 100
```

- `tone` = Gaussian formality alignment vs the genome's `mean_formality`/`std_formality`.
- `sentiment_alignment` = Gaussian alignment vs `mean_sentiment`/`std_sentiment`.
- `readability_match` = Gaussian alignment vs `mean_flesch`/`std_flesch`.
- `vocab_overlap` = Jaccard overlap between the input's content words and the
  genome's top keywords.
- A diagnostic breakdown accompanies every score for explainability.

## Benchmarking

DB-only comparison of the user genome against one competitor's
`brand_profiles` row on **tone, sentiment, readability** (no keyword-overlap
metric in the canonical contract — that dimension is Consistency-only).
Competitor `avg_sentence_length` is computed on demand from `brand_texts`
(not persisted) and cached per `brand_id` in-process.

## Analytics

- **Five fixed concept pillars** (`Sustainability`, `Precision`, `Heritage`,
  `Value`, `Innovation` — the *names* are the only hand-authored constant;
  every keyword under each pillar is corpus-derived via semantic similarity
  between a TF-IDF-ranked candidate term and the pillar's MiniLM embedding,
  weighted by corpus frequency).
- **TF-IDF** ranks candidate terms per brand from `brand_chunks` text.
- **Chunk-level t-SNE** projects chunk embeddings (MiniLM) to 2D for
  visual clustering — a visualisation aid, not an evaluation metric.
- **Tone distribution** buckets chunks by formality score into
  labelled histogram bins (heuristic, explainable thresholds — not a
  trained classifier).
- **History** (counters + score trend) is read live from `analysis_history`
  on every request, so newly-run Consistency/Rewrite events show up
  immediately; Benchmark events intentionally have no score and are excluded
  from the score trend.
- The whole artifact is content-fingerprinted against `brand_texts`+
  `brand_chunks` and rebuilt automatically when stale.

## RAG (retrieval-augmented grounding)

- Every `brand_chunks` row (all 10 competitors + `user_brand`) is embedded
  with `all-MiniLM-L6-v2` (384-d) and L2-normalized.
- One `faiss.IndexFlatIP` per `brand_id` — inner product on unit vectors is
  cosine similarity. Exact (flat) search is appropriate because each
  brand's corpus is small (tens to low hundreds of chunks).
- `POST /api/rag/retrieve` is strictly brand-scoped (never cross-brand
  leakage) and detects staleness via a corpus fingerprint (SHA-256 over
  ordered chunk_id/text_id/brand_id/chunk_text) compared against the
  index manifest.

## Rewrite

1. Pre-score the input with the canonical scorer against the user genome.
2. Build a deterministic edit plan (`generate_edit_plan`) from the
   flattened genome features (target formality/sentiment/readability/tone,
   top keywords).
3. Retrieve top-k chunks from the **user's own** RAG index only (never a
   competitor's) for grounding.
4. Call OpenAI (`openai>=2.x` Responses API) with the edit plan + retrieved
   snippets, or the deterministic `FallbackRewriteProvider` if no
   `OPENAI_API_KEY`/explicit `REWRITE_PROVIDER=fallback`.
5. Post-score the rewritten text with the **same** scorer.
6. Write one `analysis_history` row with pre/post score, retrieved chunk
   ids/scores, and the edit plan for auditability.

Rewrite output is not guaranteed to improve the score (LLM output can
vary) — the system reports `score_delta` honestly either way.
