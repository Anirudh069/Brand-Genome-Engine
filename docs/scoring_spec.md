# Scoring Specification — Brand Genome Engine

> Canonical scorer: `src/scoring/consistency.py`

## Public API

```python
compute_consistency_score(text: str, brand_profile: dict) -> dict
generate_edit_plan(text: str, brand_profile: dict) -> dict
```

## Output Schema (frozen)

`compute_consistency_score` returns a dict with **exactly** these keys,
all floats clamped to [0, 100]:

| Key                       | Description                          |
|---------------------------|--------------------------------------|
| `overall_score`           | Weighted composite score             |
| `tone_pct`                | Gaussian formality alignment         |
| `vocab_overlap_pct`       | Jaccard keyword overlap              |
| `sentiment_alignment_pct` | Gaussian sentiment alignment         |
| `readability_match_pct`   | Gaussian readability alignment       |

## Algorithm

### 1. Vocabulary Overlap (Jaccard)

```
vocab_overlap = |content_words(text) ∩ top_keywords| / |content_words(text) ∪ top_keywords|
```

Content words = lowercased alpha tokens ≥ 3 chars, stopwords removed.

### 2. Sentiment Alignment (Gaussian)

```
sentiment_align = exp(-((s - μ_s)² / (2 · σ_s²)))
```

Where `s` = text sentiment [0, 1], `μ_s` / `σ_s` from profile (`mean_sentiment` / `std_sentiment`). σ clamped to ≥ 0.01.

### 3. Readability Match (Gaussian)

```
readability = exp(-((f - μ_f)² / (2 · σ_f²)))
```

Where `f` = Flesch reading ease, `μ_f` / `σ_f` from profile (`mean_flesch` / `std_flesch`). σ clamped to ≥ 5.0.

### 4. Tone (Gaussian Formality)

```
tone = exp(-((form - μ_form)² / (2 · σ_form²)))
```

Where `form` = text formality [0, 1], `μ_form` / `σ_form` from profile (`mean_formality` / `std_formality`). σ clamped to ≥ 0.01.

> **Note:** Cosine-embedding tone (text embedding vs `mean_embedding`) is
> reserved for the offline batch pipeline.  Real-time scoring uses the
> formality proxy to avoid loading `sentence-transformers` at request time.

### 5. Overall Score (Weighted Average)

```
overall = (0.30 × tone + 0.25 × sentiment + 0.25 × vocab + 0.20 × readability) × 100
```

### Short-Text Guard

Texts with fewer than 10 words return all zeros.

### Profile Field Fallbacks

The scorer reads `mean_*` keys with `avg_*` fallbacks (e.g., `mean_sentiment` → `avg_sentiment`). Both formats are written by `scripts/build_brand_profiles.py`.

## Constraints

- **Deterministic**: Same input → same output, always.
- **No heavy deps at scoring time**: No `sentence-transformers`, no GPU ops.
- **All values 0–100**: Clamped.
- **Never crashes**: Missing profile fields → safe defaults.

## Tests

Contract tests: `tests/test_scoring.py`
API integration tests: `tests/test_api.py`
