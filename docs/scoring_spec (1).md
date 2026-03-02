# Scoring Specification — Person C (Ayesha)

**Project:** Brand Consistency Checker  
**Version:** 1.0  
**Date:** 2026-03-01  

---

## Overview

`score_consistency(text_features, brand_profile)` compares a new piece of text against a brand's learned style profile and returns five scores, all in the range **0–100**.

| Output field | Meaning |
|---|---|
| `overall_score` | Weighted average of the four metrics below |
| `tone_pct` | How closely the text's tone/style matches the brand |
| `vocab_overlap_pct` | How many of the brand's key words appear in the text |
| `sentiment_alignment_pct` | How closely the text's emotional tone matches the brand |
| `readability_match_pct` | How closely the text's reading difficulty matches the brand |

---

## Metric Formulas

### 1. Vocabulary Overlap (`vocab_overlap_pct`)

**Formula — Jaccard Similarity:**

```
vocab_overlap = |A ∩ B| / |A ∪ B|  ×  100
```

- `A` = content keywords extracted from the input text (stopwords removed, ≥ 3 chars)
- `B` = `brand_profile["top_keywords"]` — the brand's 10 most characteristic words

**Normalisation:** Result is already 0–1 before ×100. Final clamped to [0, 100].

**Edge cases:**
- Both lists empty → return 0 (not 100 — empty match is not a match)
- One list empty → intersection is 0 → result is 0

---

### 2. Sentiment Alignment (`sentiment_alignment_pct`)

**Formula — Gaussian Similarity:**

```
sentiment_alignment = exp( −((s − μ)² / (2σ²)) )  ×  100
```

- `s` = sentiment score of input text (range −1 to 1)
- `μ` = `brand_profile["mean_sentiment"]`
- `σ` = `brand_profile["std_sentiment"]`

The Gaussian function returns 1.0 when the text's sentiment exactly matches the brand mean, and decays smoothly as it diverges.

**Normalisation:** exp() returns 0–1. Final multiplied by 100 and clamped.

**Edge cases:**
- If `std_sentiment < 0.01` → clamp σ = 0.01 to prevent division by near-zero
- NaN in sentiment feature → raise `FeatureExtractionError`

---

### 3. Readability Match (`readability_match_pct`)

**Formula — Inverse Distance:**

```
tolerance = max(2 × std_flesch, 20)   # minimum tolerance = 20 Flesch points

readability_match = max(0,  1 − |f − μ_f| / tolerance)  ×  100
```

- `f` = Flesch Reading Ease score of input text (computed via standard formula)
- `μ_f` = `brand_profile["mean_flesch"]`
- `std_flesch` = `brand_profile["std_flesch"]`

A brand with high variability in readability (high `std_flesch`) gets a wider tolerance band — meaning the scorer accepts a broader range of input readability levels.

**Normalisation:** max(0, …) ensures non-negative. Final clamped to [0, 100].

**Edge cases:**
- If `std_flesch` is 0 → use tolerance = 20 (hardcoded floor)
- Text with no sentences or no words → flesch defaults to 50.0

---

### 4. Tone (`tone_pct`)

**Formula — Cosine Similarity:**

```
tone = cosine_similarity(e_text,  e_brand_mean)  ×  100
```

- `e_text` = sentence embedding of input text (384-d vector from all-MiniLM-L6-v2)
- `e_brand_mean` = `brand_profile["mean_embedding"]` — average of all brand text embeddings

Cosine similarity measures the angle between two vectors, returning 1.0 for identical direction and 0.0 for orthogonal.

**Fallback (when embeddings not available):** If `mean_embedding` is empty (e.g., before Person B's FAISS integration), tone falls back to a sentiment-distance proxy:  
```
tone_fallback = max(0,  1 − |sentiment_text − mean_sentiment_brand|)
```

**Normalisation:** cosine result clamped to [0, 1] before ×100.

**Edge cases:**
- Zero vector (all-zero embedding) → return 0
- Dimension mismatch between text and brand embeddings → raise `EmbeddingDimensionError`

---

## Overall Score

**Formula — Weighted Average:**

```
overall_score = (0.30 × tone_pct)
              + (0.25 × sentiment_alignment_pct)
              + (0.25 × vocab_overlap_pct)
              + (0.20 × readability_match_pct)
```

| Metric | Weight | Rationale |
|---|---|---|
| Tone | 0.30 | Embedding similarity is the richest style signal |
| Sentiment Alignment | 0.25 | Emotional register is critical for brand voice |
| Vocabulary Overlap | 0.25 | Shared keywords signal on-brand messaging |
| Readability Match | 0.20 | Reading level matters but varies more naturally |
| **Total** | **1.00** | |

**Normalisation:** Final score clamped to **[0, 100]**.

---

## Edge Cases — Full Reference

| Situation | Behaviour |
|---|---|
| Input text < 10 words | Scorer returns all scores = 0; caller/API sets `error = "text_too_short"` |
| Brand profile not found | Raise `BrandProfileNotFoundError(brand_id)` |
| Embedding dimension mismatch | Raise `EmbeddingDimensionError` |
| Both keyword lists empty | `vocab_overlap_pct = 0` |
| NaN in any feature | Raise `FeatureExtractionError` with field name |
| `std_sentiment < 0.01` | Clamp to 0.01 before Gaussian calculation |
| `std_flesch = 0` | Use tolerance = 20 |
| Zero embedding vector | `tone_pct = 0` |
| `mean_embedding = []` | Use sentiment-distance fallback for tone |

---

## Flesch Reading Ease — Reference Formula

```
Flesch = 206.835  −  (1.015 × avg_sentence_length)  −  (84.6 × avg_syllables_per_word)
```

Higher score = easier to read. Brand copy from luxury watchmakers typically scores 40–55.

---

## Brand Profile Builder (`brand_profile_builder.py`)

Aggregates per-brand stats from `brand_texts`:

| Profile field | How computed |
|---|---|
| `mean_sentiment` | Mean of per-text sentiment scores |
| `std_sentiment` | Std-dev of per-text sentiment scores |
| `mean_flesch` | Mean of per-text Flesch scores |
| `std_flesch` | Std-dev of per-text Flesch scores |
| `mean_vocab_richness` | Mean type-token ratio across texts |
| `mean_formality` | Mean ratio of long words (≥7 chars) to content words |
| `top_keywords` | Top 10 content words by frequency across all brand texts |
| `tone_label` | Categorical label derived from formality + sentiment means |
| `mean_embedding` | Mean of sentence embeddings (populated when Person B's pipeline available) |

---

*All field names, weights, and error types in this document match the implementation in `consistency_scorer.py` exactly. Any change requires written agreement from Person C and Person D.*
