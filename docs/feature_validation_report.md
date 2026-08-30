# Feature Validation Report – Brand Genome Engine

> **Generated:** 2026-03-01  
> **Dataset:** `data/processed/features.parquet`  
> **Rows:** 75  
> **Brands:** 5 (Omega, Patek Phillipe, Rolex, TAG Heuer, Tissot — 15 texts each)

---

## 1. Per-Feature Summary

| Feature                | Min      | Mean     | Max      | Notes |
|------------------------|----------|----------|----------|-------|
| `sentiment`            | 0.4167   | 0.5842   | 0.7222   | All positive; expected for marketing copy |
| `formality`            | 0.7264   | 0.7997   | 0.8980   | Consistently high — luxury brand language |
| `readability_flesch`   | 0.4150   | 39.3582  | 69.9040  | Range spans "difficult" to "standard" |
| `avg_sentence_length`  | 11.0000  | 26.1162  | 60.0000  | Some very long single-sentence paragraphs |
| `punctuation_density`  | 0.0097   | 0.0211   | 0.0462   | Low density — prose-heavy copy |
| `vocab_diversity`      | 0.6514   | 0.7883   | 0.9429   | High TTR — rich vocabulary |

All values fall within the ranges defined in [`ml_contract.md`](ml_contract.md).

---

## 2. Topic Sanity

| Property                        | Value   |
|---------------------------------|---------|
| `num_topics` per row            | 5       |
| `topic_weights` sum — min       | 1.0000  |
| `topic_weights` sum — mean      | 1.0000  |
| `topic_weights` sum — max       | 1.0000  |

Weights are properly normalised across all rows.

### Example `top_topics` per Brand

| Brand           | Top Topics (descending weight)                                    |
|-----------------|-------------------------------------------------------------------|
| Rolex           | heritage, performance, craftsmanship, innovation, luxury          |
| Omega           | craftsmanship, performance, innovation, lifestyle, luxury         |
| TAG Heuer       | groundbreaking, collaboration, partenerships, foundation, leadership |
| Patek Phillipe  | craftsmanship, design, heritage, connoisseurs, manufacture        |
| Tissot          | heritage, innovation, tissot, watchmakers, pioneering             |

> **Note:** The topic extractor uses a TF-IDF keyword baseline. Topics are
> not semantically grouped — they are the most distinctive terms per text.

---

## 3. Embedding Sanity

| Property                   | Value                      |
|----------------------------|----------------------------|
| Expected dimension         | 384                        |
| All rows have length 384   | ✅ Yes                     |
| `embedding_model` field    | `all-MiniLM-L6-v2` (all rows) |
| Embedding source           | **REAL model** (sentence-transformers) |
| L2 norm — min              | 1.0000                     |
| L2 norm — mean             | 1.0000                     |
| L2 norm — max              | 1.0000                     |

All embeddings are unit-normalised and produced by the real
`all-MiniLM-L6-v2` model (confirmed via the `embedding_model` column — no
rows show a fallback hash-based embedding).

---

## 4. Retrieval Sanity

**Index:** `embeddings/brand_profile_index.faiss` (FAISS backend)  
**Metadata:** `embeddings/metadata.json`  
**Method:** Cosine distance via normalised inner product

### Top-3 Nearest Competitors per Brand (excluding self)

| Query Brand    | #1 Competitor             | #2 Competitor              | #3 Competitor               |
|----------------|---------------------------|----------------------------|-----------------------------|
| Omega          | Rolex (0.9859)            | Tissot (0.9868)            | TAG Heuer (1.0238)          |
| Patek Phillipe | TAG Heuer (0.9815)        | Rolex (1.0037)             | Tissot (1.0275)             |
| Rolex          | Omega (0.9859)            | Patek Phillipe (1.0037)    | TAG Heuer (1.0080)          |
| TAG Heuer      | Patek Phillipe (0.9815)   | Tissot (0.9931)            | Rolex (1.0080)              |
| Tissot         | Omega (0.9868)            | TAG Heuer (0.9931)         | Patek Phillipe (1.0275)     |

Distance values are **cosine distance** (= 1 − cosine similarity).  Lower
values indicate greater similarity.  All brands cluster tightly (distances
~0.98–1.03), which is expected for watch brands sharing similar luxury
vocabulary.

---

## 5. Known Limitations

1. **Topic extractor is a keyword baseline.**  
   `topic_extractor.py` uses TF-IDF term extraction, not a generative or
   latent topic model (LDA / BERTopic).  Topics are the most distinctive
   *words*, not semantic clusters.  The term "tissot" appearing as a topic
   for Tissot is a symptom of this.

2. **Embeddings may be fallback if model not installed.**  
   If `sentence-transformers` is not installed at extraction time, the
   embedding extractor silently falls back to a deterministic hash-based
   pseudo-embedding.  These preserve shape (384-d) and are non-zero but
   carry **no semantic meaning**.  Check the `embedding_model` column and
   norms: hash-fallback vectors are unit-normalised but produce *different*
   norms from the real model.  In this dataset all embeddings are from the
   real model.

3. **Distance interpretation depends on backend.**  
   - **FAISS:** index stores L2-normalised vectors; search uses inner
     product.  Returned distances are `1 − IP` (= cosine distance).  
   - **sklearn fallback:** uses `NearestNeighbors(metric='cosine')`, which
     natively returns cosine distance.  
   Both backends produce the same *semantics* (lower = more similar) but
   numerical values may differ at float-precision level.

4. **Small dataset.**  
   75 texts across 5 brands is a proof-of-concept corpus.  Feature
   distributions (especially sentiment and formality) may shift
   substantially with a larger, more diverse dataset.
