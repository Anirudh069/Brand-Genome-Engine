# ML Contract – Brand Genome Engine

> **Version:** 1.0  
> **Last updated:** 2026-03-01  
> **Owner:** Feature Extraction team

---

## Purpose

This document defines the **canonical schema** that every text document must
conform to after passing through `TextFeatureExtractor.extract_all_features()`.

All downstream consumers – FAISS index builders, benchmarking scripts, API
layers, and notebooks – **depend only on `ExtractedFeatures`**.  If you change
a field here you _must_ bump the version and notify downstream owners.

---

## Schema: `ExtractedFeatures`

| # | Field                 | Type            | Range / Shape            | Description |
|---|-----------------------|-----------------|--------------------------|-------------|
| 1 | `text_id`             | `str \| None`   | free-form                | Unique document identifier (from the DB). |
| 2 | `brand_id`            | `str \| None`   | free-form                | Brand identifier. |
| 3 | `brand_name`          | `str \| None`   | free-form                | Human-readable brand name. |
| 4 | `text`                | `str`           | non-empty after cleaning | The cleaned input text. |
| 5 | `sentiment`           | `float`         | **[-1.0, 1.0]**         | Negative → positive sentiment. |
| 6 | `formality`           | `float`         | **[0.0, 1.0]**          | Informal → formal tone. |
| 7 | `readability_flesch`  | `float`         | **[0.0, ~121.0]**       | Flesch Reading Ease score (higher = easier). |
| 8 | `avg_sentence_length` | `float`         | **[0.0, ∞)**            | Average words per sentence. |
| 9 | `punctuation_density` | `float`         | **[0.0, 1.0]**          | Ratio of punctuation characters to total characters. |
|10 | `vocab_diversity`     | `float`         | **[0.0, 1.0]**          | Type-Token Ratio (unique / total tokens). |
|11 | `top_topics`          | `list[str]`     | length = `num_topics`    | Human-readable topic labels. |
|12 | `topic_weights`       | `list[float]`   | same length, sums ≤ 1.0 | Corresponding topic weights. |
|13 | `embedding`           | `list[float]`   | **length = 384**         | Dense vector from `all-MiniLM-L6-v2`. |
|14 | `embedding_model`     | `str`           | model identifier         | Name of the embedding model used. |

---

## Embedding contract

| Property   | Value                  |
|------------|------------------------|
| Model      | `all-MiniLM-L6-v2`    |
| Dimensions | **384**                |
| Norm       | Unit-normalised (L2)   |
| Library    | `sentence-transformers`|

> ⚠️  All FAISS indices, similarity computations, and downstream vector
> operations **assume 384 dimensions**.  Changing the model requires
> rebuilding every index and updating `EMBEDDING_DIM` in
> `src/feature_extraction/text_features.py`.

---

## Feature details

### Sentiment (`sentiment`)
- **Extractor:** `sentiment_extractor.py`
- **Method (planned):** Transformer-based (e.g., `cardiffnlp/twitter-roberta-base-sentiment`) or VADER.
- **Output:** Single float in [-1, 1].

### Formality (`formality`)
- **Extractor:** `formality_extractor.py`
- **Method (planned):** Fine-tuned classifier (e.g., `s-nlp/roberta-base-formality-ranker`) or heuristic POS-tag ratio.
- **Output:** Single float in [0, 1].

### Readability (`readability_flesch`, `avg_sentence_length`)
- **Extractor:** `readability_extractor.py`
- **Method (planned):** Flesch Reading Ease via `textstat`; sentence-length computed from tokenisation.
- **Output:** Two floats.

### Vocabulary (`punctuation_density`, `vocab_diversity`)
- **Extractor:** `vocabulary_extractor.py`
- **Method (planned):** Character-level punctuation ratio; Type-Token Ratio on whitespace tokens.
- **Output:** Two floats in [0, 1].

### Topics (`top_topics`, `topic_weights`)
- **Extractor:** `topic_extractor.py`
- **Method (planned):** LDA / NMF via scikit-learn or BERTopic.
- **Output:** Two parallel lists of length `num_topics`.

### Embedding (`embedding`)
- **Extractor:** `embedding_extractor.py`
- **Method (planned):** `sentence-transformers` `SentenceTransformer.encode()`.
- **Output:** `list[float]` of length **384**.

---

## Validation

`ExtractedFeatures.validate()` checks:

1. `len(embedding) == 384`
2. `sentiment ∈ [-1, 1]`
3. `formality ∈ [0, 1]`
4. `punctuation_density ∈ [0, 1]`
5. `vocab_diversity ∈ [0, 1]`
6. `len(top_topics) == len(topic_weights)`

Call `.validate()` after construction to catch contract violations early.

---

## Consuming the contract downstream

```python
from src.feature_extraction import TextFeatureExtractor, ExtractedFeatures

extractor = TextFeatureExtractor(
    embedding_model="all-MiniLM-L6-v2",
    num_topics=5,
)

features: ExtractedFeatures = extractor.extract_all_features(
    text="Rolex epitomises timeless luxury...",
    text_id="txt_001",
    brand_id="brand_rolex",
    brand_name="Rolex",
)

# Access fields
print(features.sentiment)        # float in [-1, 1]
print(len(features.embedding))   # 384
features.validate()              # raises ValueError on contract violation
```

---

## Versioning policy

| Change type          | Action required |
|----------------------|-----------------|
| Add optional field   | Bump minor version; no downstream breakage. |
| Rename / remove field| Bump major version; coordinate with all consumers. |
| Change embedding dim | Rebuild all FAISS indices; bump major version. |
