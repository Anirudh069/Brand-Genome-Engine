"""
heatmap.py – Messaging Pillar TF-IDF heatmap (competitor brands x pillars).

Formula
-------
pillar_intensity(brand, pillar) = sum(
    brand_tfidf(term) * pillar_term_relevance(term)
    for term in pillar_terms[pillar]
)

``brand_tfidf`` is computed by fitting a single deterministic TF-IDF
vectorizer over one concatenated document per competitor brand (10
documents). ``pillar_term_relevance`` is the derived ``score`` from
:func:`src.analytics.pillars.derive_pillar_keywords`.

The raw matrix is additionally min-max scaled to a 0-100 range for
visualization using one global scale factor (max over the whole matrix), so
no brand/pillar is individually tuned. Raw values are preserved alongside
the scaled ones.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{3,}\b"


def compute_pillar_heatmap(
    brand_ids: list[str],
    brand_names: list[str],
    brand_documents: list[str],
    pillar_names: list[str],
    pillar_terms: dict[str, list[dict]],
) -> dict:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        lowercase=True,
        token_pattern=_TOKEN_PATTERN,
    )
    matrix = vectorizer.fit_transform(brand_documents).toarray()
    vocab_index = {term: idx for idx, term in enumerate(vectorizer.get_feature_names_out())}

    n_brands = len(brand_ids)
    n_pillars = len(pillar_names)
    raw = np.zeros((n_brands, n_pillars), dtype=np.float64)

    for col_idx, pillar in enumerate(pillar_names):
        for term_info in pillar_terms.get(pillar, []):
            vocab_col = vocab_index.get(term_info["term"])
            if vocab_col is None:
                continue
            raw[:, col_idx] += matrix[:, vocab_col] * term_info["score"]

    max_val = float(raw.max()) if raw.size else 0.0
    scaled = (raw / max_val) * 100.0 if max_val > 0 else raw.copy()

    return {
        "brand_ids": brand_ids,
        "brands": brand_names,
        "pillars": pillar_names,
        "values": np.round(scaled, 2).tolist(),
        "raw_values": np.round(raw, 6).tolist(),
    }
