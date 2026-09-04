"""
pillars.py – Automatic derivation of Messaging Pillar keyword sets.

The five pillar CONCEPT names are the only hand-authored constants (fixed
Phase 4 contract). The keyword sets associated with each pillar are derived
deterministically from the canonical competitor corpus (brand_texts):

1. TF-IDF candidate terms/bigrams are extracted from the corpus, with
   document-frequency filtering to suppress one-off garbage and ubiquitous
   generic terms.
2. Each candidate's corpus strength is its mean TF-IDF weight across the
   documents in which it appears, normalised to [0, 1] across candidates.
3. Each candidate term is embedded with the project's existing local
   embedding abstraction and compared (cosine similarity) against the
   embedding of the pillar's concept name.
4. pillar_term_score = semantic_similarity(term, pillar) * corpus_strength(term)
5. The top ``TOP_K_PILLAR_TERMS`` terms per pillar (score > 0) are kept.

No hand-authored synonym/keyword dictionary is used.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_extraction.embedding_extractor import get_embedding

# Fixed Phase 4 pillar concept names (do not rename/replace).
PILLAR_NAMES: list[str] = ["Sustainability", "Precision", "Heritage", "Value", "Innovation"]

TOP_K_PILLAR_TERMS = 8
MIN_DOC_FREQUENCY = 2
MAX_DOC_FREQUENCY = 0.6  # drop terms present in > 60% of documents (too generic)
_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{3,}\b"


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_corpus_tfidf(documents: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    """Fit a deterministic TF-IDF vectorizer over the candidate corpus."""
    min_df = MIN_DOC_FREQUENCY if len(documents) > MIN_DOC_FREQUENCY else 1
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=MAX_DOC_FREQUENCY,
        lowercase=True,
        token_pattern=_TOKEN_PATTERN,
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def derive_pillar_keywords(
    documents: list[str],
    top_k: int = TOP_K_PILLAR_TERMS,
) -> dict[str, list[dict]]:
    """
    Derive keyword sets for each fixed pillar concept from ``documents``
    (the canonical brand_texts corpus).

    Returns
    -------
    dict pillar_name -> list of {"term", "similarity", "corpus_strength", "score"}
    ordered by descending score. Every returned term is guaranteed to occur
    literally in ``documents`` (TF-IDF vocabulary is built from the corpus).
    """
    if not documents:
        return {pillar: [] for pillar in PILLAR_NAMES}

    vectorizer, matrix = build_corpus_tfidf(documents)
    vocab = vectorizer.get_feature_names_out()
    if len(vocab) == 0:
        return {pillar: [] for pillar in PILLAR_NAMES}

    dense = matrix.toarray()
    mean_weights = np.zeros(len(vocab), dtype=np.float64)
    for j in range(len(vocab)):
        column = dense[:, j]
        nonzero = column[column > 0]
        mean_weights[j] = float(nonzero.mean()) if nonzero.size else 0.0

    max_weight = mean_weights.max() if mean_weights.size else 0.0
    corpus_strength = mean_weights / max_weight if max_weight > 0 else mean_weights

    term_embeddings = [get_embedding(term)[0] for term in vocab]

    pillar_terms: dict[str, list[dict]] = {}
    for pillar in PILLAR_NAMES:
        pillar_embedding, _ = get_embedding(pillar)
        scored = []
        for idx, term in enumerate(vocab):
            similarity = _cosine(term_embeddings[idx], pillar_embedding)
            if similarity <= 0:
                continue
            score = similarity * corpus_strength[idx]
            if score <= 0:
                continue
            scored.append(
                {
                    "term": term,
                    "similarity": round(similarity, 4),
                    "corpus_strength": round(float(corpus_strength[idx]), 4),
                    "score": round(float(score), 6),
                }
            )
        scored.sort(key=lambda entry: (-entry["score"], entry["term"]))
        pillar_terms[pillar] = scored[:top_k]

    return pillar_terms
