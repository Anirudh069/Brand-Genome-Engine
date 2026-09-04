"""
tone.py – Real tone distribution derived from brand_texts.

Reuses the project's canonical deterministic tone-label heuristic
(``src.api.genome_service._tone_label``: formality + sentiment -> one of
{authoritative, formal, motivational, neutral}) instead of inventing a new
subjective classifier. Also exposes a continuous formality histogram since
that is the real underlying measured feature.
"""

from __future__ import annotations

import numpy as np

from src.api.genome_service import _tone_label
from src.feature_extraction.formality_extractor import extract_formality
from src.feature_extraction.sentiment_extractor import extract_sentiment

TONE_LABELS = ["authoritative", "formal", "motivational", "neutral"]


def compute_tone_distribution(texts: list[dict]) -> dict:
    """
    Parameters
    ----------
    texts : list of {"brand_id", "brand_name", "text"}

    Returns
    -------
    {
      "labels": [...],
      "by_brand": {brand_id: {label: count, ...}, ...},
      "totals": {label: count, ...},
      "formality_histogram": {"bins": [...], "counts": [...]},
    }
    """
    by_brand: dict[str, dict[str, int]] = {}
    totals = {label: 0 for label in TONE_LABELS}
    formality_scores: list[float] = []

    for row in texts:
        formality = extract_formality(row["text"])
        sentiment = extract_sentiment(row["text"])
        label = _tone_label(formality, sentiment)
        formality_scores.append(formality)

        brand_id = row["brand_id"]
        bucket = by_brand.setdefault(brand_id, {tone_label: 0 for tone_label in TONE_LABELS})
        bucket[label] += 1
        totals[label] += 1

    if formality_scores:
        hist, bin_edges = np.histogram(formality_scores, bins=10, range=(0, 1))
        formality_histogram = {"bins": bin_edges.tolist(), "counts": hist.tolist()}
    else:
        formality_histogram = {"bins": [], "counts": []}

    return {
        "labels": TONE_LABELS,
        "by_brand": by_brand,
        "totals": totals,
        "formality_histogram": formality_histogram,
    }
