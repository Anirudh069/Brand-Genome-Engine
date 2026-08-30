"""
vocabulary_extractor.py – Vocabulary, punctuation, and sentence-length metrics.

Delegates entirely to the helpers in ``feature_utils`` so that logic is
never duplicated.
"""

from __future__ import annotations

import logging

from src.feature_extraction.feature_utils import (
    avg_sentence_length,
    clean_text,
    punctuation_density,
    sentence_split,
    vocab_diversity,
    word_tokenize,
)

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, float] = {
    "vocab_diversity": 0.0,
    "avg_sentence_length": 0.0,
    "punctuation_density": 0.0,
}


def extract_vocab_metrics(text: str | None) -> dict[str, float]:
    """
    Compute vocabulary / surface-form metrics for *text*.

    Returns
    -------
    dict with keys:
        * ``vocab_diversity``      – [0.0, 1.0] TTR
        * ``avg_sentence_length``  – [0.0, ∞)   tokens per sentence
        * ``punctuation_density``  – [0.0, 1.0] punct chars / total chars

    The function **never raises**; it returns sane zero-defaults for
    ``None``, empty, or otherwise degenerate input.
    """
    try:
        cleaned = clean_text(text)
        if not cleaned:
            return dict(_DEFAULTS)

        tokens = word_tokenize(cleaned)

        return {
            "vocab_diversity": vocab_diversity(tokens),
            "avg_sentence_length": avg_sentence_length(cleaned),
            "punctuation_density": punctuation_density(cleaned),
        }
    except Exception:
        logger.exception("extract_vocab_metrics failed – returning defaults")
        return dict(_DEFAULTS)
