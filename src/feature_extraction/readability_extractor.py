"""
readability_extractor.py – Flesch Reading Ease and sentence-length metrics.

Implements the classic Flesch Reading Ease formula using only stdlib.
Delegates tokenisation and sentence-splitting to ``feature_utils``.

    FRE = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

Score contract:
    * flesch_reading_ease : typically [0, 100+], higher = easier to read.
    * avg_sentence_length : [0, ∞), words per sentence.
"""

from __future__ import annotations

import logging
import re

from src.feature_extraction.feature_utils import (
    avg_sentence_length as _avg_sentence_length,
    clean_text,
    sentence_split,
    word_tokenize,
)

logger = logging.getLogger(__name__)

_DEFAULT_FLESCH: float = 0.0
_DEFAULT_ASL: float = 0.0

# ── Syllable counting ────────────────────────────────────────────────────
_RE_ALPHA_WORD = re.compile(r"^[a-zA-Z]+$")
_RE_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _count_syllables(word: str) -> int:
    """
    Estimate syllable count for an English word.

    Heuristic:
    1. Count vowel groups (a, e, i, o, u, y).
    2. Subtract 1 for trailing silent-e (but ensure ≥ 1).
    3. Every word has at least 1 syllable.
    """
    word = word.lower().strip()
    if not word:
        return 0

    vowel_groups = _RE_VOWEL_GROUP.findall(word)
    count = len(vowel_groups)

    # Trailing silent-e: "make" → 1 syl, not 2
    if word.endswith("e") and count > 1:
        count -= 1

    # Trailing "-le" after consonant gets a syllable back: "table"
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiouy":
        count += 1

    # Trailing "-ed" is usually silent in past tense: "walked"
    if word.endswith("ed") and count > 1 and len(word) > 3:
        if word[-3] not in "dt":  # "wanted", "added" keep the -ed syllable
            count -= 1

    return max(1, count)


def _total_syllables(tokens: list[str]) -> int:
    """Sum syllables across all alphabetic tokens."""
    total = 0
    for tok in tokens:
        if _RE_ALPHA_WORD.match(tok):
            total += _count_syllables(tok)
    return total


def extract_readability(text: str | None) -> tuple[float, float]:
    """
    Return ``(flesch_reading_ease, avg_sentence_length)``.

    * flesch_reading_ease : typically [0.0, ~121.0] – higher = easier.
    * avg_sentence_length : [0.0, ∞) – average words per sentence.

    The function **never raises**; returns ``(0.0, 0.0)`` on any failure.
    """
    try:
        cleaned = clean_text(text)
        if not cleaned:
            return _DEFAULT_FLESCH, _DEFAULT_ASL

        sentences = sentence_split(cleaned)
        if not sentences:
            return _DEFAULT_FLESCH, _DEFAULT_ASL

        tokens = word_tokenize(cleaned)
        # Only count alpha tokens as "words" for Flesch
        alpha_tokens = [t for t in tokens if _RE_ALPHA_WORD.match(t)]

        num_sentences = len(sentences)
        num_words = len(alpha_tokens)

        if num_words == 0 or num_sentences == 0:
            return _DEFAULT_FLESCH, _DEFAULT_ASL

        num_syllables = _total_syllables(alpha_tokens)

        # ── Flesch Reading Ease ───────────────────────────────────────
        words_per_sentence = num_words / num_sentences
        syllables_per_word = num_syllables / num_words

        fre = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word

        # Clamp to [0, 121.22] (theoretical max)
        fre = max(0.0, min(121.22, fre))

        # ── Average sentence length (delegate to feature_utils) ───────
        asl = _avg_sentence_length(cleaned)

        return fre, asl

    except Exception:
        logger.exception("extract_readability failed – returning defaults")
        return _DEFAULT_FLESCH, _DEFAULT_ASL


def flesch_reading_ease(text: str | None) -> float:
    """
    Compute the Flesch Reading Ease score for *text*.

    This is a convenience wrapper around :func:`extract_readability` that
    returns only the FRE score (discarding avg-sentence-length).

    Returns
    -------
    float
        Flesch Reading Ease in [0.0, 121.22].  Higher = easier.
        Returns ``0.0`` for ``None``, empty, or degenerate input.
        **Never raises.**
    """
    fre, _ = extract_readability(text)
    return fre
