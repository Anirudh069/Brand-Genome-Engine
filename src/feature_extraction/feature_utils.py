"""
feature_utils.py – Shared text-cleaning and helper utilities.

All functions are **deterministic**, never raise on bad input, and have
zero heavy-NLP dependencies (no spaCy, no NLTK downloads required).
"""

from __future__ import annotations

import re
import string
import unicodedata

# ── Pre-compiled patterns (module-level for speed) ────────────────────────
_RE_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
_RE_WHITESPACE = re.compile(r"\s+")
_RE_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"          # split after sentence-ending punctuation
)
_RE_WORD_TOKEN = re.compile(
    r"[A-Za-z0-9]+(?:'[A-Za-z]+)?"  # words, including contractions like don't
    r"|[^\s]"                         # or any single non-whitespace char
)

_PUNCTUATION_SET = set(string.punctuation)


# ── Text cleaning ─────────────────────────────────────────────────────────

def clean_text(text: str | None, *, max_chars: int = 0) -> str:
    """
    Robust text cleaning.  Handles ``None``, control characters,
    collapsed whitespace, and optional truncation.

    Parameters
    ----------
    text : str | None
        Raw input.  ``None`` is silently treated as ``""``.
    max_chars : int
        If > 0, the cleaned text is truncated via :func:`safe_truncate`.

    Returns
    -------
    str
        Cleaned (and possibly truncated) text.  Never raises.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)
    # 2. Remove control characters (keep newlines / tabs for now — they
    #    get collapsed in step 4)
    text = _RE_CONTROL_CHARS.sub("", text)
    # 3. Strip leading / trailing whitespace
    text = text.strip()
    # 4. Collapse all interior whitespace runs into a single space
    text = _RE_WHITESPACE.sub(" ", text)

    if max_chars > 0:
        text = safe_truncate(text, max_chars=max_chars)

    return text


# ── Truncation ────────────────────────────────────────────────────────────

def safe_truncate(text: str | None, *, max_chars: int = 5000) -> str:
    """
    Deterministic truncation to at most *max_chars* characters.

    Tries to break at the last space before the limit so words aren't
    split in half.  If the text is already short enough it is returned
    unchanged.

    Parameters
    ----------
    text : str | None
        Input text.  ``None`` → ``""``.
    max_chars : int
        Hard character cap (default 5 000).

    Returns
    -------
    str
    """
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    # Try to break at a word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated


# ── Sentence splitting ────────────────────────────────────────────────────

def sentence_split(text: str | None) -> list[str]:
    """
    Split *text* into sentences using a lightweight regex heuristic.

    Rules
    -----
    * Splits after ``[.!?]`` followed by whitespace.
    * Returns an empty list for ``None`` / empty input.
    * Never raises.

    Returns
    -------
    list[str]
        Non-empty sentence strings (empty fragments are dropped).
    """
    if not text:
        return []
    parts = _RE_SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


# ── Word tokenisation ────────────────────────────────────────────────────

def word_tokenize(text: str | None) -> list[str]:
    """
    Simple regex-based tokeniser.  Extracts words (including
    contractions like *don't*) and individual punctuation characters.

    Returns
    -------
    list[str]
        Tokens.  Empty list for ``None`` / empty input.
    """
    if not text:
        return []
    return _RE_WORD_TOKEN.findall(text)


# ── Numeric feature helpers ──────────────────────────────────────────────

def punctuation_density(text: str | None) -> float:
    """
    Ratio of punctuation characters to total characters.

    Returns **0.0** for ``None`` / empty strings (never divides by zero).

    Returns
    -------
    float
        Value in [0.0, 1.0].
    """
    if not text:
        return 0.0
    total = len(text)
    if total == 0:
        return 0.0
    punct_count = sum(1 for ch in text if ch in _PUNCTUATION_SET)
    return punct_count / total


def vocab_diversity(tokens: list[str] | None) -> float:
    """
    Type-Token Ratio: ``unique_tokens / total_tokens``.

    Returns **0.0** for ``None`` / empty token lists (never divides by zero).

    Returns
    -------
    float
        Value in [0.0, 1.0].
    """
    if not tokens:
        return 0.0
    total = len(tokens)
    if total == 0:
        return 0.0
    return len(set(tokens)) / total


def avg_sentence_length(text: str | None) -> float:
    """
    Average number of word-tokens per sentence.

    Uses :func:`sentence_split` and :func:`word_tokenize` internally.
    Returns **0.0** for ``None`` / empty strings (never divides by zero).

    Returns
    -------
    float
        Non-negative float.
    """
    sentences = sentence_split(text)
    if not sentences:
        return 0.0
    token_counts = [len(word_tokenize(s)) for s in sentences]
    total_tokens = sum(token_counts)
    return total_tokens / len(sentences)
