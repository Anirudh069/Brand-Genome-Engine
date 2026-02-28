"""
formality_extractor.py – Deterministic heuristic formality scorer.

Scores text formality on **[0.0, 1.0]** (informal → formal) using only
lightweight, deterministic heuristics – no large models, no network calls.

Signals (each mapped to [0, 1], then combined via weighted sum):

    Signal                  Weight   Direction
    ─────────────────────── ──────── ──────────────
    Sentence length reward    0.10   longer → formal (capped to avoid
                                     rewarding run-on text)
    Average word length       0.10   longer words → formal
    Long-word ratio           0.05   ≥ 7-char words → formal
    Contraction / slang       0.10   contractions → informal
    Formal/informal markers   0.20   keyword balance
    Informal-word density     0.15   raw slang-word fraction → informal
    Exclamation / question    0.10   excessive !/?  → informal
    Emoji penalty             0.10   emoji present  → informal
    Pronoun ratio             0.10   1st/2nd person → informal

The function **never raises**; it returns 0.5 for ``None``, empty, or
otherwise degenerate input.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from src.feature_extraction.feature_utils import (
    avg_sentence_length,
    clean_text,
    sentence_split,
    word_tokenize,
)

logger = logging.getLogger(__name__)

_DEFAULT_FORMALITY: float = 0.5

# ── Pre-compiled patterns ────────────────────────────────────────────────
_RE_CONTRACTION = re.compile(r"[a-zA-Z]+n't|[a-zA-Z]+'[a-zA-Z]+")
_RE_ALPHA_WORD = re.compile(r"^[a-zA-Z]+$")

# Informal markers (lower-cased)
_INFORMAL_MARKERS: frozenset[str] = frozenset(
    {
        "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "yeah",
        "yep", "nope", "nah", "hey", "hi", "yo", "wow", "omg", "lol",
        "lmao", "haha", "hehe", "btw", "fyi", "imo", "imho", "tbh",
        "ok", "okay", "cool", "dude", "bro", "bruh", "stuff", "thing",
        "things", "guy", "guys", "ain't", "aint", "sup", "cuz", "tho",
        "prolly", "ya", "yolo", "smh", "rofl",
    }
)

# Formal markers (lower-cased)
_FORMAL_MARKERS: frozenset[str] = frozenset(
    {
        "furthermore", "moreover", "nevertheless", "notwithstanding",
        "consequently", "accordingly", "henceforth", "therefore", "thus",
        "hence", "whereas", "whereby", "herein", "therein", "thereof",
        "pursuant", "aforementioned", "subsequently", "preceding",
        "respectively", "comprehensive", "encompassing", "pertaining",
        "demonstrate", "illustrate", "constitute", "facilitate",
        "implement", "acquisition", "endeavour", "endeavor",
        "manufacture", "calibrate", "meticulous", "distinguished",
    }
)

# First/second-person pronouns (informal in many registers)
_FIRST_SECOND_PRONOUNS: frozenset[str] = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
     "ourselves", "you", "your", "yours", "yourself", "yourselves"}
)


# ── Helper functions ──────────────────────────────────────────────────────

def _contraction_ratio(tokens: list[str], text: str) -> float:
    """Fraction of tokens that are contractions."""
    if not tokens:
        return 0.0
    contraction_count = len(_RE_CONTRACTION.findall(text))
    return contraction_count / len(tokens)


def _long_word_ratio(tokens: list[str]) -> float:
    """Fraction of alpha tokens with ≥ 7 characters (proxy for sophistication)."""
    alpha_tokens = [t for t in tokens if _RE_ALPHA_WORD.match(t)]
    if not alpha_tokens:
        return 0.0
    return sum(1 for t in alpha_tokens if len(t) >= 7) / len(alpha_tokens)


def _avg_word_length(tokens: list[str]) -> float:
    """Mean character-length of alpha tokens."""
    alpha_tokens = [t for t in tokens if _RE_ALPHA_WORD.match(t)]
    if not alpha_tokens:
        return 0.0
    return sum(len(t) for t in alpha_tokens) / len(alpha_tokens)


def _marker_score(tokens: list[str]) -> float:
    """
    Value in [-1, 1] based on formal-vs-informal marker density.
    Positive = more formal, negative = more informal.
    """
    if not tokens:
        return 0.0
    lower_tokens = [t.lower() for t in tokens]
    formal_hits = sum(1 for t in lower_tokens if t in _FORMAL_MARKERS)
    informal_hits = sum(1 for t in lower_tokens if t in _INFORMAL_MARKERS)
    total = formal_hits + informal_hits
    if total == 0:
        return 0.0
    return (formal_hits - informal_hits) / total


def _informal_density(tokens: list[str]) -> float:
    """
    Fraction of tokens that are informal markers.

    Unlike :func:`_marker_score` (which balances formal vs. informal hits),
    this measures the *raw density* of informal words in the token stream.
    A text like "hey dude lol gonna yeah" has ~100 % density.
    """
    if not tokens:
        return 0.0
    lower_tokens = [t.lower() for t in tokens]
    hits = sum(1 for t in lower_tokens if t in _INFORMAL_MARKERS)
    return hits / len(lower_tokens)


def _emoji_ratio(text: str) -> float:
    """
    Fraction of characters that are emoji (Unicode category ``So``
    or in common emoji ranges).  Returns 0.0 for empty text.

    Heuristic: any codepoint in the Emoticons, Dingbats,
    Miscellaneous-Symbols, or Supplemental-Symbols blocks, plus
    category ``So`` (Symbol, other) above U+00FF.
    """
    if not text:
        return 0.0
    emoji_count = sum(
        1 for ch in text
        if ord(ch) > 0xFF and unicodedata.category(ch) == "So"
    )
    return emoji_count / len(text)


def _pronoun_ratio(tokens: list[str]) -> float:
    """Fraction of alpha tokens that are first/second-person pronouns."""
    alpha_tokens = [t for t in tokens if _RE_ALPHA_WORD.match(t)]
    if not alpha_tokens:
        return 0.0
    hits = sum(1 for t in alpha_tokens if t.lower() in _FIRST_SECOND_PRONOUNS)
    return hits / len(alpha_tokens)


# ── Public API ────────────────────────────────────────────────────────────

def extract_formality(text: str | None) -> float:
    """
    Return a formality score in **[0.0, 1.0]**.

    0.0 = very informal, 1.0 = very formal.
    Deterministic – same input always yields the same output.

    Heuristic signals (weighted blend, then clamped to [0, 1]):

    1. Sentence-length reward (capped at ~25 words to avoid rewarding
       run-on text).
    2. Average word-length reward (longer words → more formal).
    3. Long-word ratio (≥7 characters).
    4. Contraction / slang penalty.
    5. Formal / informal marker keywords.
    6. Excessive exclamation / question-mark penalty.
    7. Emoji penalty.
    8. First / second-person pronoun ratio penalty.

    The function **never raises**; returns ``0.5`` on any failure.
    """
    try:
        cleaned = clean_text(text)
        if not cleaned:
            return _DEFAULT_FORMALITY

        tokens = word_tokenize(cleaned)
        if not tokens:
            return _DEFAULT_FORMALITY

        # ── Individual signals (each mapped to [0.0, 1.0]) ───────────

        # 1. Sentence-length signal
        #    Short (≤ 3 words) → 0.0, plateau at ~25 words → 1.0
        asl = avg_sentence_length(cleaned)
        sl_signal = min(1.0, max(0.0, (asl - 3.0) / 22.0))

        # 2. Average word-length signal
        #    Short words (≤ 3 chars) → 0.0, long words (≥ 8 chars) → 1.0
        awl = _avg_word_length(tokens)
        awl_signal = min(1.0, max(0.0, (awl - 3.0) / 5.0))

        # 3. Long-word ratio → already [0, 1]
        lw_signal = _long_word_ratio(tokens)

        # 4. Contraction penalty → 0 contractions = 1.0, heavy = 0.0
        cr = _contraction_ratio(tokens, cleaned)
        contr_signal = max(0.0, 1.0 - cr * 5.0)

        # 5. Marker signal → [-1, 1] → remap to [0, 1]
        ms = _marker_score(tokens)
        marker_signal = (ms + 1.0) / 2.0

        # 5b. Informal-density penalty → high density of slang tokens → 0.0
        #     Scaled aggressively: 40 % informal words → signal = 0.0
        inf_d = _informal_density(tokens)
        informal_density_signal = max(0.0, 1.0 - inf_d * 2.5)

        # 6. Exclamation + question-mark penalty
        excl_qmark_count = cleaned.count("!") + cleaned.count("?")
        eq_ratio = excl_qmark_count / len(cleaned)
        excl_signal = max(0.0, 1.0 - eq_ratio * 40.0)

        # 7. Emoji penalty
        er = _emoji_ratio(cleaned)
        emoji_signal = max(0.0, 1.0 - er * 50.0)

        # 8. First/second-person pronoun penalty
        pr = _pronoun_ratio(tokens)
        pronoun_signal = max(0.0, 1.0 - pr * 3.0)

        # ── Weighted blend ────────────────────────────────────────────
        #
        # Weights sum to 1.0.  The marker and informal-density signals
        # together carry 0.35, ensuring that slang-heavy text reliably
        # scores below 0.5 even when all other "absence-of-negative"
        # signals (no contractions, no emoji, etc.) default to 1.0.
        score = (
            0.10 * sl_signal
            + 0.10 * awl_signal
            + 0.05 * lw_signal
            + 0.10 * contr_signal
            + 0.20 * marker_signal
            + 0.15 * informal_density_signal
            + 0.10 * excl_signal
            + 0.10 * emoji_signal
            + 0.10 * pronoun_signal
        )

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, score))

    except Exception:
        logger.exception("extract_formality failed – returning default")
        return _DEFAULT_FORMALITY
