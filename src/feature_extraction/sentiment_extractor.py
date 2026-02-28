"""
sentiment_extractor.py – Lightweight lexicon-based sentiment scorer.

Uses a compact hand-curated word-list approach (no heavy-NLP deps).
Delegates tokenisation to ``feature_utils.word_tokenize`` so that the
tokenisation logic is never duplicated.

Score contract: **[0.0, 1.0]** (0 = very negative, 0.5 = neutral, 1.0 = very positive).
"""

from __future__ import annotations

import logging

from src.feature_extraction.feature_utils import clean_text, word_tokenize

logger = logging.getLogger(__name__)

# ── Lexicons (compact, brand-copy oriented) ───────────────────────────────
# Positive words frequently found in luxury / watch / brand marketing copy.
_POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        # General positive
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "outstanding", "brilliant", "superb", "perfect", "best", "love",
        "loved", "loving", "beautiful", "gorgeous", "stunning", "elegant",
        "happy", "glad", "pleased", "delighted", "enjoy", "enjoyed",
        "enjoyable", "impressive", "remarkable", "incredible", "awesome",
        "magnificent", "marvelous", "exceptional", "splendid", "fine",
        "nice", "lovely", "charming", "pleasant", "positive", "success",
        "successful", "win", "winning", "triumph", "achieve", "achievement",
        "superior", "premium", "exquisite", "refined", "distinguished",
        # Brand / luxury specific
        "luxury", "luxurious", "prestige", "prestigious", "iconic",
        "heritage", "craftsmanship", "masterpiece", "innovation",
        "innovative", "precision", "timeless", "exclusive", "elite",
        "sophisticated", "artisan", "bespoke", "classic", "legendary",
        "excellence", "quality", "reliable", "reliability", "trusted",
        "durable", "durability", "authentic", "authenticity",
        # Emotion / experience
        "inspire", "inspired", "inspiring", "passion", "passionate",
        "admire", "admired", "admiration", "celebrate", "celebrated",
        "treasure", "treasured", "cherish", "cherished",
    }
)

_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        # General negative
        "bad", "terrible", "horrible", "awful", "poor", "worst", "hate",
        "hated", "hating", "ugly", "boring", "dull", "disappointing",
        "disappointed", "disappointment", "sad", "unhappy", "angry",
        "annoying", "annoyed", "frustrating", "frustrated", "frustration",
        "mediocre", "inferior", "cheap", "flimsy", "broken", "defective",
        "failure", "fail", "failed", "failing", "useless", "worthless",
        "weak", "painful", "unpleasant", "negative", "problem", "problems",
        "mistake", "mistakes", "error", "errors", "flaw", "flawed",
        "damage", "damaged", "loss", "lost", "worse", "worsening",
        # Brand / product criticism
        "overpriced", "counterfeit", "fake", "knockoff", "unreliable",
        "fragile", "scratched", "tarnished", "dated", "outdated",
        "generic", "ordinary", "common", "tacky",
    }
)

# Negation words that flip the next sentiment-carrying word.
_NEGATORS: frozenset[str] = frozenset(
    {
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "nor", "cannot", "without", "hardly", "barely", "scarcely",
    }
)

# Intensifiers that boost the magnitude of the next sentiment word.
_INTENSIFIERS: frozenset[str] = frozenset(
    {
        "very", "extremely", "incredibly", "remarkably", "truly",
        "absolutely", "totally", "utterly", "highly", "exceptionally",
        "really", "so", "most", "deeply",
    }
)

# ── Constants ─────────────────────────────────────────────────────────────
_DEFAULT_SENTIMENT: float = 0.5
_INTENSIFIER_BOOST: float = 0.5  # extra weight when preceded by intensifier


def extract_sentiment(text: str | None) -> float:
    """
    Return a sentiment score in **[0.0, 1.0]**.

    0.0 = very negative, 0.5 = neutral, 1.0 = very positive.

    Algorithm
    ---------
    1. Tokenise with ``feature_utils.word_tokenize``.
    2. Walk through tokens; track *negation* and *intensifier* state.
    3. Accumulate score (+1 for positive, -1 for negative) with modifiers.
    4. Normalise polarity to [-1, 1] using ``score / (|score| + α)`` (smooth clamp).
    5. Map to [0, 1] via ``(polarity + 1) / 2``.

    The function **never raises**; returns 0.5 on any failure.
    """
    try:
        cleaned = clean_text(text)
        if not cleaned:
            return _DEFAULT_SENTIMENT

        tokens = word_tokenize(cleaned)
        if not tokens:
            return _DEFAULT_SENTIMENT

        lower_tokens = [t.lower() for t in tokens]

        score: float = 0.0
        negate: bool = False
        intensify: bool = False

        for tok in lower_tokens:
            # Check negation / intensifier modifiers first
            if tok in _NEGATORS:
                negate = True
                continue
            if tok in _INTENSIFIERS:
                intensify = True
                continue

            # Score sentiment words
            word_score: float = 0.0
            if tok in _POSITIVE_WORDS:
                word_score = 1.0
            elif tok in _NEGATIVE_WORDS:
                word_score = -1.0

            if word_score != 0.0:
                if intensify:
                    word_score *= (1.0 + _INTENSIFIER_BOOST)
                if negate:
                    word_score *= -1.0
                score += word_score

            # Reset modifiers after any non-modifier token
            negate = False
            intensify = False

        # Smooth normalisation: score / (|score| + α)
        # α controls how quickly we approach ±1.  α=5 means you need ~5
        # net sentiment hits to reach ±0.5.
        alpha: float = 5.0
        polarity = score / (abs(score) + alpha)

        # Map from [-1, 1] polarity to [0, 1] sentiment
        mapped = (polarity + 1.0) / 2.0

        # Hard clamp for safety
        return max(0.0, min(1.0, mapped))

    except Exception:
        logger.exception("extract_sentiment failed – returning default")
        return _DEFAULT_SENTIMENT
