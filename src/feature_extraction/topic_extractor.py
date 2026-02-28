"""
topic_extractor.py – Lightweight keyword-frequency topic extractor.

Two-layer scoring approach (no heavy-NLP deps):

1. **Domain lexicon layer** – curated topic categories with indicator words
   give instant signal for brand/watch domain text.
2. **TF keyword layer** – after removing a built-in stopword list, compute
   term-frequency, boost longer tokens, penalise very short ones, and pick
   the top keywords.  This layer generalises to *any* text, not just domain
   copy.

The two layers are fused by normalised score addition so that domain terms
rank high when present while generic keywords still surface for out-of-domain
text.

Returns ``(top_topics, topic_weights)`` where weights sum to ~1.0.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

from src.feature_extraction.feature_utils import clean_text, word_tokenize

logger = logging.getLogger(__name__)

# ── Built-in stopword list (compact, English) ─────────────────────────────
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "not", "no", "nor", "so", "yet", "both", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "am", "any",
    "because", "before", "below", "between", "come", "during", "even",
    "get", "go", "going", "got", "he", "her", "here", "him", "his",
    "how", "i", "into", "it", "its", "let", "like", "me", "my",
    "now", "only", "our", "out", "over", "own", "re", "same", "she",
    "so", "still", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "under", "until", "up", "upon",
    "us", "want", "we", "what", "when", "where", "which", "while",
    "who", "whom", "why", "you", "your",
})

# ── Domain topic lexicons ─────────────────────────────────────────────────
# Each key is a human-readable topic label; value is a set of indicator words.
_TOPIC_LEXICONS: dict[str, frozenset[str]] = {
    "craftsmanship": frozenset({
        "craft", "craftsmanship", "handmade", "handcrafted", "artisan",
        "artisanal", "bespoke", "workshop", "atelier", "master",
        "watchmaker", "horologist", "horology", "assemble", "assembled",
        "finish", "finishing", "polish", "polished", "engrave", "engraved",
        "guilloché", "enamel", "lacquer", "dial", "movement", "caliber",
        "calibre", "complication", "tourbillon", "perpetual", "chronograph",
        "manufacture", "manufactured",
    }),
    "heritage": frozenset({
        "heritage", "history", "historical", "tradition", "traditional",
        "legacy", "founded", "origin", "origins", "century", "decades",
        "since", "established", "founding", "ancestor", "ancestral",
        "archive", "archives", "vintage", "antique", "classic", "classical",
        "iconic", "timeless", "enduring", "generations",
    }),
    "innovation": frozenset({
        "innovation", "innovative", "technology", "technological",
        "patent", "patented", "breakthrough", "pioneer", "pioneering",
        "cutting", "edge", "advanced", "advancement", "modern",
        "contemporary", "future", "futuristic", "digital", "smart",
        "sensor", "silicon", "ceramic", "titanium", "carbon", "fiber",
        "sapphire", "luminescent", "luminova", "superluminova",
        "antimagnetic", "waterproof", "resistant",
    }),
    "luxury": frozenset({
        "luxury", "luxurious", "prestige", "prestigious", "exclusive",
        "exclusivity", "premium", "elite", "opulent", "opulence",
        "refined", "refinement", "sophisticated", "sophistication",
        "elegant", "elegance", "exquisite", "magnificent", "splendid",
        "sumptuous", "lavish", "precious", "gold", "platinum", "diamond",
        "diamonds", "ruby", "sapphire", "jewel", "jewels", "gem", "gems",
    }),
    "performance": frozenset({
        "performance", "precision", "accurate", "accuracy", "chronometer",
        "certified", "certification", "cosc", "test", "tested", "testing",
        "robust", "robustness", "reliable", "reliability", "durable",
        "durability", "tough", "toughness", "shock", "pressure", "depth",
        "meters", "bar", "atm", "speed", "racing", "sport", "sports",
        "dive", "diving", "pilot", "aviation", "explorer", "expedition",
    }),
    "design": frozenset({
        "design", "designed", "designer", "aesthetic", "aesthetics",
        "style", "stylish", "look", "looks", "appearance", "visual",
        "shape", "form", "line", "lines", "curve", "curves", "contour",
        "profile", "silhouette", "face", "bezel", "crown", "bracelet",
        "strap", "leather", "rubber", "steel", "stainless", "case",
        "slim", "thin", "bold", "minimalist", "minimal",
    }),
    "lifestyle": frozenset({
        "lifestyle", "life", "living", "experience", "adventure",
        "travel", "journey", "discover", "discovery", "explore",
        "exploring", "wear", "wearing", "everyday", "daily", "occasion",
        "occasions", "formal", "casual", "dress", "fashion", "trend",
        "trendy", "celebrity", "ambassador", "brand", "collection",
        "limited", "edition", "series", "model", "range",
    }),
    "sustainability": frozenset({
        "sustainable", "sustainability", "environment", "environmental",
        "eco", "green", "ethical", "responsible", "responsibility",
        "recycle", "recycled", "renewable", "solar", "energy", "ocean",
        "conservation", "protect", "protection", "community", "social",
        "fair", "trade", "carbon", "neutral", "footprint", "impact",
    }),
}

# All domain keywords in one flat set (for quick membership checks)
_ALL_DOMAIN_KEYWORDS: frozenset[str] = frozenset().union(
    *_TOPIC_LEXICONS.values()
)

# ── Word-length scoring helpers ───────────────────────────────────────────
_MIN_KEYWORD_LEN = 2     # tokens shorter than this are penalised
_LONG_WORD_THRESHOLD = 6  # tokens >= this length get a boost


def _length_boost(word: str) -> float:
    """Return a multiplier based on word length."""
    n = len(word)
    if n < _MIN_KEYWORD_LEN:
        return 0.25          # heavy penalty for 1-char tokens
    if n >= _LONG_WORD_THRESHOLD:
        return 1.0 + 0.1 * (n - _LONG_WORD_THRESHOLD)  # gentle boost
    return 1.0


# ── Core scoring ──────────────────────────────────────────────────────────

def _score_domain_topics(lower_tokens: list[str]) -> dict[str, float]:
    """Score curated domain topics by keyword hit-count."""
    token_counts = Counter(lower_tokens)
    scores: dict[str, float] = {}
    for topic, keywords in _TOPIC_LEXICONS.items():
        hits = sum(token_counts.get(kw, 0) for kw in keywords)
        scores[topic] = float(hits)
    return scores


def _score_tf_keywords(lower_tokens: list[str]) -> dict[str, float]:
    """
    Score individual tokens by TF × length-boost after removing stopwords
    and non-alphabetic tokens.
    """
    filtered = [
        t for t in lower_tokens
        if t.isalpha() and t not in _STOPWORDS and len(t) >= _MIN_KEYWORD_LEN
    ]
    if not filtered:
        return {}

    tf = Counter(filtered)
    scored: dict[str, float] = {}
    for word, count in tf.items():
        scored[word] = count * _length_boost(word)
    return scored


def _fuse_and_select(
    domain_scores: dict[str, float],
    tf_scores: dict[str, float],
    num_topics: int,
) -> tuple[list[str], list[float]]:
    """
    Merge domain-topic scores and TF-keyword scores into a single ranked
    list of ``num_topics`` entries with normalised weights.

    Strategy
    --------
    * Domain topics with score > 0 are added first (they carry a recognisable
      human-friendly label like "craftsmanship").
    * Remaining slots are filled with top TF keywords that are NOT already
      covered by a domain-topic keyword.
    * If still not enough, pad with ``"unknown"`` / 0.0.
    """
    # 1. Rank domain topics by score (descending), keep only positive
    ranked_domain = sorted(
        ((t, s) for t, s in domain_scores.items() if s > 0),
        key=lambda kv: (-kv[1], kv[0]),
    )

    # 2. Rank TF keywords (exclude any that are themselves domain labels or
    #    already captured by a domain keyword)
    ranked_tf = sorted(
        ((w, s) for w, s in tf_scores.items() if w not in domain_scores),
        key=lambda kv: (-kv[1], kv[0]),
    )

    # 3. Fill slots
    selected: list[tuple[str, float]] = []
    for label, score in ranked_domain:
        if len(selected) >= num_topics:
            break
        selected.append((label, score))

    for word, score in ranked_tf:
        if len(selected) >= num_topics:
            break
        selected.append((word, score))

    # 4. Pad
    while len(selected) < num_topics:
        selected.append(("unknown", 0.0))

    labels = [lbl for lbl, _ in selected]
    raw = [s for _, s in selected]

    # 5. Normalise weights → sum to 1.0
    total = sum(raw)
    if total > 0:
        weights = [w / total for w in raw]
    else:
        weights = [0.0] * num_topics

    return labels, weights


# ── Public API ────────────────────────────────────────────────────────────

def extract_topics(
    text: str | None,
    num_topics: int = 5,
    *,
    n_topics: int | None = None,
) -> tuple[list[str], list[float]]:
    """
    Return ``(top_topics, topic_weights)``.

    Parameters
    ----------
    text : str | None
        Input text (cleaned internally).
    num_topics : int
        How many topics to return.  Alias: *n_topics*.
    n_topics : int | None
        Alternative parameter name (takes precedence over *num_topics*
        if supplied).

    Returns
    -------
    tuple[list[str], list[float]]
        * **top_topics** – list of topic labels, length = ``num_topics``.
        * **topic_weights** – normalised weights, same length, sum ≈ 1.0
          for meaningful text, all-zero for empty/None.

    Algorithm
    ---------
    1. Tokenise with ``feature_utils.word_tokenize``.
    2. Score curated domain-topic categories (keyword frequency).
    3. Score remaining tokens by TF × word-length boost (after stopword
       removal).
    4. Fuse, rank, and normalise to top ``num_topics`` entries.

    The function **never raises**; returns ``(["unknown"]*n, [0.0]*n)``
    on failure.
    """
    try:
        # Allow n_topics to override num_topics
        if n_topics is not None:
            num_topics = n_topics

        if num_topics < 1:
            num_topics = 1

        cleaned = clean_text(text)
        if not cleaned:
            return ["unknown"] * num_topics, [0.0] * num_topics

        tokens = word_tokenize(cleaned)
        if not tokens:
            return ["unknown"] * num_topics, [0.0] * num_topics

        lower_tokens = [t.lower() for t in tokens]

        # Layer 1 – domain lexicon scores
        domain_scores = _score_domain_topics(lower_tokens)

        # Layer 2 – TF keyword scores
        tf_scores = _score_tf_keywords(lower_tokens)

        return _fuse_and_select(domain_scores, tf_scores, num_topics)

    except Exception:
        logger.exception("extract_topics failed – returning defaults")
        return ["unknown"] * max(num_topics, 1), [0.0] * max(num_topics, 1)
