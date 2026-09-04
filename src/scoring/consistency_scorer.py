# src/scoring/consistency_scorer.py
# Person C — Consistency Scorer
#
# Compares a new text's features to a brand profile and returns five scores.
# ScoreResult field names are FROZEN — Person D's API and UI bind to them.
#
# All feature functions are imported from brand_profile_builder rather than
# redefined here. The profile and the scorer MUST measure a text the same way;
# when the two modules each kept their own copy of the Flesch and sentiment
# functions they drifted apart, and a text was compared against a mean that had
# been computed by different maths.

import math
from dataclasses import dataclass, asdict

from src.profiles.brand_profile_builder import (
    flesch_score,
    sentiment_proxy,
    vocab_richness,
    formality_proxy,
    mean_sentence_length,
    _content_words,
    _tokenize,
    BrandProfileNotFoundError,
)

MIN_WORDS = 10              # below this, scores are not statistically meaningful
MIN_SENTIMENT_SIGMA = 0.15  # floor on the brand's sentiment spread — see below

# Weights — see docs/scoring_spec_v2.md for the rationale behind each.
W_TONE = 0.30
W_SENTIMENT = 0.25
W_VOCAB = 0.25
W_READABILITY = 0.20

# ── Preset weight profiles (Phase 4A deliverable 2) ──────────────────────────
#
# Not every brand weighs the same things. A heritage brand is defined by its
# vocabulary — the words are the asset, and using the wrong one is the error
# that matters. A challenger brand is defined by its register; it can talk
# about anything as long as it sounds like itself. The presets let one scorer
# serve both instead of pretending every brand has identical priorities.
#
# Every preset sums to 1.0. Adding one is a dictionary entry, not a code change.
WEIGHT_PRESETS = {
    # Even-handed. The right default when nothing is known about the brand.
    "balanced": {"tone": 0.30, "sentiment": 0.25, "vocab": 0.25, "readability": 0.20},

    # Register above all — for brands whose identity is how they speak rather
    # than what they say. Vocabulary drops because a distinctive voice survives
    # a change of subject.
    "tone_heavy": {"tone": 0.45, "sentiment": 0.25, "vocab": 0.15, "readability": 0.15},

    # Vocabulary and meaning above all — for brands built on owned terminology,
    # where using a competitor's word for something is the real error.
    "semantic_heavy": {"tone": 0.30, "sentiment": 0.15, "vocab": 0.40, "readability": 0.15},
}

DEFAULT_PRESET = "balanced"


def resolve_weights(preset=None):
    """
    Return a weight mapping.

    `preset` may be a preset name, a dict of custom weights, or None for the
    default. Custom dicts are normalised to sum to 1.0 so a caller cannot
    accidentally inflate every score by passing weights that sum above one.
    Unknown names fall back to the default rather than raising — a bad preset
    from the UI should not take the endpoint down.
    """
    if isinstance(preset, dict):
        weights = {k: float(preset.get(k, 0.0))
                   for k in ("tone", "sentiment", "vocab", "readability")}
        total = sum(weights.values())
        if total <= 0:
            return dict(WEIGHT_PRESETS[DEFAULT_PRESET])
        return {k: v / total for k, v in weights.items()}

    name = (preset or DEFAULT_PRESET)
    if isinstance(name, str):
        name = name.strip().lower().replace("-", "_").replace(" ", "_")
    return dict(WEIGHT_PRESETS.get(name, WEIGHT_PRESETS[DEFAULT_PRESET]))


# ── Frozen output contract (do NOT rename fields) ─────────────────────────────

@dataclass
class ScoreResult:
    overall_score: float            # 0–100
    tone_pct: float                 # 0–100
    vocab_overlap_pct: float        # 0–100
    sentiment_alignment_pct: float  # 0–100
    readability_match_pct: float    # 0–100

    def to_dict(self):
        return asdict(self)


# ── Exceptions ────────────────────────────────────────────────────────────────

class EmbeddingDimensionError(Exception):
    """Text embedding and brand embedding have different dimensions."""


class FeatureExtractionError(Exception):
    """A required feature was missing, non-numeric or NaN."""


# BrandProfileNotFoundError is defined in brand_profile_builder and re-exported
# here so callers can import every scoring exception from one place.
__all__ = [
    "ScoreResult", "score_consistency", "extract_text_features",
    "BrandProfileNotFoundError", "EmbeddingDimensionError",
    "FeatureExtractionError",
]


# ── Similarity primitives ─────────────────────────────────────────────────────

def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


def gaussian_similarity(value, mean, std):
    """
    exp(-((value - mean)^2 / (2 * std^2))), range (0, 1].

    Returns 1.0 when the value sits exactly on the brand mean and decays
    smoothly with distance, scaled by how much the brand itself varies:
    a brand that writes consistently is judged strictly, one that varies
    is judged loosely. std is floored at 0.01 to avoid division collapse.
    """
    std = max(abs(std), 0.01)
    return math.exp(-((value - mean) ** 2) / (2 * std ** 2))


def inverse_distance(value, mean, tolerance):
    """max(0, 1 - |value - mean| / tolerance), range [0, 1]."""
    tolerance = max(abs(tolerance), 1.0)
    return max(0.0, 1.0 - abs(value - mean) / tolerance)


def cosine_similarity(a, b):
    """Cosine similarity of two vectors. Zero vector -> 0."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise EmbeddingDimensionError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def vocabulary_coverage(text_words, brand_keywords):
    """
    Length-fair coverage of the brand's characteristic vocabulary, range [0, 1].

        hits     = |unique content words of text  ∩  brand keywords|
        expected = min(|brand_keywords|, max(5, unique_content_words / 3))
        coverage = min(1, hits / expected)

    Plain Jaccard is wrong here because the two sets are different sizes by
    design: a brand profile holds ~15 characteristic words while an input text
    may contain 200 unique words. Jaccard divides by the union, so a text that
    used EVERY brand keyword perfectly would still score about 7%.

    Coverage asks the question a brand manager actually asks — "how much of our
    vocabulary is in this copy?" — and the `expected` denominator scales with
    text length, so a 30-word caption is not held to the same absolute hit
    count as a 300-word page.
    """
    text_set = set(text_words)
    brand_set = set(brand_keywords)
    if not brand_set or not text_set:
        return 0.0
    hits = len(text_set & brand_set)
    expected = min(len(brand_set), max(5.0, len(text_set) / 3.0))
    return min(1.0, hits / expected)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_text_features(text):
    """
    Compute every feature the scorer needs from raw text.

    Person B's pipeline may supply richer values (model-based sentiment,
    embeddings); anything it supplies overrides what is computed here, and
    anything it omits falls back to these CPU-only proxies so the system
    always runs.
    """
    return {
        "text": text,
        "sentiment_score": sentiment_proxy(text),
        "flesch_reading_ease": flesch_score(text),
        "vocab_richness": vocab_richness(text),
        "formality": formality_proxy(text),
        "sentence_length": mean_sentence_length(text),
        "content_words": _content_words(text),
        "embedding": [],
    }


def _resolve(features, key, fallback_fn, text):
    """Use a pre-computed feature if present, otherwise compute it."""
    value = features.get(key)
    if value is None:
        value = fallback_fn(text)
    return value


def _check_numeric(name, value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise FeatureExtractionError(f"Non-numeric value in field: {name}")
    if math.isnan(f) or math.isinf(f):
        raise FeatureExtractionError(f"NaN or infinite value in feature field: {name}")
    return f


# ── Tone ──────────────────────────────────────────────────────────────────────

def _style_tone(features, profile, text):
    """
    Tone without embeddings: agreement on three register dimensions.

        formality        weight 0.50
        vocabulary richness  0.30
        sentence length      0.20

    Each is a Gaussian similarity against the brand's own mean and spread.

    The previous fallback was `1 - |sentiment - brand_mean_sentiment|`, which
    is the sentiment metric a second time. Because tone carries the heaviest
    weight (0.30) and sentiment carries 0.25, that put 55% of the overall score
    on one crude lexicon count — a sentence about pizza delivery scored 68 on
    tone against the Rolex profile. Formality, lexical variety and sentence
    length are what actually separate "This timepiece embodies enduring
    precision" from "this watch is super easy to wear", and none of them
    duplicate a signal already being measured elsewhere.
    """
    formality = _check_numeric("formality", _resolve(features, "formality", formality_proxy, text))
    richness = _check_numeric("vocab_richness", _resolve(features, "vocab_richness", vocab_richness, text))
    sent_len = _check_numeric("sentence_length", _resolve(features, "sentence_length", mean_sentence_length, text))

    g_formality = gaussian_similarity(
        formality,
        float(profile.get("mean_formality", 0.5)),
        float(profile.get("std_formality", 0.1)),
    )
    g_richness = gaussian_similarity(
        richness,
        float(profile.get("mean_vocab_richness", 0.6)),
        float(profile.get("std_vocab_richness", 0.1)),
    )
    g_length = gaussian_similarity(
        sent_len,
        float(profile.get("mean_sentence_length", 20.0)),
        float(profile.get("std_sentence_length", 8.0)),
    )
    return 0.50 * g_formality + 0.30 * g_richness + 0.20 * g_length


# ── Main scoring function (API contract — field names frozen) ─────────────────

def score_consistency(text_features, brand_profile, preset=None):
    """
    Compare a new text's features against a brand profile.

    Parameters
    ----------
    text_features : dict
        Must contain "text". May also contain pre-computed
        "sentiment_score", "flesch_reading_ease", "vocab_richness",
        "formality", "sentence_length", "top_keywords", "embedding".
    brand_profile : dict
        Parsed profile_json from the brand_profiles table.

    Returns
    -------
    ScoreResult — five floats in [0, 100].

    Raises
    ------
    FeatureExtractionError   if a feature is missing, non-numeric or NaN
    EmbeddingDimensionError  if embedding lengths differ
    """
    if brand_profile is None:
        raise BrandProfileNotFoundError("brand_profile is None")

    text = text_features.get("text", "") or ""
    words = _tokenize(text)

    # ── Short-text guard ──────────────────────────────────────────────────────
    # Below ~10 words the style statistics are noise: one long word swings
    # formality, one sentence swings readability. Returning zeros lets the API
    # answer error="text_too_short" instead of publishing a confident number
    # it cannot justify.
    if len(words) < MIN_WORDS:
        return ScoreResult(0.0, 0.0, 0.0, 0.0, 0.0)

    # ── Features (pre-computed values win; proxies fill the gaps) ─────────────
    sentiment = _check_numeric(
        "sentiment", _resolve(text_features, "sentiment_score", sentiment_proxy, text))
    flesch = _check_numeric(
        "flesch", _resolve(text_features, "flesch_reading_ease", flesch_score, text))

    content = text_features.get("content_words")
    if content is None:
        content = _content_words(text)

    embedding = text_features.get("embedding") or []
    brand_embedding = brand_profile.get("mean_embedding") or []

    # ── Brand stats ───────────────────────────────────────────────────────────
    # Profiles built by the older script store mean_sentiment on a 0..1 scale.
    # This module measures on a signed -1..1 scale and stamps its own profiles
    # with sentiment_scale="signed". Without this check a text would be compared
    # against a mean expressed in different units, and every sentiment score
    # would be quietly wrong.
    if brand_profile.get("sentiment_scale") != "signed":
        sentiment = (sentiment + 1.0) / 2.0

    mean_sentiment = float(brand_profile.get("mean_sentiment", 0.0))
    std_sentiment = float(brand_profile.get("std_sentiment", 0.1))
    mean_flesch = float(brand_profile.get("mean_flesch", 50.0))
    std_flesch = float(brand_profile.get("std_flesch", 10.0))
    brand_keywords = brand_profile.get("top_keywords") or []
    name_tokens = set(brand_profile.get("brand_name_tokens") or [])

    # ── 1) Vocabulary overlap ────────────────────────────────────────────────
    # Brand-name tokens are removed before comparison. Phase 4A requires name
    # mentions to be a NEUTRAL signal; if they counted, repeating the brand
    # name would be the cheapest possible way to score well.
    scored_words = [w for w in content if w not in name_tokens]
    vocab = vocabulary_coverage(scored_words, brand_keywords)

    # ── 2) Sentiment alignment ───────────────────────────────────────────────
    # std is floored at MIN_SENTIMENT_SIGMA. A Gaussian falls to ~1% at three
    # standard deviations, so a brand whose corpus happens to be emotionally
    # uniform would score almost any real copy at zero. The floor keeps the
    # metric a measure of alignment rather than a near-binary gate.
    sentiment_align = gaussian_similarity(
        sentiment, mean_sentiment, max(std_sentiment, MIN_SENTIMENT_SIGMA))

    # ── 3) Readability match ─────────────────────────────────────────────────
    # Tolerance follows the brand's own variability, floored at 15 Flesch
    # points so a very consistent brand does not become impossible to match.
    tolerance = max(2.0 * std_flesch, 15.0)
    readability = inverse_distance(flesch, mean_flesch, tolerance)

    # ── 4) Tone ──────────────────────────────────────────────────────────────
    if brand_embedding and embedding:
        tone = max(0.0, cosine_similarity(embedding, brand_embedding))
    else:
        tone = _style_tone(text_features, brand_profile, text)

    # ── Overall ──────────────────────────────────────────────────────────────
    # Weights come from the preset, falling back to the profile's own stored
    # preference and then to "balanced". Storing a preference on the profile
    # means a brand keeps its weighting without every caller having to remember
    # to pass it.
    weights = resolve_weights(preset or brand_profile.get("weight_preset"))
    overall = (
        weights["tone"] * tone
        + weights["sentiment"] * sentiment_align
        + weights["vocab"] * vocab
        + weights["readability"] * readability
    ) * 100.0

    return ScoreResult(
        overall_score=_clamp(overall),
        tone_pct=_clamp(tone * 100.0),
        vocab_overlap_pct=_clamp(vocab * 100.0),
        sentiment_alignment_pct=_clamp(sentiment_align * 100.0),
        readability_match_pct=_clamp(readability * 100.0),
    )
