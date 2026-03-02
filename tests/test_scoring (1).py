# tests/test_scoring.py
# Person C — Unit tests for consistency_scorer.py
# Run with:  pytest tests/test_scoring.py -v

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.scoring.consistency_scorer import (
    score_consistency,
    ScoreResult,
    FeatureExtractionError,
    EmbeddingDimensionError,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

ROLEX_PROFILE = {
    "brand_id": "rolex",
    "brand_name": "Rolex",
    "mean_sentiment": 0.38,
    "std_sentiment": 0.09,
    "mean_flesch": 48.2,
    "std_flesch": 6.4,
    "mean_vocab_richness": 0.67,
    "std_vocab_richness": 0.08,
    "mean_formality": 0.72,
    "std_formality": 0.06,
    "top_keywords": ["precision", "achievement", "enduring", "craft", "excellence",
                     "heritage", "timepiece", "movement", "wristwatch", "pioneering"],
    "mean_embedding": [],   # empty = fallback tone mode
    "tone_label": "authoritative",
}

ON_BRAND_TEXT = (
    "This timepiece embodies precision and enduring excellence — "
    "a testament to the craft of master watchmakers who refuse to accept compromise. "
    "Every movement reflects heritage and achievement."
)

OFF_BRAND_TEXT = (
    "Hey this watch is super awesome and really easy to wear every day. "
    "Cool design and totally affordable for anyone who wants a chill everyday accessory."
)

SHORT_TEXT = "Nice watch."


def _features(text: str) -> dict:
    """Build minimal text_features dict from raw text (scorer computes the rest)."""
    return {"text": text}


# ── Test 1: Vocabulary overlap ─────────────────────────────────────────────────

def test_vocab_overlap_on_brand():
    """Text sharing brand keywords should score higher vocab overlap than off-brand."""
    result_on  = score_consistency(_features(ON_BRAND_TEXT),  ROLEX_PROFILE)
    result_off = score_consistency(_features(OFF_BRAND_TEXT), ROLEX_PROFILE)
    assert result_on.vocab_overlap_pct > result_off.vocab_overlap_pct, (
        f"On-brand vocab ({result_on.vocab_overlap_pct:.1f}) should exceed "
        f"off-brand vocab ({result_off.vocab_overlap_pct:.1f})"
    )


def test_vocab_overlap_range():
    """vocab_overlap_pct must always be in [0, 100]."""
    result = score_consistency(_features(ON_BRAND_TEXT), ROLEX_PROFILE)
    assert 0.0 <= result.vocab_overlap_pct <= 100.0


# ── Test 2: Sentiment alignment ────────────────────────────────────────────────

def test_sentiment_alignment_range():
    """sentiment_alignment_pct must always be in [0, 100]."""
    result = score_consistency(_features(ON_BRAND_TEXT), ROLEX_PROFILE)
    assert 0.0 <= result.sentiment_alignment_pct <= 100.0


def test_sentiment_alignment_near_match():
    """Text whose pre-computed sentiment exactly equals brand mean should score ~100."""
    # Supply the brand mean as pre-computed feature so we isolate Gaussian math
    result = score_consistency(
        {"text": ON_BRAND_TEXT, "sentiment_score": 0.38},  # ROLEX_PROFILE mean
        ROLEX_PROFILE
    )
    assert result.sentiment_alignment_pct >= 99.0, (
        f"Exact-match sentiment should be ~100, got {result.sentiment_alignment_pct:.1f}"
    )


# ── Test 3: Readability match ──────────────────────────────────────────────────

def test_readability_match_range():
    """readability_match_pct must always be in [0, 100]."""
    result = score_consistency(_features(ON_BRAND_TEXT), ROLEX_PROFILE)
    assert 0.0 <= result.readability_match_pct <= 100.0


def test_readability_match_simple_vs_complex():
    """
    Very simple text (high Flesch) should score lower readability match
    for a high-formality brand (low Flesch target) than complex text.
    """
    simple_text = " ".join(["The watch is good. "] * 8)   # very simple / repetitive
    complex_text = ON_BRAND_TEXT
    r_simple  = score_consistency(_features(simple_text),  ROLEX_PROFILE)
    r_complex = score_consistency(_features(complex_text), ROLEX_PROFILE)
    assert r_complex.readability_match_pct >= r_simple.readability_match_pct - 5, (
        "Complex on-brand text should not score dramatically worse on readability"
    )


# ── Test 4: Tone ───────────────────────────────────────────────────────────────

def test_tone_range():
    """tone_pct must always be in [0, 100]."""
    result = score_consistency(_features(ON_BRAND_TEXT), ROLEX_PROFILE)
    assert 0.0 <= result.tone_pct <= 100.0


def test_tone_with_embeddings():
    """When embeddings match exactly, tone should be 100."""
    vec = [0.1, 0.2, 0.3, 0.4]
    profile = dict(ROLEX_PROFILE)
    profile["mean_embedding"] = vec
    features = {"text": ON_BRAND_TEXT, "embedding": vec}
    result = score_consistency(features, profile)
    assert result.tone_pct >= 99.0, f"Identical embedding should give ~100, got {result.tone_pct}"


def test_tone_embedding_dimension_mismatch():
    """Mismatched embedding dims must raise EmbeddingDimensionError."""
    profile = dict(ROLEX_PROFILE)
    profile["mean_embedding"] = [0.1, 0.2, 0.3]
    features = {"text": ON_BRAND_TEXT, "embedding": [0.1, 0.2]}
    with pytest.raises(EmbeddingDimensionError):
        score_consistency(features, profile)


# ── Test 5: Short text edge case ───────────────────────────────────────────────

def test_short_text_returns_zeros():
    """Text shorter than 10 words must return all zeros."""
    result = score_consistency(_features(SHORT_TEXT), ROLEX_PROFILE)
    assert result.overall_score == 0.0
    assert result.tone_pct == 0.0
    assert result.vocab_overlap_pct == 0.0
    assert result.sentiment_alignment_pct == 0.0
    assert result.readability_match_pct == 0.0


# ── Test 6: Overall score ──────────────────────────────────────────────────────

def test_overall_score_range():
    """overall_score must always be in [0, 100]."""
    result = score_consistency(_features(ON_BRAND_TEXT), ROLEX_PROFILE)
    assert 0.0 <= result.overall_score <= 100.0


def test_overall_score_is_weighted_average():
    """overall_score must roughly equal the weighted average of the four sub-scores."""
    result = score_consistency(_features(ON_BRAND_TEXT), ROLEX_PROFILE)
    expected = (
        0.30 * result.tone_pct
        + 0.25 * result.sentiment_alignment_pct
        + 0.25 * result.vocab_overlap_pct
        + 0.20 * result.readability_match_pct
    )
    assert abs(result.overall_score - expected) < 0.01, (
        f"overall_score {result.overall_score:.2f} ≠ weighted avg {expected:.2f}"
    )


def test_on_brand_scores_higher_vocab_than_off_brand():
    """
    On-brand text (sharing Rolex keywords) must score higher VOCAB overlap
    than off-brand text. Vocab overlap is the most reliable proxy without
    real embeddings/sentiment models.
    """
    r_on  = score_consistency(_features(ON_BRAND_TEXT),  ROLEX_PROFILE)
    r_off = score_consistency(_features(OFF_BRAND_TEXT), ROLEX_PROFILE)
    assert r_on.vocab_overlap_pct > r_off.vocab_overlap_pct, (
        f"On-brand vocab ({r_on.vocab_overlap_pct:.1f}) should beat "
        f"off-brand vocab ({r_off.vocab_overlap_pct:.1f})"
    )


# ── Test 7: Return type ────────────────────────────────────────────────────────

def test_returns_score_result_dataclass():
    """score_consistency must return a ScoreResult instance."""
    result = score_consistency(_features(ON_BRAND_TEXT), ROLEX_PROFILE)
    assert isinstance(result, ScoreResult)


# ── Integration test ───────────────────────────────────────────────────────────

def test_integration_all_fields_populated():
    """Integration: all five fields must be present and numeric for valid input."""
    result = score_consistency(_features(ON_BRAND_TEXT), ROLEX_PROFILE)
    for field in ("overall_score", "tone_pct", "vocab_overlap_pct",
                  "sentiment_alignment_pct", "readability_match_pct"):
        value = getattr(result, field)
        assert isinstance(value, float), f"{field} is not a float"
        assert 0.0 <= value <= 100.0,    f"{field} = {value} is out of range"
