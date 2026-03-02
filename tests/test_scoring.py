"""
Contract tests for the canonical scorer at src/scoring/consistency.py.

Asserts:
- Output keys match the frozen API schema exactly.
- All values are in [0, 100].
- Deterministic output for the same input.
- Short text (<10 words) returns all zeros.
- Missing / empty profile fields do not crash.
- On-brand text scores higher than off-brand text.
"""

import pytest

from src.scoring.consistency import compute_consistency_score, generate_edit_plan

EXPECTED_KEYS = {
    "overall_score",
    "tone_pct",
    "vocab_overlap_pct",
    "sentiment_alignment_pct",
    "readability_match_pct",
}

SAMPLE_PROFILE = {
    "brand_id": "rolex",
    "brand_name": "Rolex",
    "mean_sentiment": 0.5593,
    "std_sentiment": 0.0926,
    "mean_formality": 0.8045,
    "std_formality": 0.0231,
    "mean_flesch": 44.41,
    "std_flesch": 9.44,
    "top_keywords": [
        "rolex", "watch", "case", "oyster", "time",
        "first", "crown", "submariner", "hans", "wilsdorf",
    ],
    "tone_label": "formal",
}

ON_BRAND = (
    "The Oyster Perpetual embodies precision craftsmanship and perpetual "
    "excellence, a testament to enduring horological mastery."
)
OFF_BRAND = (
    "This watch is awesome and super easy to wear every day. Cool design "
    "and pretty nice overall."
)
SHORT_TEXT = "Hello world"


class TestOutputSchema:
    """Output must have exactly the five frozen keys."""

    def test_keys_exact_match(self):
        result = compute_consistency_score(ON_BRAND, SAMPLE_PROFILE)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_no_extra_keys(self):
        result = compute_consistency_score(ON_BRAND, SAMPLE_PROFILE)
        assert len(result) == len(EXPECTED_KEYS)


class TestValueRanges:
    """Every value must be a float in [0, 100]."""

    def test_all_in_range(self):
        result = compute_consistency_score(ON_BRAND, SAMPLE_PROFILE)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float"
            assert 0.0 <= val <= 100.0, f"{key}={val} out of [0, 100]"

    def test_off_brand_in_range(self):
        result = compute_consistency_score(OFF_BRAND, SAMPLE_PROFILE)
        for key, val in result.items():
            assert 0.0 <= val <= 100.0, f"{key}={val} out of [0, 100]"


class TestDeterminism:
    """Same input must always produce the same output."""

    def test_deterministic(self):
        a = compute_consistency_score(ON_BRAND, SAMPLE_PROFILE)
        b = compute_consistency_score(ON_BRAND, SAMPLE_PROFILE)
        assert a == b

    def test_deterministic_off_brand(self):
        a = compute_consistency_score(OFF_BRAND, SAMPLE_PROFILE)
        b = compute_consistency_score(OFF_BRAND, SAMPLE_PROFILE)
        assert a == b


class TestShortText:
    """Text with fewer than 10 words must return all zeros."""

    def test_short_returns_zeros(self):
        result = compute_consistency_score(SHORT_TEXT, SAMPLE_PROFILE)
        for key, val in result.items():
            assert val == 0.0, f"{key}={val}, expected 0.0 for short text"

    def test_empty_string_returns_zeros(self):
        result = compute_consistency_score("", SAMPLE_PROFILE)
        for key, val in result.items():
            assert val == 0.0

    def test_none_text_returns_zeros(self):
        result = compute_consistency_score(None, SAMPLE_PROFILE)
        for key, val in result.items():
            assert val == 0.0


class TestMissingProfileFields:
    """Missing or empty profile must not crash."""

    def test_empty_profile(self):
        result = compute_consistency_score(ON_BRAND, {})
        assert set(result.keys()) == EXPECTED_KEYS

    def test_none_profile(self):
        result = compute_consistency_score(ON_BRAND, None)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_partial_profile(self):
        partial = {"mean_sentiment": 0.6}
        result = compute_consistency_score(ON_BRAND, partial)
        assert set(result.keys()) == EXPECTED_KEYS
        for val in result.values():
            assert 0.0 <= val <= 100.0

    def test_avg_keys_fallback(self):
        """Profile using only avg_* keys (no mean_*) must still work."""
        avg_profile = {
            "avg_sentiment": 0.55,
            "avg_formality": 0.78,
            "avg_readability_flesch": 45.0,
            "top_keywords": ["rolex", "watch"],
        }
        result = compute_consistency_score(ON_BRAND, avg_profile)
        assert set(result.keys()) == EXPECTED_KEYS
        for val in result.values():
            assert 0.0 <= val <= 100.0


class TestRelativeOrdering:
    """On-brand text should score higher than off-brand text."""

    def test_on_brand_beats_off_brand(self):
        on = compute_consistency_score(ON_BRAND, SAMPLE_PROFILE)
        off = compute_consistency_score(OFF_BRAND, SAMPLE_PROFILE)
        assert on["overall_score"] > off["overall_score"], (
            f"Expected on-brand ({on['overall_score']}) > "
            f"off-brand ({off['overall_score']})"
        )


class TestEditPlan:
    """generate_edit_plan must return a well-formed dict."""

    def test_returns_dict(self):
        result = generate_edit_plan(ON_BRAND, SAMPLE_PROFILE)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = generate_edit_plan(ON_BRAND, SAMPLE_PROFILE)
        for key in ("brand_id", "goals", "avoid_terms", "prefer_terms",
                     "style_rules", "tone_direction", "grounding_chunks"):
            assert key in result, f"Missing key: {key}"

    def test_empty_profile_no_crash(self):
        result = generate_edit_plan(ON_BRAND, {})
        assert isinstance(result, dict)
        assert "goals" in result
