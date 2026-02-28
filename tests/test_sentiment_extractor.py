# filepath: tests/test_sentiment_extractor.py
"""
Comprehensive tests for sentiment_extractor.extract_sentiment.

Contract: returns a float in [0.0, 1.0], never raises.
"""

from __future__ import annotations

import pytest

from src.feature_extraction.sentiment_extractor import extract_sentiment


# ── Return type & range ───────────────────────────────────────────────────

class TestReturnType:
    """extract_sentiment always returns a float in [0.0, 1.0]."""

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_returns_float(self, text):
        result = extract_sentiment(text)
        assert isinstance(result, float)

    @pytest.mark.parametrize("text", [
        None, "", "   ",
        "hello world",
        "This is absolutely terrible and horrible.",
        "This is amazing and wonderful!",
    ])
    def test_in_valid_range(self, text):
        result = extract_sentiment(text)
        assert 0.0 <= result <= 1.0


# ── Empty / None defaults ────────────────────────────────────────────────

class TestEmptyNone:
    """None, empty, and whitespace-only input returns 0.5 (neutral)."""

    def test_none(self):
        assert extract_sentiment(None) == 0.5

    def test_empty_string(self):
        assert extract_sentiment("") == 0.5

    def test_whitespace_only(self):
        assert extract_sentiment("   ") == 0.5


# ── Positive sentiment ───────────────────────────────────────────────────

class TestPositiveSentiment:
    """Positive words should push the score above 0.5."""

    def test_single_positive_word(self):
        assert extract_sentiment("excellent") > 0.5

    def test_multiple_positive_words(self):
        score = extract_sentiment(
            "This is an amazing, wonderful, and fantastic experience."
        )
        assert score > 0.5

    def test_brand_luxury_copy(self):
        score = extract_sentiment(
            "Exquisite craftsmanship meets timeless elegance in this "
            "prestigious masterpiece of luxury watchmaking."
        )
        assert score > 0.5

    def test_more_positive_words_higher_score(self):
        short = extract_sentiment("good")
        long = extract_sentiment(
            "great excellent amazing wonderful fantastic superb"
        )
        assert long > short


# ── Negative sentiment ───────────────────────────────────────────────────

class TestNegativeSentiment:
    """Negative words should push the score below 0.5."""

    def test_single_negative_word(self):
        assert extract_sentiment("terrible") < 0.5

    def test_multiple_negative_words(self):
        score = extract_sentiment(
            "This is a horrible, awful, and disappointing failure."
        )
        assert score < 0.5

    def test_brand_criticism(self):
        score = extract_sentiment(
            "Overpriced, cheap-looking, and unreliable. A total disappointment."
        )
        assert score < 0.5


# ── Negation handling ────────────────────────────────────────────────────

class TestNegation:
    """Negation words flip the polarity of the next sentiment word."""

    def test_not_good_is_negative(self):
        score = extract_sentiment("not good")
        assert score < 0.5

    def test_not_bad_is_positive(self):
        score = extract_sentiment("not bad")
        assert score > 0.5

    def test_never_disappointing(self):
        # "never" + "disappointing" → positive
        score = extract_sentiment("never disappointing")
        assert score > 0.5


# ── Intensifier handling ─────────────────────────────────────────────────

class TestIntensifiers:
    """Intensifiers boost the magnitude of the next sentiment word."""

    def test_very_good_stronger_than_good(self):
        base = extract_sentiment("good")
        intensified = extract_sentiment("very good")
        assert intensified > base

    def test_extremely_bad_stronger_than_bad(self):
        base = extract_sentiment("bad")
        intensified = extract_sentiment("extremely bad")
        assert intensified < base  # more negative


# ── Neutral text ──────────────────────────────────────────────────────────

class TestNeutral:
    """Text with no sentiment words should be close to 0.5."""

    def test_factual_text(self):
        score = extract_sentiment("The watch has a 42mm case and a date window.")
        assert abs(score - 0.5) < 0.15  # roughly neutral

    def test_numbers_only(self):
        score = extract_sentiment("123 456 789")
        assert score == 0.5


# ── Deterministic ─────────────────────────────────────────────────────────

class TestDeterministic:
    """extract_sentiment returns the same result for the same input."""

    def test_deterministic_positive(self):
        text = "This is wonderful and amazing!"
        assert extract_sentiment(text) == extract_sentiment(text)

    def test_deterministic_negative(self):
        text = "Terrible and horrible experience."
        assert extract_sentiment(text) == extract_sentiment(text)

    def test_deterministic_neutral(self):
        text = "The meeting is at 3pm."
        assert extract_sentiment(text) == extract_sentiment(text)


# ── Never throws ─────────────────────────────────────────────────────────

class TestNeverThrows:
    """extract_sentiment never raises, regardless of input."""

    @pytest.mark.parametrize("text", [
        None,
        "",
        "   ",
        42,
        ["a", "list"],
        "\x00\x01\x02\x7f",
        "I love this! 😀🎉🔥💯 Amazing day 🌟✨",
        "word " * 10_000,
    ])
    def test_never_raises(self, text):
        result = extract_sentiment(text)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
