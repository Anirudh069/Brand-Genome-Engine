"""
Tests for src.feature_extraction.vocabulary_extractor.extract_vocab_metrics.
"""

from __future__ import annotations

import pytest

from src.feature_extraction.vocabulary_extractor import extract_vocab_metrics

# ── Required keys (match the ML contract) ─────────────────────────────────
EXPECTED_KEYS = {"vocab_diversity", "avg_sentence_length", "punctuation_density"}


# ── Helpers ────────────────────────────────────────────────────────────────

class TestReturnShape:
    """Every call must return a dict with exactly the contract keys."""

    @pytest.mark.parametrize("text", [None, "", " ", "hello world"])
    def test_keys_present(self, text):
        result = extract_vocab_metrics(text)
        assert isinstance(result, dict)
        assert set(result.keys()) == EXPECTED_KEYS

    @pytest.mark.parametrize("text", [None, "", " ", "hello world"])
    def test_values_are_floats(self, text):
        result = extract_vocab_metrics(text)
        for key in EXPECTED_KEYS:
            assert isinstance(result[key], float), f"{key} is not float"


# ── None / empty input ────────────────────────────────────────────────────

class TestEmptyNone:
    """None and empty strings must produce sane zero defaults."""

    def test_none_returns_defaults(self):
        result = extract_vocab_metrics(None)
        assert result["vocab_diversity"] == 0.0
        assert result["avg_sentence_length"] == 0.0
        assert result["punctuation_density"] == 0.0

    def test_empty_string_returns_defaults(self):
        result = extract_vocab_metrics("")
        assert result["vocab_diversity"] == 0.0
        assert result["avg_sentence_length"] == 0.0
        assert result["punctuation_density"] == 0.0

    def test_whitespace_only_returns_defaults(self):
        result = extract_vocab_metrics("    ")
        assert result["vocab_diversity"] == 0.0
        assert result["avg_sentence_length"] == 0.0
        assert result["punctuation_density"] == 0.0


# ── avg_sentence_length differs for short vs long ─────────────────────────

class TestAvgSentenceLength:
    """Short informal vs. longer formal paragraph should differ."""

    SHORT = "Cool watch!"
    LONG = (
        "The intricate mechanics of this Swiss-made timepiece exemplify "
        "centuries of horological innovation. Each component is crafted "
        "with meticulous precision. The sapphire crystal protects a "
        "beautifully finished dial."
    )

    def test_short_sentence(self):
        result = extract_vocab_metrics(self.SHORT)
        assert result["avg_sentence_length"] > 0.0

    def test_long_paragraph(self):
        result = extract_vocab_metrics(self.LONG)
        assert result["avg_sentence_length"] > 0.0

    def test_long_has_higher_avg(self):
        short_avg = extract_vocab_metrics(self.SHORT)["avg_sentence_length"]
        long_avg = extract_vocab_metrics(self.LONG)["avg_sentence_length"]
        assert long_avg > short_avg


# ── Punctuation-heavy string ──────────────────────────────────────────────

class TestPunctuationDensity:
    """Punctuation-heavy text should yield higher density."""

    def test_no_punctuation(self):
        result = extract_vocab_metrics("hello world")
        assert result["punctuation_density"] == 0.0

    def test_heavy_punctuation(self):
        result = extract_vocab_metrics("!!! ??? ... ,,, ---")
        assert result["punctuation_density"] > 0.5

    def test_higher_than_prose(self):
        heavy = extract_vocab_metrics("!!! ??? ...")["punctuation_density"]
        prose = extract_vocab_metrics("Hello world this is text")["punctuation_density"]
        assert heavy > prose


# ── vocab_diversity in [0, 1] ─────────────────────────────────────────────

class TestVocabDiversity:
    """TTR must always be in the closed interval [0, 1]."""

    @pytest.mark.parametrize(
        "text",
        [
            "one two three four five",                # all unique → 1.0
            "the the the the the",                    # all same → low
            "Hello, this is a more diverse sentence.", # mixed
            None,
            "",
        ],
    )
    def test_in_unit_interval(self, text):
        vd = extract_vocab_metrics(text)["vocab_diversity"]
        assert 0.0 <= vd <= 1.0

    def test_all_unique_tokens(self):
        result = extract_vocab_metrics("alpha bravo charlie delta echo")
        assert result["vocab_diversity"] == 1.0

    def test_all_same_tokens(self):
        result = extract_vocab_metrics("the the the the the")
        assert result["vocab_diversity"] < 0.5


# ── Never-throws guarantee ────────────────────────────────────────────────

class TestNeverThrows:
    """extract_vocab_metrics must not raise on any input."""

    @pytest.mark.parametrize(
        "text",
        [
            None,
            "",
            "   ",
            42,
            ["not", "a", "string"],
            "Hello\x00\x01\x02 World\x7f",
            "I love this! 😀🎉🔥💯 Amazing day 🌟✨",
            "word " * 10_000,
        ],
    )
    def test_never_raises(self, text):
        # Should return a dict with zero-defaults at worst, never raise.
        result = extract_vocab_metrics(text)
        assert isinstance(result, dict)
        assert set(result.keys()) == EXPECTED_KEYS
