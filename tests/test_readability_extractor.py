# filepath: tests/test_readability_extractor.py
"""
Comprehensive tests for readability_extractor.extract_readability.

Contract: returns (flesch_reading_ease: float, avg_sentence_length: float),
          never raises.
"""

from __future__ import annotations

import pytest

from src.feature_extraction.readability_extractor import (
    extract_readability,
    flesch_reading_ease,
    _count_syllables,
)


# ── Return shape ──────────────────────────────────────────────────────────

class TestReturnShape:
    """extract_readability returns a 2-tuple of floats."""

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_returns_tuple_of_two(self, text):
        result = extract_readability(text)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_both_are_floats(self, text):
        fre, asl = extract_readability(text)
        assert isinstance(fre, float)
        assert isinstance(asl, float)


# ── Empty / None defaults ────────────────────────────────────────────────

class TestEmptyNone:
    """None/empty/whitespace returns (0.0, 0.0)."""

    def test_none(self):
        assert extract_readability(None) == (0.0, 0.0)

    def test_empty_string(self):
        assert extract_readability("") == (0.0, 0.0)

    def test_whitespace_only(self):
        assert extract_readability("   ") == (0.0, 0.0)


# ── Flesch Reading Ease range ────────────────────────────────────────────

class TestFleschRange:
    """FRE should be in [0.0, 121.22] and follow expected patterns."""

    def test_simple_text_high_score(self):
        fre, _ = extract_readability("The cat sat on the mat. It was fun.")
        assert fre > 50.0  # very simple → high FRE

    def test_complex_text_lower_score(self):
        fre, _ = extract_readability(
            "The implementation of comprehensive methodological frameworks "
            "necessitates considerable deliberation regarding the fundamental "
            "epistemological underpinnings of interdisciplinary collaboration."
        )
        # Complex long words → lower FRE
        assert fre < 50.0

    def test_simple_vs_complex(self):
        simple_fre, _ = extract_readability(
            "I like cats. Cats are nice. They are soft."
        )
        complex_fre, _ = extract_readability(
            "Notwithstanding the aforementioned complications, "
            "the implementation necessitates comprehensive deliberation."
        )
        assert simple_fre > complex_fre

    def test_fre_non_negative(self):
        fre, _ = extract_readability("Hello world.")
        assert fre >= 0.0

    def test_fre_capped(self):
        fre, _ = extract_readability("Go. Do. Be. At. Up.")
        assert fre <= 121.22


# ── Average sentence length ──────────────────────────────────────────────

class TestAvgSentenceLength:
    """ASL should reflect words per sentence."""

    def test_short_sentences(self):
        _, asl = extract_readability("Hi. Hello. Yes.")
        assert asl < 5.0

    def test_long_sentences(self):
        _, asl = extract_readability(
            "The quick brown fox jumped over the lazy dog and then "
            "continued running through the forest at full speed."
        )
        assert asl > 5.0

    def test_asl_non_negative(self):
        _, asl = extract_readability("Hello world.")
        assert asl >= 0.0


# ── Syllable counting (internal) ─────────────────────────────────────────

class TestSyllableCounting:
    """_count_syllables heuristic should be reasonable."""

    @pytest.mark.parametrize("word,expected_min,expected_max", [
        ("the", 1, 1),
        ("cat", 1, 1),
        ("hello", 2, 2),
        ("beautiful", 3, 4),
        ("extraordinary", 5, 6),
        ("a", 1, 1),
        ("I", 1, 1),
    ])
    def test_syllable_range(self, word, expected_min, expected_max):
        count = _count_syllables(word)
        assert expected_min <= count <= expected_max

    def test_empty_word(self):
        assert _count_syllables("") == 0

    def test_always_at_least_one(self):
        """Non-empty words have ≥ 1 syllable."""
        assert _count_syllables("x") >= 1


# ── Never throws ─────────────────────────────────────────────────────────

class TestNeverThrows:
    """extract_readability never raises, regardless of input."""

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
        result = extract_readability(text)
        assert isinstance(result, tuple)
        assert len(result) == 2
        fre, asl = result
        assert isinstance(fre, float)
        assert isinstance(asl, float)
        assert fre >= 0.0
        assert asl >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests for flesch_reading_ease() – the public convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════


class TestFleschReadingEaseEmptyNone:
    """flesch_reading_ease returns 0.0 for None / empty / whitespace."""

    def test_none_returns_zero(self):
        assert flesch_reading_ease(None) == 0.0

    def test_empty_string_returns_zero(self):
        assert flesch_reading_ease("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert flesch_reading_ease("   ") == 0.0


class TestFleschReadingEaseType:
    """flesch_reading_ease always returns a float."""

    @pytest.mark.parametrize("text", [
        None, "", "Hello world.", "The cat sat on the mat.",
    ])
    def test_returns_float(self, text):
        result = flesch_reading_ease(text)
        assert isinstance(result, float)


class TestFleschReadingEaseSanity:
    """Simple text should score higher than complex text."""

    def test_simple_higher_than_complex(self):
        simple = flesch_reading_ease(
            "I like cats. Cats are nice. They are soft."
        )
        complex_ = flesch_reading_ease(
            "Notwithstanding the aforementioned complications, "
            "the implementation necessitates comprehensive deliberation."
        )
        assert simple > complex_

    def test_score_non_negative(self):
        assert flesch_reading_ease("Go. Do. Be.") >= 0.0

    def test_score_bounded(self):
        assert flesch_reading_ease("Go. Do. Be.") <= 121.22


class TestFleschReadingEaseDeterministic:
    """Calling with the same input always yields the same result."""

    def test_deterministic(self):
        text = "The quick brown fox jumped over the lazy dog."
        first = flesch_reading_ease(text)
        second = flesch_reading_ease(text)
        assert first == second

    def test_deterministic_complex(self):
        text = (
            "The implementation of comprehensive methodological frameworks "
            "necessitates considerable deliberation regarding the fundamental "
            "epistemological underpinnings."
        )
        assert flesch_reading_ease(text) == flesch_reading_ease(text)


class TestFleschReadingEaseConsistency:
    """flesch_reading_ease must match the first element of extract_readability."""

    @pytest.mark.parametrize("text", [
        None, "", "hello", "The cat sat on the mat. It was fun.",
        "Complex epistemological deliberation.",
    ])
    def test_matches_extract_readability(self, text):
        fre_standalone = flesch_reading_ease(text)
        fre_tuple, _ = extract_readability(text)
        assert fre_standalone == fre_tuple


class TestFleschReadingEaseNeverThrows:
    """flesch_reading_ease never raises, regardless of input."""

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
        result = flesch_reading_ease(text)
        assert isinstance(result, float)
        assert result >= 0.0
