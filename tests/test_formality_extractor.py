# filepath: tests/test_formality_extractor.py
"""
Comprehensive tests for formality_extractor.extract_formality.

Contract: returns a float in [0.0, 1.0], deterministic, never raises.
Returns 0.5 for None/empty input.
"""

from __future__ import annotations

import pytest

from src.feature_extraction.formality_extractor import extract_formality


# ── Return type & range ───────────────────────────────────────────────────

class TestReturnType:
    """extract_formality always returns a float in [0.0, 1.0]."""

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_returns_float(self, text):
        result = extract_formality(text)
        assert isinstance(result, float)

    @pytest.mark.parametrize("text", [
        None, "", "   ",
        "hey dude what's up lol",
        "Furthermore, the aforementioned provisions shall apply.",
        "omg lol 😀🎉🔥 bruh this is lit!!!",
        "word " * 10_000,
        "a",
        "123 456 789",
    ])
    def test_in_valid_range(self, text):
        result = extract_formality(text)
        assert 0.0 <= result <= 1.0


# ── Empty / None defaults ────────────────────────────────────────────────

class TestEmptyNone:
    """None, empty, and whitespace-only returns 0.5 (midpoint)."""

    def test_none(self):
        assert extract_formality(None) == 0.5

    def test_empty_string(self):
        assert extract_formality("") == 0.5

    def test_whitespace_only(self):
        assert extract_formality("   ") == 0.5


# ── Ordering test: formal > casual > emoji/slang-heavy ───────────────────

class TestOrdering:
    """A formal paragraph > casual text > emoji/slang-heavy text."""

    FORMAL_TEXT = (
        "Furthermore, the aforementioned provisions shall apply to all "
        "subsequent amendments. Consequently, the comprehensive analysis "
        "demonstrates that the acquisition of distinguished assets "
        "constitutes a meticulous endeavour requiring considerable "
        "expertise and discernment."
    )
    CASUAL_TEXT = (
        "So I was thinking we could maybe meet up later and grab some "
        "food. What do you think? It's been a while since we hung out."
    )
    SLANG_EMOJI_TEXT = (
        "omg lol 😀🎉🔥💯 bruh this is so lit!!! gonna be awesome "
        "yeah dude haha wanna hang?? 🌟✨ tbh idk lmao"
    )

    def test_formal_greater_than_casual(self):
        formal = extract_formality(self.FORMAL_TEXT)
        casual = extract_formality(self.CASUAL_TEXT)
        assert formal > casual, f"formal={formal} should > casual={casual}"

    def test_casual_greater_than_slang_emoji(self):
        casual = extract_formality(self.CASUAL_TEXT)
        slang = extract_formality(self.SLANG_EMOJI_TEXT)
        assert casual > slang, f"casual={casual} should > slang={slang}"

    def test_formal_greater_than_slang_emoji(self):
        formal = extract_formality(self.FORMAL_TEXT)
        slang = extract_formality(self.SLANG_EMOJI_TEXT)
        assert formal > slang, f"formal={formal} should > slang={slang}"


# ── Informal text ────────────────────────────────────────────────────────

class TestInformal:
    """Informal text should score lower than formal text."""

    def test_casual_slang(self):
        score = extract_formality("hey dude lol this is so cool gonna be awesome yeah")
        assert score < 0.5

    def test_contractions(self):
        score = extract_formality("I can't believe it's not butter! Don't you think?")
        formal = extract_formality(
            "I cannot believe it is not butter. Do you not think so?"
        )
        assert score < formal

    def test_exclamations(self):
        casual = extract_formality("Wow!!! Amazing!!! So cool!!!")
        measured = extract_formality("The product demonstrates remarkable quality.")
        assert casual < measured

    def test_question_marks(self):
        """Excessive question marks should penalise."""
        qmarks = extract_formality("Really??? Are you serious??? Why???")
        measured = extract_formality("The methodology requires further analysis.")
        assert qmarks < measured

    def test_emoji_heavy(self):
        """Emoji-heavy text should score lower."""
        emoji = extract_formality("Great job! 😀🎉🔥💯🌟✨ Amazing! 🎊👏")
        plain = extract_formality("Great job. The performance was commendable.")
        assert emoji < plain

    def test_first_person_pronouns(self):
        """Heavy first/second-person pronoun usage is less formal."""
        personal = extract_formality(
            "I think you should do what I told you. My idea is yours."
        )
        impersonal = extract_formality(
            "The analysis suggests implementing the proposed methodology."
        )
        assert personal < impersonal


# ── Formal text ───────────────────────────────────────────────────────────

class TestFormal:
    """Formal text should score higher than informal."""

    def test_formal_markers(self):
        score = extract_formality(
            "Furthermore, the aforementioned provisions shall apply. "
            "Consequently, all subsequent amendments are hereby acknowledged."
        )
        informal = extract_formality("hey what's up lol")
        assert score > informal

    def test_long_sophisticated_sentences(self):
        score = extract_formality(
            "The comprehensive analysis demonstrates that the acquisition of "
            "distinguished timepieces constitutes a meticulous endeavour "
            "requiring considerable expertise and discernment."
        )
        assert score > 0.3  # should lean formal

    def test_formal_vs_casual_comparison(self):
        formal = extract_formality(
            "Nevertheless, the manufacturing process encompasses multiple "
            "stages of calibration and quality assurance."
        )
        casual = extract_formality("yeah it's kinda ok I guess lol")
        assert formal > casual


# ── Determinism ───────────────────────────────────────────────────────────

class TestDeterministic:
    """Same input must always produce the same output."""

    def test_deterministic_simple(self):
        text = "Hello, how are you doing today?"
        assert extract_formality(text) == extract_formality(text)

    def test_deterministic_formal(self):
        text = (
            "The comprehensive analysis demonstrates that the acquisition of "
            "distinguished timepieces constitutes a meticulous endeavour."
        )
        a = extract_formality(text)
        b = extract_formality(text)
        assert a == b

    def test_deterministic_informal(self):
        text = "lol dude that's so cool gonna be lit bruh"
        assert extract_formality(text) == extract_formality(text)

    def test_deterministic_emoji(self):
        text = "omg 😀🎉🔥 so awesome!!! 💯✨"
        assert extract_formality(text) == extract_formality(text)

    def test_deterministic_across_calls(self):
        """Ten consecutive calls yield the same value."""
        text = "The committee has resolved to implement the proposal."
        results = [extract_formality(text) for _ in range(10)]
        assert len(set(results)) == 1


# ── Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary / degenerate inputs."""

    def test_single_word(self):
        result = extract_formality("hello")
        assert 0.0 <= result <= 1.0

    def test_punctuation_only(self):
        result = extract_formality("!!! ??? ...")
        assert 0.0 <= result <= 1.0

    def test_numbers_only(self):
        result = extract_formality("123 456 789")
        assert 0.0 <= result <= 1.0

    def test_single_emoji(self):
        result = extract_formality("😀")
        assert 0.0 <= result <= 1.0

    def test_very_long_text(self):
        result = extract_formality("word " * 10_000)
        assert 0.0 <= result <= 1.0


# ── Never throws ─────────────────────────────────────────────────────────

class TestNeverThrows:
    """extract_formality never raises, regardless of input."""

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
        result = extract_formality(text)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
