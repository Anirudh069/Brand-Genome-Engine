"""
tests/test_feature_utils.py – Unit tests for src.feature_extraction.feature_utils.

Coverage targets:
  • empty string
  • None input
  • punctuation-heavy text
  • emoji-heavy text
  • very long text
  • normal prose
  • edge cases (control chars, mixed whitespace)

All functions must **never throw** and must return sane numeric defaults.
"""

from __future__ import annotations

import pytest

from src.feature_extraction.feature_utils import (
    avg_sentence_length,
    clean_text,
    punctuation_density,
    safe_truncate,
    sentence_split,
    vocab_diversity,
    word_tokenize,
)


# ─── Fixtures ────────────────────────────────────────────────────────────

EMPTY = ""
NONE = None
NORMAL = "The quick brown fox jumps over the lazy dog. It was a sunny day!"
PUNCT_HEAVY = "!!! ??? ... ,,, ;;; ::: ---"
EMOJI_HEAVY = "I love this! 😀🎉🔥💯 Amazing day 🌟✨"
CONTROL_CHARS = "Hello\x00\x01\x02 World\x7f\x80"
MULTI_WHITESPACE = "   lots   of    spaces\t\ttabs\n\nnewlines   "
VERY_LONG = "word " * 10_000  # 50 000 chars


# ═══════════════════════════════════════════════════════════════════════════
# clean_text
# ═══════════════════════════════════════════════════════════════════════════

class TestCleanText:
    """Tests for clean_text()."""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_empty_returns_empty(self):
        assert clean_text("") == ""

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_whitespace(self):
        assert clean_text(MULTI_WHITESPACE) == "lots of spaces tabs newlines"

    def test_removes_control_chars(self):
        result = clean_text(CONTROL_CHARS)
        assert "\x00" not in result
        assert "\x7f" not in result
        assert "Hello" in result
        assert "World" in result

    def test_handles_emoji(self):
        result = clean_text(EMOJI_HEAVY)
        # Emojis should survive cleaning
        assert "😀" in result
        assert "Amazing" in result

    def test_max_chars_truncation(self):
        result = clean_text(VERY_LONG, max_chars=100)
        assert len(result) <= 100

    def test_non_string_input(self):
        assert clean_text(12345) == "12345"
        assert clean_text(3.14) == "3.14"

    def test_never_raises(self):
        """Fuzz-like: various weird inputs must not throw."""
        for val in [None, "", 0, False, [], {}, 42, 3.14, object()]:
            result = clean_text(val)
            assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# safe_truncate
# ═══════════════════════════════════════════════════════════════════════════

class TestSafeTruncate:
    """Tests for safe_truncate()."""

    def test_none_returns_empty(self):
        assert safe_truncate(None) == ""

    def test_short_text_unchanged(self):
        assert safe_truncate("short") == "short"

    def test_truncates_at_word_boundary(self):
        text = "one two three four five"
        result = safe_truncate(text, max_chars=14)
        assert len(result) <= 14
        # Should not split a word
        assert not result.endswith("fo")

    def test_very_long_text(self):
        result = safe_truncate(VERY_LONG, max_chars=200)
        assert len(result) <= 200

    def test_exact_length_unchanged(self):
        text = "exact"
        assert safe_truncate(text, max_chars=5) == "exact"

    def test_deterministic(self):
        """Same input → same output, always."""
        for _ in range(10):
            assert safe_truncate(VERY_LONG, max_chars=100) == safe_truncate(
                VERY_LONG, max_chars=100
            )


# ═══════════════════════════════════════════════════════════════════════════
# sentence_split
# ═══════════════════════════════════════════════════════════════════════════

class TestSentenceSplit:
    """Tests for sentence_split()."""

    def test_none_returns_empty_list(self):
        assert sentence_split(None) == []

    def test_empty_returns_empty_list(self):
        assert sentence_split("") == []

    def test_single_sentence(self):
        result = sentence_split("Hello world")
        assert result == ["Hello world"]

    def test_multiple_sentences(self):
        result = sentence_split(NORMAL)
        assert len(result) == 2
        assert result[0].startswith("The quick")
        assert result[1].startswith("It was")

    def test_question_and_exclamation(self):
        result = sentence_split("Really? Yes! Okay.")
        assert len(result) == 3

    def test_no_trailing_empty_strings(self):
        result = sentence_split("One. Two. ")
        assert all(s.strip() for s in result)

    def test_emoji_text(self):
        result = sentence_split(EMOJI_HEAVY)
        # Should not crash; at least one segment
        assert len(result) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# word_tokenize
# ═══════════════════════════════════════════════════════════════════════════

class TestWordTokenize:
    """Tests for word_tokenize()."""

    def test_none_returns_empty_list(self):
        assert word_tokenize(None) == []

    def test_empty_returns_empty_list(self):
        assert word_tokenize("") == []

    def test_normal_text(self):
        tokens = word_tokenize("Hello, world!")
        assert "Hello" in tokens
        assert "world" in tokens
        assert "," in tokens

    def test_contractions(self):
        tokens = word_tokenize("I don't know")
        assert "don't" in tokens

    def test_emoji_text_does_not_crash(self):
        tokens = word_tokenize(EMOJI_HEAVY)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_punctuation_heavy(self):
        tokens = word_tokenize(PUNCT_HEAVY)
        assert isinstance(tokens, list)
        assert len(tokens) > 0


# ═══════════════════════════════════════════════════════════════════════════
# punctuation_density
# ═══════════════════════════════════════════════════════════════════════════

class TestPunctuationDensity:
    """Tests for punctuation_density()."""

    def test_none_returns_zero(self):
        assert punctuation_density(None) == 0.0

    def test_empty_returns_zero(self):
        assert punctuation_density("") == 0.0

    def test_no_punctuation(self):
        assert punctuation_density("hello world") == 0.0

    def test_all_punctuation(self):
        result = punctuation_density("!!!")
        assert result == 1.0

    def test_mixed(self):
        # "Hi!" → 1 punct char out of 3 total = 1/3
        result = punctuation_density("Hi!")
        assert abs(result - 1 / 3) < 1e-9

    def test_punctuation_heavy_text(self):
        result = punctuation_density(PUNCT_HEAVY)
        assert 0.0 < result <= 1.0

    def test_return_type(self):
        assert isinstance(punctuation_density("test."), float)

    def test_range(self):
        """Result must always be in [0, 1]."""
        for text in [EMPTY, NORMAL, PUNCT_HEAVY, EMOJI_HEAVY, VERY_LONG]:
            result = punctuation_density(text)
            assert 0.0 <= result <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# vocab_diversity
# ═══════════════════════════════════════════════════════════════════════════

class TestVocabDiversity:
    """Tests for vocab_diversity()."""

    def test_none_returns_zero(self):
        assert vocab_diversity(None) == 0.0

    def test_empty_list_returns_zero(self):
        assert vocab_diversity([]) == 0.0

    def test_all_unique(self):
        assert vocab_diversity(["a", "b", "c"]) == 1.0

    def test_all_same(self):
        assert vocab_diversity(["a", "a", "a"]) == pytest.approx(1 / 3)

    def test_mixed(self):
        result = vocab_diversity(["the", "cat", "the", "dog"])
        # 3 unique / 4 total = 0.75
        assert result == pytest.approx(0.75)

    def test_return_type(self):
        assert isinstance(vocab_diversity(["x"]), float)

    def test_range(self):
        """Result must always be in [0, 1]."""
        for tokens in [[], ["a"], ["a", "a"], ["a", "b"]]:
            result = vocab_diversity(tokens)
            assert 0.0 <= result <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# avg_sentence_length
# ═══════════════════════════════════════════════════════════════════════════

class TestAvgSentenceLength:
    """Tests for avg_sentence_length()."""

    def test_none_returns_zero(self):
        assert avg_sentence_length(None) == 0.0

    def test_empty_returns_zero(self):
        assert avg_sentence_length("") == 0.0

    def test_single_sentence(self):
        # "Hello world" → 1 sentence, 2 word tokens
        result = avg_sentence_length("Hello world")
        assert result == pytest.approx(2.0)

    def test_two_sentences(self):
        result = avg_sentence_length("Hi there. Bye now.")
        # Sentence 1: "Hi there." → 3 tokens (Hi, there, .)
        # Sentence 2: "Bye now." → 3 tokens (Bye, now, .)
        # avg = 3.0
        assert result == pytest.approx(3.0)

    def test_normal_text(self):
        result = avg_sentence_length(NORMAL)
        assert result > 0.0

    def test_return_type(self):
        assert isinstance(avg_sentence_length("Test."), float)

    def test_emoji_text_does_not_crash(self):
        result = avg_sentence_length(EMOJI_HEAVY)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_very_long_text(self):
        result = avg_sentence_length(VERY_LONG)
        assert isinstance(result, float)
        assert result >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: no function ever throws
# ═══════════════════════════════════════════════════════════════════════════

class TestNeverThrows:
    """Every public function must return without raising for any input."""

    INPUTS_TEXT = [None, "", " ", "a", PUNCT_HEAVY, EMOJI_HEAVY, VERY_LONG, CONTROL_CHARS]
    INPUTS_TOKENS = [None, [], [""], ["a", "b"], ["!"] * 1000]

    @pytest.mark.parametrize("text", INPUTS_TEXT)
    def test_clean_text(self, text):
        result = clean_text(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", INPUTS_TEXT)
    def test_safe_truncate(self, text):
        result = safe_truncate(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", INPUTS_TEXT)
    def test_sentence_split(self, text):
        result = sentence_split(text)
        assert isinstance(result, list)

    @pytest.mark.parametrize("text", INPUTS_TEXT)
    def test_word_tokenize(self, text):
        result = word_tokenize(text)
        assert isinstance(result, list)

    @pytest.mark.parametrize("text", INPUTS_TEXT)
    def test_punctuation_density(self, text):
        result = punctuation_density(text)
        assert isinstance(result, float)

    @pytest.mark.parametrize("tokens", INPUTS_TOKENS)
    def test_vocab_diversity(self, tokens):
        result = vocab_diversity(tokens)
        assert isinstance(result, float)

    @pytest.mark.parametrize("text", INPUTS_TEXT)
    def test_avg_sentence_length(self, text):
        result = avg_sentence_length(text)
        assert isinstance(result, float)
        assert result >= 0.0
