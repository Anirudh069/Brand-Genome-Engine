# filepath: tests/test_topic_extractor.py
"""
Comprehensive tests for topic_extractor.extract_topics.

Contract: returns (list[str], list[float]) of equal length,
          weights sum ≈ 1.0 for meaningful text, never raises, deterministic.
"""

from __future__ import annotations

import pytest

from src.feature_extraction.topic_extractor import extract_topics


# ── Return shape ──────────────────────────────────────────────────────────

class TestReturnShape:
    """extract_topics returns two lists of equal length."""

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_returns_tuple_of_two_lists(self, text):
        topics, weights = extract_topics(text)
        assert isinstance(topics, list)
        assert isinstance(weights, list)

    @pytest.mark.parametrize("num_topics", [1, 3, 5, 8])
    def test_length_equals_num_topics(self, num_topics):
        topics, weights = extract_topics("luxury craftsmanship", num_topics=num_topics)
        assert len(topics) == num_topics
        assert len(weights) == num_topics

    def test_default_num_topics_is_5(self):
        topics, weights = extract_topics("hello world")
        assert len(topics) == 5
        assert len(weights) == 5

    def test_topics_are_strings(self):
        topics, _ = extract_topics("luxury watch heritage")
        assert all(isinstance(t, str) for t in topics)

    def test_weights_are_floats(self):
        _, weights = extract_topics("luxury watch heritage")
        assert all(isinstance(w, float) for w in weights)

    def test_n_topics_alias(self):
        """n_topics keyword works the same as num_topics."""
        topics, weights = extract_topics("luxury craftsmanship", n_topics=3)
        assert len(topics) == 3
        assert len(weights) == 3


# ── Empty / None defaults ────────────────────────────────────────────────

class TestEmptyNone:
    """None/empty returns (["unknown"]*n, [0.0]*n)."""

    def test_none(self):
        topics, weights = extract_topics(None)
        assert topics == ["unknown"] * 5
        assert weights == [0.0] * 5

    def test_empty_string(self):
        topics, weights = extract_topics("")
        assert topics == ["unknown"] * 5
        assert weights == [0.0] * 5

    def test_whitespace_only(self):
        topics, weights = extract_topics("   ")
        assert topics == ["unknown"] * 5
        assert weights == [0.0] * 5


# ── Weight constraints ───────────────────────────────────────────────────

class TestWeightConstraints:
    """Weights must be non-negative and sum ≈ 1.0 for meaningful text."""

    def test_weights_non_negative(self):
        _, weights = extract_topics(
            "luxury craftsmanship heritage innovation design"
        )
        assert all(w >= 0.0 for w in weights)

    def test_weights_sum_approximately_one(self):
        _, weights = extract_topics(
            "luxury craftsmanship heritage innovation design "
            "precision quality premium exclusive"
        )
        total = sum(weights)
        # Should sum to ~1.0 when there's meaningful content
        assert abs(total - 1.0) < 1e-9

    def test_weights_sum_at_most_one(self):
        _, weights = extract_topics(
            "luxury craftsmanship heritage innovation design "
            "precision quality premium exclusive"
        )
        assert sum(weights) <= 1.0 + 1e-9  # allow float rounding

    def test_nonzero_weights_for_gibberish_text(self):
        """Gibberish words are valid TF keywords – weights should be nonzero."""
        topics, weights = extract_topics("xyzzy plugh foobar")
        # TF layer picks up any alphabetic non-stopword tokens
        assert sum(weights) > 0.0
        assert all(w >= 0.0 for w in weights)

    def test_zero_weights_for_stopword_only_text(self):
        """Text made entirely of stopwords produces all-unknown / zero."""
        _, weights = extract_topics("the and is was are were")
        assert all(w == 0.0 for w in weights)

    def test_weights_sum_one_for_single_domain_word(self):
        """Even one domain keyword should produce weights summing to ~1."""
        _, weights = extract_topics("luxury")
        total = sum(weights)
        assert abs(total - 1.0) < 1e-9 or total == 0.0


# ── Topic identification ─────────────────────────────────────────────────

class TestTopicIdentification:
    """Domain-specific text should surface the expected topic."""

    def test_craftsmanship_detected(self):
        topics, weights = extract_topics(
            "The master watchmaker carefully assembled the tourbillon "
            "movement in the atelier with meticulous finishing."
        )
        assert "craftsmanship" in topics
        idx = topics.index("craftsmanship")
        assert weights[idx] > 0.0

    def test_luxury_detected(self):
        topics, weights = extract_topics(
            "This exclusive luxury timepiece features a gold case "
            "adorned with diamonds and precious gems."
        )
        assert "luxury" in topics
        idx = topics.index("luxury")
        assert weights[idx] > 0.0

    def test_heritage_detected(self):
        topics, weights = extract_topics(
            "Founded in 1845, the brand has a rich heritage spanning "
            "generations of traditional watchmaking since its origin."
        )
        assert "heritage" in topics
        idx = topics.index("heritage")
        assert weights[idx] > 0.0

    def test_innovation_detected(self):
        topics, weights = extract_topics(
            "Pioneering breakthrough technology with patented ceramic "
            "and titanium materials for advanced performance."
        )
        assert "innovation" in topics
        idx = topics.index("innovation")
        assert weights[idx] > 0.0

    def test_top_topic_is_dominant(self):
        """When text is heavily themed, that theme should rank first."""
        topics, weights = extract_topics(
            "luxury luxurious prestige prestigious exclusive elite "
            "premium opulent refined exquisite gold diamond",
            num_topics=3,
        )
        assert topics[0] == "luxury"
        assert weights[0] >= weights[1]


# ── TF keyword extraction (non-domain text) ──────────────────────────────

class TestTFKeywordExtraction:
    """General text (outside brand domain) should still produce topics."""

    def test_repeated_word_appears_as_topic(self):
        """A word repeated many times should show up in the topic list."""
        topics, weights = extract_topics(
            "python python python python python code code code"
        )
        assert "python" in topics

    def test_repeated_word_has_positive_weight(self):
        topics, weights = extract_topics(
            "python python python python python code code code"
        )
        idx = topics.index("python")
        assert weights[idx] > 0.0

    def test_longer_words_boosted(self):
        """Longer non-stopword tokens should rank higher than short ones."""
        topics, weights = extract_topics(
            "architecture architecture xx xx xx xx xx xx",
            num_topics=3,
        )
        # 'architecture' has length boost so should appear
        assert "architecture" in topics

    def test_stopwords_excluded(self):
        """Common stopwords should NOT appear as topics."""
        topics, _ = extract_topics(
            "the the the and and and is is is was was was"
        )
        for stop in ("the", "and", "is", "was"):
            assert stop not in topics

    def test_non_domain_text_still_produces_topics(self):
        topics, weights = extract_topics(
            "database migration framework deployment configuration"
        )
        total = sum(weights)
        # Should have meaningful weights even for non-domain text
        assert total > 0.0


# ── Deterministic ─────────────────────────────────────────────────────────

class TestDeterministic:
    """extract_topics must return identical results for identical input."""

    def test_deterministic_domain_text(self):
        text = "luxury craftsmanship heritage innovation design"
        r1 = extract_topics(text)
        r2 = extract_topics(text)
        assert r1 == r2

    def test_deterministic_general_text(self):
        text = "python programming language development framework"
        r1 = extract_topics(text)
        r2 = extract_topics(text)
        assert r1 == r2

    def test_deterministic_empty(self):
        r1 = extract_topics(None)
        r2 = extract_topics(None)
        assert r1 == r2

    def test_deterministic_across_num_topics(self):
        text = "luxury craftsmanship heritage"
        t3a, w3a = extract_topics(text, num_topics=3)
        t3b, w3b = extract_topics(text, num_topics=3)
        assert t3a == t3b
        assert w3a == w3b


# ── num_topics edge cases ────────────────────────────────────────────────

class TestNumTopicsEdge:

    def test_num_topics_one(self):
        topics, weights = extract_topics("luxury watch", num_topics=1)
        assert len(topics) == 1
        assert len(weights) == 1

    def test_num_topics_larger_than_categories(self):
        """If num_topics > number of categories, pad with 'unknown'."""
        topics, weights = extract_topics("luxury watch", num_topics=20)
        assert len(topics) == 20
        assert len(weights) == 20
        # Extra slots should be "unknown" with weight 0.0
        assert "unknown" in topics

    def test_num_topics_zero_becomes_one(self):
        topics, weights = extract_topics("luxury", num_topics=0)
        assert len(topics) >= 1


# ── Never throws ─────────────────────────────────────────────────────────

class TestNeverThrows:
    """extract_topics never raises, regardless of input."""

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
        topics, weights = extract_topics(text)
        assert isinstance(topics, list)
        assert isinstance(weights, list)
        assert len(topics) == len(weights)
