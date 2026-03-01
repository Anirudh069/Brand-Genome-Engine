# filepath: tests/test_embedding_extractor.py
"""
Comprehensive tests for embedding_extractor.

Contract: returns list[float] of length 384, never raises.

Real-model tests (require model download) only run when the environment
variable ``RUN_EMBEDDING_TESTS=1`` is set.

Note: On CPython 3.9 + macOS (LibreSSL 2.8), importing both faiss and
sentence-transformers in the same process can segfault.  This entire module
is marked ``requires_model`` and is skipped during combined runs unless
``--include-model-tests`` is passed.  Run in isolation:
``python -m pytest tests/test_embedding_extractor.py -v``
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.requires_model

from src.feature_extraction.embedding_extractor import (
    extract_embedding,
    get_embedding,
    _hash_fallback,
    _EMBEDDING_DIM,
    _DEFAULT_MODEL_NAME,
)


# ── Return shape (extract_embedding) ─────────────────────────────────────

class TestReturnShape:
    """extract_embedding always returns a list of 384 floats."""

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_returns_list(self, text):
        result = extract_embedding(text)
        assert isinstance(result, list)

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_correct_length(self, text):
        result = extract_embedding(text)
        assert len(result) == _EMBEDDING_DIM

    @pytest.mark.parametrize("text", ["hello world", "luxury watches"])
    def test_all_elements_are_floats(self, text):
        result = extract_embedding(text)
        assert all(isinstance(v, float) for v in result)


# ── get_embedding contract ────────────────────────────────────────────────

class TestGetEmbeddingContract:
    """get_embedding returns (list[float], str) with correct shapes."""

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_returns_tuple(self, text):
        result = get_embedding(text)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_embedding_is_list_of_correct_length(self, text):
        embedding, _ = get_embedding(text)
        assert isinstance(embedding, list)
        assert len(embedding) == _EMBEDDING_DIM

    @pytest.mark.parametrize("text", [None, "", "   ", "hello world"])
    def test_model_name_is_string(self, text):
        _, model_name = get_embedding(text)
        assert isinstance(model_name, str)
        assert model_name == _DEFAULT_MODEL_NAME

    @pytest.mark.parametrize("text", ["hello world", "luxury watches"])
    def test_all_elements_are_floats(self, text):
        embedding, _ = get_embedding(text)
        assert all(isinstance(v, float) for v in embedding)

    def test_custom_model_name_returned(self):
        _, model_name = get_embedding("test", model_name="custom-model")
        assert model_name == "custom-model"


# ── Empty / None defaults ────────────────────────────────────────────────

class TestEmptyNone:
    """None/empty/whitespace returns zero vector."""

    def test_none_is_zero_vector(self):
        result = extract_embedding(None)
        assert result == [0.0] * _EMBEDDING_DIM

    def test_empty_string_is_zero_vector(self):
        result = extract_embedding("")
        assert result == [0.0] * _EMBEDDING_DIM

    def test_whitespace_is_zero_vector(self):
        result = extract_embedding("   ")
        assert result == [0.0] * _EMBEDDING_DIM

    def test_get_embedding_none_is_zero_vector(self):
        embedding, model_name = get_embedding(None)
        assert embedding == [0.0] * _EMBEDDING_DIM
        assert model_name == _DEFAULT_MODEL_NAME

    def test_get_embedding_empty_is_zero_vector(self):
        embedding, _ = get_embedding("")
        assert embedding == [0.0] * _EMBEDDING_DIM


# ── Non-empty text produces non-zero vector ──────────────────────────────

class TestNonZero:
    """Non-empty text should NOT produce an all-zero vector (hash fallback)."""

    def test_non_empty_not_all_zero(self):
        result = extract_embedding("hello world")
        assert any(v != 0.0 for v in result)

    def test_different_texts_different_vectors(self):
        v1 = extract_embedding("luxury watches")
        v2 = extract_embedding("cheap plastic toys")
        assert v1 != v2


# ── Hash fallback (internal) ─────────────────────────────────────────────

class TestHashFallback:
    """_hash_fallback produces deterministic, normalised vectors."""

    def test_correct_length(self):
        result = _hash_fallback("test", dim=384)
        assert len(result) == 384

    def test_deterministic(self):
        v1 = _hash_fallback("hello")
        v2 = _hash_fallback("hello")
        assert v1 == v2

    def test_different_inputs_different_outputs(self):
        v1 = _hash_fallback("hello")
        v2 = _hash_fallback("world")
        assert v1 != v2

    def test_unit_normalised(self):
        vec = _hash_fallback("test text")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_values_in_range(self):
        vec = _hash_fallback("test")
        # After L2-normalisation, values should be in [-1, 1]
        assert all(-1.0 <= v <= 1.0 for v in vec)

    def test_custom_dim(self):
        vec = _hash_fallback("test", dim=128)
        assert len(vec) == 128


# ── Embedding dimension constant ─────────────────────────────────────────

class TestDimensionContract:
    """The embedding dimension contract is 384."""

    def test_embedding_dim_is_384(self):
        assert _EMBEDDING_DIM == 384


# ── Never throws ─────────────────────────────────────────────────────────

class TestNeverThrows:
    """extract_embedding and get_embedding never raise, regardless of input."""

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
    def test_extract_embedding_never_raises(self, text):
        result = extract_embedding(text)
        assert isinstance(result, list)
        assert len(result) == _EMBEDDING_DIM

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
    def test_get_embedding_never_raises(self, text):
        result = get_embedding(text)
        assert isinstance(result, tuple)
        embedding, model_name = result
        assert isinstance(embedding, list)
        assert len(embedding) == _EMBEDDING_DIM
        assert isinstance(model_name, str)


# ── Real model tests (opt-in) ────────────────────────────────────────────

_SKIP_REAL = os.environ.get("RUN_EMBEDDING_TESTS", "0") != "1"


@pytest.mark.skipif(_SKIP_REAL, reason="Set RUN_EMBEDDING_TESTS=1 to run real model tests")
class TestRealModel:
    """
    Integration tests that download and run the real all-MiniLM-L6-v2 model.

    These are **opt-in**: set ``RUN_EMBEDDING_TESTS=1`` before running pytest.

    .. code-block:: bash

        RUN_EMBEDDING_TESTS=1 pytest tests/test_embedding_extractor.py -k TestRealModel -v
    """

    def test_real_embedding_shape(self):
        embedding, model_name = get_embedding("Luxury watches are timeless.")
        assert len(embedding) == _EMBEDDING_DIM
        assert model_name == _DEFAULT_MODEL_NAME

    def test_real_embedding_non_zero(self):
        embedding, _ = get_embedding("Hello world")
        assert any(v != 0.0 for v in embedding)

    def test_real_embedding_all_floats(self):
        embedding, _ = get_embedding("Testing float types")
        assert all(isinstance(v, float) for v in embedding)

    def test_real_embedding_deterministic(self):
        e1, _ = get_embedding("Determinism check")
        e2, _ = get_embedding("Determinism check")
        assert e1 == e2

    def test_real_embedding_different_texts(self):
        e1, _ = get_embedding("luxury watches")
        e2, _ = get_embedding("cheap plastic toys")
        assert e1 != e2

    def test_real_embedding_empty_uses_placeholder(self):
        embedding, model_name = get_embedding(None)
        # With real model, None still gets a result (placeholder " ")
        assert len(embedding) == _EMBEDDING_DIM
        assert model_name == _DEFAULT_MODEL_NAME

    def test_real_extract_embedding_compat(self):
        """extract_embedding returns same vector as get_embedding."""
        vec = extract_embedding("compatibility test")
        embedding, _ = get_embedding("compatibility test")
        assert vec == embedding

    def test_real_embedding_bounded(self):
        """Real embeddings should have values roughly in [-1, 1]."""
        embedding, _ = get_embedding("A normal sentence for testing.")
        assert all(-5.0 <= v <= 5.0 for v in embedding)
