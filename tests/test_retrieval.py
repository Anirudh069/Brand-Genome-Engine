# filepath: tests/test_retrieval.py
"""
Comprehensive tests for src.benchmarking.retrieval.

Tests run against whichever backend is available (faiss-cpu preferred,
sklearn fallback).  A second test class forces the sklearn path even
when faiss is installed.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.benchmarking.retrieval import (
    _FAISS_AVAILABLE,
    _IndexWrapper,
    backend_name,
    build_index,
    load_index,
    query,
    save_index,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_embeddings(n: int = 10, dim: int = 8, seed: int = 0) -> list[list[float]]:
    """Deterministic synthetic embeddings for testing."""
    rng = np.random.RandomState(seed)
    mat = rng.randn(n, dim).astype(np.float32)
    # L2-normalise so cosine distance is well-defined
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / norms
    return mat.tolist()


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Brute-force cosine distance for reference."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    sim = np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-12)
    return float(1.0 - sim)


# ═══════════════════════════════════════════════════════════════════════════
#  Tests that run against the *default* (preferred) backend
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildIndex:
    """build_index returns a valid index wrapper."""

    def test_returns_index_wrapper(self):
        embs = _make_embeddings(5, 8)
        idx = build_index(embs)
        assert isinstance(idx, _IndexWrapper)

    def test_dimension_stored(self):
        embs = _make_embeddings(5, 16)
        idx = build_index(embs, metric="cosine")
        assert idx.dim == 16

    def test_count_stored(self):
        embs = _make_embeddings(7, 8)
        idx = build_index(embs)
        assert idx.n == 7

    def test_unsupported_metric_raises(self):
        embs = _make_embeddings(3, 4)
        with pytest.raises(ValueError, match="Unsupported metric"):
            build_index(embs, metric="euclidean")

    def test_empty_embeddings_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_index([])

    def test_single_embedding(self):
        embs = _make_embeddings(1, 8)
        idx = build_index(embs)
        assert idx.n == 1


class TestQuery:
    """query returns correct ids and distances."""

    def test_returns_tuple_of_two_lists(self):
        embs = _make_embeddings(10, 8)
        idx = build_index(embs)
        ids, dists = query(idx, embs[0], k=3)
        assert isinstance(ids, list)
        assert isinstance(dists, list)

    def test_correct_k(self):
        embs = _make_embeddings(10, 8)
        idx = build_index(embs)
        ids, dists = query(idx, embs[0], k=5)
        assert len(ids) == 5
        assert len(dists) == 5

    def test_k_clamped_to_n(self):
        embs = _make_embeddings(3, 8)
        idx = build_index(embs)
        ids, dists = query(idx, embs[0], k=100)
        assert len(ids) == 3

    def test_self_match_is_top_1(self):
        """A vector should be its own nearest neighbour."""
        embs = _make_embeddings(10, 8)
        idx = build_index(embs)
        for i in range(len(embs)):
            ids, dists = query(idx, embs[i], k=1)
            assert ids[0] == i
            assert dists[0] < 1e-4  # nearly zero distance

    def test_distances_are_non_negative(self):
        embs = _make_embeddings(10, 8)
        idx = build_index(embs)
        _, dists = query(idx, embs[0], k=5)
        assert all(d >= -1e-6 for d in dists)

    def test_distances_sorted_ascending(self):
        embs = _make_embeddings(10, 8)
        idx = build_index(embs)
        _, dists = query(idx, embs[0], k=5)
        for a, b in zip(dists, dists[1:]):
            assert a <= b + 1e-6

    def test_ids_are_valid_indices(self):
        embs = _make_embeddings(10, 8)
        idx = build_index(embs)
        ids, _ = query(idx, embs[0], k=5)
        assert all(0 <= i < 10 for i in ids)

    def test_deterministic(self):
        embs = _make_embeddings(10, 8)
        idx = build_index(embs)
        r1 = query(idx, embs[3], k=3)
        r2 = query(idx, embs[3], k=3)
        assert r1 == r2


class TestSaveLoad:
    """Round-trip save → load preserves index behaviour."""

    def test_round_trip_faiss_or_pkl(self, tmp_path: Path):
        embs = _make_embeddings(10, 8)
        idx = build_index(embs)

        ext = ".faiss" if idx.backend == "faiss" else ".pkl"
        path = str(tmp_path / f"test_index{ext}")
        save_index(idx, path)

        loaded = load_index(path)
        assert loaded.dim == idx.dim
        assert loaded.n == idx.n

        # Query results must match
        ids_orig, dists_orig = query(idx, embs[0], k=3)
        ids_load, dists_load = query(loaded, embs[0], k=3)
        assert ids_orig == ids_load
        for d1, d2 in zip(dists_orig, dists_load):
            assert abs(d1 - d2) < 1e-5

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        embs = _make_embeddings(3, 4)
        idx = build_index(embs)
        ext = ".faiss" if idx.backend == "faiss" else ".pkl"
        nested = tmp_path / "a" / "b" / f"index{ext}"
        save_index(idx, str(nested))
        assert nested.exists()


class TestBackendName:
    """backend_name returns a known string."""

    def test_returns_string(self):
        name = backend_name()
        assert name in ("faiss", "sklearn")

    def test_consistent_with_flag(self):
        if _FAISS_AVAILABLE:
            assert backend_name() == "faiss"
        else:
            assert backend_name() == "sklearn"


# ═══════════════════════════════════════════════════════════════════════════
#  Force-sklearn tests (always exercises the fallback path)
# ═══════════════════════════════════════════════════════════════════════════


class TestSklearnFallback:
    """Exercises the sklearn code-path even when faiss is installed."""

    @staticmethod
    def _build_sklearn_index(embeddings: list[list[float]]) -> _IndexWrapper:
        """Build an index using the sklearn backend regardless of faiss."""
        from sklearn.neighbors import NearestNeighbors

        mat = np.asarray(embeddings, dtype=np.float32)
        n, dim = mat.shape
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(mat)
        return _IndexWrapper(nn, "sklearn", dim, n)

    def test_sklearn_build_and_query(self):
        embs = _make_embeddings(10, 8)
        idx = self._build_sklearn_index(embs)
        ids, dists = query(idx, embs[0], k=3)
        assert len(ids) == 3
        assert ids[0] == 0
        assert dists[0] < 1e-4

    def test_sklearn_self_match(self):
        embs = _make_embeddings(10, 8)
        idx = self._build_sklearn_index(embs)
        for i in range(len(embs)):
            ids, _ = query(idx, embs[i], k=1)
            assert ids[0] == i

    def test_sklearn_save_load_round_trip(self, tmp_path: Path):
        embs = _make_embeddings(6, 8)
        idx = self._build_sklearn_index(embs)
        path = str(tmp_path / "test.pkl")
        save_index(idx, path)
        loaded = load_index(path)
        assert loaded.dim == 8
        assert loaded.n == 6
        ids, dists = query(loaded, embs[0], k=2)
        assert ids[0] == 0

    def test_sklearn_deterministic(self):
        embs = _make_embeddings(10, 8)
        idx = self._build_sklearn_index(embs)
        r1 = query(idx, embs[2], k=3)
        r2 = query(idx, embs[2], k=3)
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_single_vector_self_query(self):
        embs = _make_embeddings(1, 8)
        idx = build_index(embs)
        ids, dists = query(idx, embs[0], k=1)
        assert ids == [0]
        assert dists[0] < 1e-4

    def test_large_dim(self):
        embs = _make_embeddings(5, 384)
        idx = build_index(embs)
        ids, _ = query(idx, embs[0], k=1)
        assert ids[0] == 0

    def test_identical_vectors(self):
        """All-same vectors: any of them can be the top match."""
        vec = [1.0] * 8
        embs = [vec[:] for _ in range(5)]
        idx = build_index(embs)
        ids, dists = query(idx, vec, k=5)
        assert len(ids) == 5
        # All distances should be ~0
        assert all(d < 1e-4 for d in dists)

    def test_k_equals_n(self):
        embs = _make_embeddings(5, 8)
        idx = build_index(embs)
        ids, dists = query(idx, embs[0], k=5)
        assert len(ids) == 5
        assert sorted(ids) == [0, 1, 2, 3, 4]
