"""
retrieval.py – Brand-level competitor retrieval and FAISS/sklearn index.

Provides a backend-agnostic API for building, saving, loading, and querying
a nearest-neighbour index over brand embeddings.

Backend selection
-----------------
* **FAISS** (``faiss-cpu``) is preferred.  If it cannot be imported the
  module falls back transparently to **scikit-learn**
  ``NearestNeighbors(metric='cosine')``.
* The active backend is logged exactly once on first ``build_index`` call.

Public API
----------
* ``build_index(embeddings, metric)`` → opaque index object
* ``save_index(index, path)``
* ``load_index(path)``
* ``query(index, query_embedding, k)`` → ``(ids, distances)``

All functions are **CPU-only and deterministic**.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Backend detection (evaluated once at import time) ─────────────────────
_FAISS_AVAILABLE: bool
try:
    import faiss  # type: ignore[import-untyped]

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

_BACKEND_LOGGED = False


def _log_backend_once() -> None:
    """Emit a single INFO log the first time the backend is used."""
    global _BACKEND_LOGGED  # noqa: PLW0603
    if not _BACKEND_LOGGED:
        backend = "faiss-cpu" if _FAISS_AVAILABLE else "sklearn (NearestNeighbors)"
        logger.info("Retrieval backend: %s", backend)
        _BACKEND_LOGGED = True


def backend_name() -> str:
    """Return a human-readable string identifying the active backend."""
    return "faiss" if _FAISS_AVAILABLE else "sklearn"


# ── Index wrapper ─────────────────────────────────────────────────────────

class _IndexWrapper:
    """Thin wrapper so save / load / query have a uniform interface."""

    __slots__ = ("_impl", "_backend", "_dim", "_n")

    def __init__(self, impl: Any, backend: str, dim: int, n: int) -> None:
        self._impl = impl
        self._backend = backend
        self._dim = dim
        self._n = n

    @property
    def impl(self) -> Any:
        return self._impl

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n(self) -> int:
        return self._n


# ── Public API ────────────────────────────────────────────────────────────

def build_index(
    embeddings: list[list[float]],
    metric: str = "cosine",
) -> _IndexWrapper:
    """
    Build a nearest-neighbour index over *embeddings*.

    Parameters
    ----------
    embeddings : list[list[float]]
        Matrix of shape ``(n, dim)`` – one row per brand.
    metric : str
        Distance metric.  Only ``"cosine"`` is currently supported.

    Returns
    -------
    _IndexWrapper
        Opaque object accepted by :func:`save_index`, :func:`load_index`,
        and :func:`query`.

    Raises
    ------
    ValueError
        If *embeddings* is empty or *metric* is unsupported.
    """
    if metric != "cosine":
        raise ValueError(f"Unsupported metric {metric!r}; only 'cosine' is supported")
    if not embeddings:
        raise ValueError("embeddings must be non-empty")

    _log_backend_once()

    mat = np.asarray(embeddings, dtype=np.float32)
    n, dim = mat.shape

    if _FAISS_AVAILABLE:
        # Normalise so inner-product == cosine similarity
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        return _IndexWrapper(index, "faiss", dim, n)

    # sklearn fallback
    from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(mat)
    return _IndexWrapper(nn, "sklearn", dim, n)


def save_index(index: _IndexWrapper, path: str) -> None:
    """
    Persist *index* to disk.

    * FAISS backend → uses ``faiss.write_index`` (binary file).
    * sklearn backend → uses ``pickle``.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if index.backend == "faiss":
        faiss.write_index(index.impl, str(path))
    else:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "impl": index.impl,
                    "dim": index.dim,
                    "n": index.n,
                },
                fh,
            )
    logger.info("Index saved → %s", path)


def load_index(path: str) -> _IndexWrapper:
    """
    Load a previously saved index from *path*.

    Automatically detects whether the file is FAISS binary or a pickle.
    """
    _log_backend_once()

    path_str = str(path)

    if _FAISS_AVAILABLE and path_str.endswith(".faiss"):
        impl = faiss.read_index(path_str)
        return _IndexWrapper(impl, "faiss", impl.d, impl.ntotal)

    # sklearn / pickle path
    with open(path_str, "rb") as fh:
        data = pickle.load(fh)  # noqa: S301
    return _IndexWrapper(data["impl"], "sklearn", data["dim"], data["n"])


def query(
    index: _IndexWrapper,
    query_embedding: list[float],
    k: int = 5,
) -> tuple[list[int], list[float]]:
    """
    Find the *k* nearest neighbours of *query_embedding* in *index*.

    Parameters
    ----------
    index : _IndexWrapper
        An index built by :func:`build_index` (or loaded via :func:`load_index`).
    query_embedding : list[float]
        Dense vector of same dimensionality as the indexed embeddings.
    k : int
        Number of neighbours to return.

    Returns
    -------
    tuple[list[int], list[float]]
        * **ids** – indices into the original *embeddings* list.
        * **distances** – corresponding distances (lower = more similar for
          cosine distance; for FAISS inner-product, we return ``1 − score``
          so the semantics are *distance*, not similarity).
    """
    k = min(k, index.n)
    vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

    if index.backend == "faiss":
        faiss.normalize_L2(vec)
        scores, ids = index.impl.search(vec, k)
        # Convert inner-product similarity → cosine distance
        distances = (1.0 - scores[0]).tolist()
        return ids[0].tolist(), distances

    # sklearn path
    dists, ids = index.impl.kneighbors(vec, n_neighbors=k)
    return ids[0].tolist(), dists[0].tolist()
