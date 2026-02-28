"""
embedding_extractor.py – Dense-vector embeddings via sentence-transformers.

If ``sentence-transformers`` is installed, loads the model lazily on first
call and caches it (CPU-only, deterministic).  If the library is unavailable
or any error occurs, falls back gracefully to a deterministic hash-based
pseudo-embedding of the correct dimensionality.

Public API
----------
* ``get_embedding(text) -> (embedding, model_name)`` — canonical entry point.
* ``extract_embedding(text, model_name)`` — legacy helper (returns just the vector).

Contract: all-MiniLM-L6-v2 → **384 dimensions**.
"""

from __future__ import annotations

import hashlib
import logging
import random
import struct

from src.feature_extraction.feature_utils import clean_text

logger = logging.getLogger(__name__)

# Contract: all-MiniLM-L6-v2 → 384 dimensions
_EMBEDDING_DIM = 384
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Placeholder text used when input is empty/None so that the model still
# receives a valid string.  A single space produces a near-zero vector in
# most transformer models.
_PLACEHOLDER_TEXT = " "

# ── Lazy model cache ──────────────────────────────────────────────────────
_model_cache: dict[str, object] = {}


def _seed_deterministic() -> None:
    """Pin random seeds for reproducibility (CPU-only)."""
    random.seed(42)
    try:
        import numpy as np  # type: ignore[import-untyped]
        np.random.seed(42)
    except ImportError:
        pass
    try:
        import torch  # type: ignore[import-untyped]
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(False)  # avoid CUBLAS errors on CPU
    except ImportError:
        pass


def _get_model(model_name: str):
    """Load a SentenceTransformer model (CPU-only), caching across calls."""
    if model_name not in _model_cache:
        _seed_deterministic()
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        model = SentenceTransformer(model_name, device="cpu")
        _model_cache[model_name] = model
    return _model_cache[model_name]


def _hash_fallback(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    """
    Deterministic fallback: produce a *dim*-length float vector by
    repeatedly hashing the input text.  The values are in [-1, 1] and
    the vector is normalised to unit length.

    This is NOT a semantic embedding – it simply guarantees the correct
    shape so downstream code never breaks.
    """
    result: list[float] = []
    seed = text.encode("utf-8", errors="replace")
    idx = 0
    while len(result) < dim:
        digest = hashlib.sha256(seed + idx.to_bytes(4, "big")).digest()
        # Each sha256 gives 32 bytes → 8 floats (4 bytes each)
        for i in range(0, 32, 4):
            if len(result) >= dim:
                break
            # Unpack 4 bytes as unsigned int, map to [-1, 1]
            (val,) = struct.unpack(">I", digest[i : i + 4])
            result.append(val / (2**32 - 1) * 2.0 - 1.0)
        idx += 1

    # L2-normalise
    norm = sum(v * v for v in result) ** 0.5
    if norm > 0:
        result = [v / norm for v in result]

    return result


# ── Public API ────────────────────────────────────────────────────────────

def get_embedding(
    text: str | None,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> tuple[list[float], str]:
    """
    Return ``(embedding, model_name)`` for *text*.

    Parameters
    ----------
    text : str | None
        Raw or pre-cleaned input text.
    model_name : str
        HuggingFace model identifier.  Must produce 384-d vectors.

    Returns
    -------
    tuple[list[float], str]
        * **embedding** – list of 384 floats.
        * **model_name** – the model identifier that was (or would have been)
          used.  When the hash fallback is active the name is still returned
          so callers know which model was *intended*.

    Behaviour
    ---------
    * CPU-only; deterministic (seeds are pinned).
    * Never throws – returns a zero vector on catastrophic failure.
    * Uses caching so the model is loaded only once per process.
    * ``None`` / empty *text* → the model is fed a placeholder string
      (``" "``) when the real model is available, or a zero vector when
      it is not.
    """
    try:
        cleaned = clean_text(text)
        encode_text = cleaned if cleaned else _PLACEHOLDER_TEXT
        use_placeholder = not cleaned

        # Try the real model first
        try:
            _seed_deterministic()
            model = _get_model(model_name)
            vec = model.encode(encode_text).tolist()  # type: ignore[union-attr]
            if len(vec) == _EMBEDDING_DIM:
                return vec, model_name
            logger.warning(
                "Model %s produced %d dims (expected %d); using fallback",
                model_name,
                len(vec),
                _EMBEDDING_DIM,
            )
        except ImportError:
            logger.info(
                "sentence-transformers not installed; using hash fallback"
            )
        except Exception:
            logger.warning(
                "SentenceTransformer encoding failed; using hash fallback",
                exc_info=True,
            )

        # Deterministic fallback
        if use_placeholder:
            return [0.0] * _EMBEDDING_DIM, model_name
        return _hash_fallback(encode_text, _EMBEDDING_DIM), model_name

    except Exception:
        logger.exception("get_embedding failed – returning zero vector")
        return [0.0] * _EMBEDDING_DIM, model_name


def extract_embedding(
    text: str | None,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> list[float]:
    """
    Return a dense embedding vector of length **384** (MiniLM contract).

    This is a convenience wrapper around :func:`get_embedding` that
    discards the model name and returns only the vector.

    The function **never raises**.
    """
    embedding, _ = get_embedding(text, model_name=model_name)
    return embedding
