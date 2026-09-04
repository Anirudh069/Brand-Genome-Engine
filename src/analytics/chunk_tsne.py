"""
chunk_tsne.py – Chunk-level t-SNE projection of brand_chunks.

Source is ALWAYS brand_chunks (never brand-profile centroids). Sampling is
deterministic: up to ``MAX_CHUNKS_PER_BRAND`` chunks per brand, chosen with a
fixed-seed ``numpy.random.RandomState`` from chunks sorted by chunk_id, so
the sampled chunk_id set is reproducible across runs.
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE

from src.feature_extraction.embedding_extractor import get_embedding

RANDOM_STATE = 42
MAX_CHUNKS_PER_BRAND = 50


def sample_chunks(chunks: list[dict], max_per_brand: int = MAX_CHUNKS_PER_BRAND) -> list[dict]:
    by_brand: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_brand.setdefault(chunk["brand_id"], []).append(chunk)

    sampled: list[dict] = []
    rng = np.random.RandomState(RANDOM_STATE)
    for brand_id in sorted(by_brand.keys()):
        brand_chunks = sorted(by_brand[brand_id], key=lambda c: c["chunk_id"])
        if len(brand_chunks) > max_per_brand:
            indices = rng.choice(len(brand_chunks), size=max_per_brand, replace=False)
            indices.sort()
            brand_chunks = [brand_chunks[i] for i in indices]
        sampled.extend(brand_chunks)
    return sampled


def compute_chunk_tsne(chunks: list[dict]) -> dict:
    """
    Parameters
    ----------
    chunks : list of {"chunk_id", "brand_id", "brand_name", "chunk_text"}

    Returns
    -------
    {"points": [{"chunk_id","brand_id","brand_name","x","y"}, ...],
     "random_state": 42, "perplexity": int, "sample_total": int}
    """
    sampled = sample_chunks(chunks)
    if len(sampled) < 4:
        return {"points": [], "random_state": RANDOM_STATE, "perplexity": 0, "sample_total": 0}

    embeddings = np.array(
        [get_embedding(chunk["chunk_text"])[0] for chunk in sampled],
        dtype=np.float64,
    )

    perplexity = max(5, min(30, (len(sampled) - 1) // 3))
    perplexity = min(perplexity, len(sampled) - 1)

    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)

    points = [
        {
            "chunk_id": chunk["chunk_id"],
            "brand_id": chunk["brand_id"],
            "brand_name": chunk["brand_name"],
            "x": float(x),
            "y": float(y),
        }
        for chunk, (x, y) in zip(sampled, coords)
    ]

    return {
        "points": points,
        "random_state": RANDOM_STATE,
        "perplexity": perplexity,
        "sample_total": len(sampled),
    }
