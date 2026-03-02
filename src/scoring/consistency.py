"""
consistency.py –* **readability_match_pct** – Gaussian similarity on Flesch score vs profile mean.
* **tone_pct** – Gaussian similarity on formality vs profile mean.
  (Cosine embedding similarity reserved for offline batch pipeline.)nical brand-consistency scorer.

Public API
----------
    compute_consistency_score(text: str, brand_profile: dict) -> dict
    generate_edit_plan(text: str, brand_profile: dict) -> dict

The returned dict from ``compute_consistency_score`` has **exactly** these keys
(all floats clamped to [0, 100]):

    overall_score, tone_pct, vocab_overlap_pct,
    sentiment_alignment_pct, readability_match_pct

Algorithm (from ``docs/scoring_spec.md``):

* **vocab_overlap_pct** – Jaccard similarity of text content-words vs
  ``brand_profile["top_keywords"]``.
* **sentiment_alignment_pct** – Gaussian similarity:
  ``exp(-((s - μ)² / (2σ²))) * 100`` where μ/σ come from the profile.
* **readability_match_pct** – Inverse-distance with adaptive tolerance:
  ``max(0, 1 - |f - μ_f| / tolerance) * 100``.
* **tone_pct** – Formality-distance proxy:
  ``max(0, 1 - |formality_text - formality_brand|) * 100``.
  (Cosine embedding similarity reserved for offline batch pipeline.)
* **overall_score** – Weighted average:
  ``0.30*tone + 0.25*sentiment + 0.25*vocab + 0.20*readability``.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Feature extractors (real pipeline) ────────────────────────────────────
# Imported eagerly – they are lightweight (no model load at import time).

from src.feature_extraction.sentiment_extractor import extract_sentiment
from src.feature_extraction.formality_extractor import extract_formality
from src.feature_extraction.readability_extractor import flesch_reading_ease
from src.feature_extraction.feature_utils import clean_text, word_tokenize

# NOTE: Embedding model is NOT loaded at scoring time to avoid segfaults
# on CPython 3.9 + macOS when faiss-cpu is co-loaded.  Tone uses a
# formality-distance proxy instead.  Cosine-embedding tone is reserved
# for the offline batch pipeline.

# ── Text helpers ──────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z']+")

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "as", "at", "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those", "by", "from", "you", "we",
    "they", "he", "she", "i", "our", "your", "their", "its", "not", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "can", "all", "more", "also", "than", "into", "which", "about",
    "so", "if", "when", "what", "there", "each", "just", "most", "other",
    "some", "such", "only", "over", "new", "very", "after", "before",
    "between", "been",
})


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _content_words(text: str) -> list[str]:
    return [w for w in _tokenize(text) if w not in _STOPWORDS and len(w) >= 3]


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# ── Math primitives (from scoring spec) ───────────────────────────────────

def _jaccard(set_a: list[str], set_b: list[str]) -> float:
    """Jaccard similarity.  Both empty → 0 (not 100)."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _gaussian_similarity(value: float, mean: float, std: float) -> float:
    """exp(-((value-mean)² / (2*std²))).  std clamped to ≥ 0.01."""
    std = max(std, 0.01)
    return math.exp(-((value - mean) ** 2) / (2 * std ** 2))


def _inverse_distance(value: float, mean: float, tolerance: float) -> float:
    """max(0, 1 - |value-mean| / tolerance).  tolerance clamped to ≥ 20."""
    tolerance = max(tolerance, 20.0)
    return max(0.0, 1.0 - abs(value - mean) / tolerance)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity.  Zero-vector → 0.  Dimension mismatch → 0 + log."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        logger.warning(
            "Embedding dimension mismatch: %d vs %d — returning 0.",
            len(a), len(b),
        )
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Safe profile field access ─────────────────────────────────────────────

def _float(profile: dict, key: str, fallback: float) -> float:
    """Get a float from *profile*, returning *fallback* on any error."""
    try:
        return float(profile.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_consistency_score(text: str, brand_profile: dict[str, Any]) -> dict[str, Any]:
    """
    Score how consistent *text* is with *brand_profile*.

    Returns a dict with exactly five keys (all floats, 0–100):
        overall_score, tone_pct, vocab_overlap_pct,
        sentiment_alignment_pct, readability_match_pct

    Never raises on bad input — returns zeros and logs warnings.
    """
    bp = brand_profile or {}

    # ── Short-text guard ──────────────────────────────────────────────────
    words = _tokenize(text or "")
    if len(words) < 10:
        return {
            "overall_score": 0.0,
            "tone_pct": 0.0,
            "vocab_overlap_pct": 0.0,
            "sentiment_alignment_pct": 0.0,
            "readability_match_pct": 0.0,
        }

    # ── Extract features from input text ──────────────────────────────────
    cleaned = clean_text(text) or text or ""
    text_sentiment = extract_sentiment(cleaned)          # [0, 1]
    text_formality = extract_formality(cleaned)          # [0, 1]
    text_flesch = flesch_reading_ease(cleaned)            # ~0–120
    text_keywords = _content_words(cleaned)

    # ── Brand profile targets ─────────────────────────────────────────────
    mean_sentiment = _float(bp, "mean_sentiment", _float(bp, "avg_sentiment", 0.5))
    std_sentiment = _float(bp, "std_sentiment", 0.15)
    mean_flesch = _float(bp, "mean_flesch", _float(bp, "avg_readability_flesch", 50.0))
    std_flesch = _float(bp, "std_flesch", 10.0)
    mean_formality = _float(bp, "mean_formality", _float(bp, "avg_formality", 0.5))
    std_formality = _float(bp, "std_formality", 0.05)
    brand_keywords: list[str] = bp.get("top_keywords", []) or []

    # ── 1) Vocabulary overlap (Jaccard) ───────────────────────────────────
    vocab_overlap = _jaccard(text_keywords, brand_keywords)

    # ── 2) Sentiment alignment (Gaussian) ─────────────────────────────────
    sentiment_align = _gaussian_similarity(text_sentiment, mean_sentiment, std_sentiment)

    # ── 3) Readability match (Gaussian) ───────────────────────────────────
    readability = _gaussian_similarity(text_flesch, mean_flesch, max(std_flesch, 5.0))

    # ── 4) Tone (Gaussian formality-distance) ─────────────────────────────
    tone = _gaussian_similarity(text_formality, mean_formality, max(std_formality, 0.01))

    # ── Overall (weighted average from spec) ──────────────────────────────
    overall = (
        0.30 * tone
        + 0.25 * sentiment_align
        + 0.25 * vocab_overlap
        + 0.20 * readability
    ) * 100.0

    return {
        "overall_score": round(_clamp(overall), 1),
        "tone_pct": round(_clamp(tone * 100.0), 1),
        "vocab_overlap_pct": round(_clamp(vocab_overlap * 100.0), 1),
        "sentiment_alignment_pct": round(_clamp(sentiment_align * 100.0), 1),
        "readability_match_pct": round(_clamp(readability * 100.0), 1),
    }


def generate_edit_plan(text: str, brand_profile: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a structured edit plan for aligning *text* to *brand_profile*.
    """
    bp = brand_profile or {}
    brand_id = bp.get("brand_id", "unknown")
    brand_keywords = bp.get("top_keywords", ["precision", "excellence"])
    tone_label = bp.get("tone_label", "authoritative")

    cleaned = clean_text(text) or text or ""
    text_formality = extract_formality(cleaned)
    text_sentiment = extract_sentiment(cleaned)
    text_readability = flesch_reading_ease(cleaned)

    target_formality = _float(bp, "mean_formality", _float(bp, "avg_formality", 0.5))
    target_sentiment = _float(bp, "mean_sentiment", _float(bp, "avg_sentiment", 0.5))
    target_readability = _float(bp, "mean_flesch", _float(bp, "avg_readability_flesch", 50.0))

    goals: list[str] = []
    style_rules: list[str] = []

    if text_formality < target_formality - 0.1:
        goals.append("Increase formality to match brand voice")
        style_rules.append("Use formal sentence structures")
        style_rules.append("Avoid contractions")
    elif text_formality > target_formality + 0.1:
        goals.append("Reduce formality for approachability")
        style_rules.append("Use shorter, more conversational sentences")

    if text_sentiment < target_sentiment - 0.1:
        goals.append("Raise sentiment closer to brand mean")
    elif text_sentiment > target_sentiment + 0.1:
        goals.append("Moderate sentiment to avoid over-enthusiasm")

    if text_readability > target_readability + 10:
        goals.append("Reduce reading ease (increase sophistication)")
    elif text_readability < target_readability - 10:
        goals.append("Increase reading ease for accessibility")

    if not goals:
        goals.append("Fine-tune vocabulary to strengthen brand alignment")

    avoid_terms = [
        "awesome", "cool", "super", "stuff", "things", "nice",
        "basically", "literally", "pretty much",
    ]

    return {
        "brand_id": brand_id,
        "goals": goals,
        "avoid_terms": avoid_terms,
        "prefer_terms": brand_keywords[:10],
        "style_rules": style_rules or ["Maintain current sentence structure"],
        "tone_direction": tone_label,
        "grounding_chunks": [],
    }
