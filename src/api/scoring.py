"""
scoring.py – Real brand-consistency scoring and edit-plan generation.

Replaces the mock functions in main.py with actual NLP-based scoring that
uses the feature extraction pipeline to compare input text against a stored
brand profile.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.feature_extraction.sentiment_extractor import extract_sentiment
from src.feature_extraction.formality_extractor import extract_formality
from src.feature_extraction.readability_extractor import flesch_reading_ease
from src.feature_extraction.vocabulary_extractor import extract_vocab_metrics
from src.feature_extraction.feature_utils import clean_text, word_tokenize

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────

def _pct_match(actual: float, target: float, max_deviation: float) -> float:
    """
    Return a 0–100 percentage score for how close *actual* is to *target*.

    ``max_deviation`` is the value at which the score drops to 0.
    If target is 0 and actual is 0, return 100.
    """
    if max_deviation <= 0:
        return 100.0
    diff = abs(actual - target)
    return max(0.0, min(100.0, (1.0 - diff / max_deviation) * 100.0))


def _keyword_overlap(text_tokens: list[str], brand_keywords: list[str]) -> float:
    """
    Return 0–100 overlap score: what fraction of brand keywords appear in text.
    """
    if not brand_keywords:
        return 50.0  # neutral when no brand keywords defined
    text_lower = {t.lower() for t in text_tokens}
    brand_lower = {k.lower() for k in brand_keywords}
    hits = text_lower & brand_lower
    return min(100.0, (len(hits) / len(brand_lower)) * 100.0)


# ── Public API ────────────────────────────────────────────────────────────

def score_consistency(text: str, brand_profile: dict[str, Any]) -> dict[str, Any]:
    """
    Score how consistent *text* is with *brand_profile*.

    Returns dict with keys:
        overall_score, tone_pct, vocab_overlap_pct,
        sentiment_alignment_pct, readability_match_pct
    All values are 0–100 floats.
    """
    cleaned = clean_text(text) or ""

    # ── Extract features from the input text ──────────────────────────────
    text_sentiment = extract_sentiment(cleaned)       # [0, 1]
    text_formality = extract_formality(cleaned)       # [0, 1]
    text_readability = flesch_reading_ease(cleaned)    # [0, ~120]
    vocab = extract_vocab_metrics(cleaned)
    text_tokens = word_tokenize(cleaned) if cleaned else []

    # ── Brand profile targets (with sensible defaults) ────────────────────
    bp = brand_profile or {}
    target_sentiment = bp.get("avg_sentiment", bp.get("sentiment", 0.5))
    target_formality = bp.get("avg_formality", bp.get("formality", 0.5))
    target_readability = bp.get("avg_readability_flesch",
                                bp.get("readability_flesch", 50.0))
    brand_keywords = bp.get("top_keywords", [])

    # ── Per-dimension scores ──────────────────────────────────────────────
    tone_pct = _pct_match(text_formality, target_formality, max_deviation=0.5)
    sentiment_pct = _pct_match(text_sentiment, target_sentiment, max_deviation=0.5)
    readability_pct = _pct_match(text_readability, target_readability,
                                 max_deviation=60.0)
    vocab_pct = _keyword_overlap(text_tokens, brand_keywords)

    # ── Overall (weighted average) ────────────────────────────────────────
    overall = (
        0.30 * tone_pct
        + 0.25 * vocab_pct
        + 0.25 * sentiment_pct
        + 0.20 * readability_pct
    )

    return {
        "overall_score": round(overall, 1),
        "tone_pct": round(tone_pct, 1),
        "vocab_overlap_pct": round(vocab_pct, 1),
        "sentiment_alignment_pct": round(sentiment_pct, 1),
        "readability_match_pct": round(readability_pct, 1),
    }


def generate_edit_plan(text: str, brand_profile: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a structured edit plan for aligning *text* to *brand_profile*.
    """
    bp = brand_profile or {}
    brand_id = bp.get("brand_id", "unknown")
    brand_keywords = bp.get("top_keywords", ["precision", "excellence"])
    tone_label = bp.get("tone_label", bp.get("tone", "authoritative"))

    cleaned = clean_text(text) or ""
    text_formality = extract_formality(cleaned)
    text_sentiment = extract_sentiment(cleaned)
    target_formality = bp.get("avg_formality", bp.get("formality", 0.5))
    target_sentiment = bp.get("avg_sentiment", bp.get("sentiment", 0.5))

    goals: list[str] = []
    style_rules: list[str] = []

    # Formality direction
    if text_formality < target_formality - 0.1:
        goals.append("Increase formality to match brand voice")
        style_rules.append("Use formal sentence structures")
        style_rules.append("Avoid contractions")
    elif text_formality > target_formality + 0.1:
        goals.append("Reduce formality for approachability")
        style_rules.append("Use shorter, more conversational sentences")

    # Sentiment direction
    if text_sentiment < target_sentiment - 0.1:
        goals.append("Raise sentiment closer to brand mean")
    elif text_sentiment > target_sentiment + 0.1:
        goals.append("Moderate sentiment to avoid over-enthusiasm")

    # Readability
    text_read = flesch_reading_ease(cleaned)
    target_read = bp.get("avg_readability_flesch",
                         bp.get("readability_flesch", 50.0))
    if text_read > target_read + 10:
        goals.append("Reduce reading ease (increase sophistication)")
    elif text_read < target_read - 10:
        goals.append("Increase reading ease for accessibility")

    if not goals:
        goals.append("Fine-tune vocabulary to strengthen brand alignment")

    # Avoid terms: common informal words that clash with luxury/formal brands
    avoid_terms = ["awesome", "cool", "super", "stuff", "things", "nice",
                   "basically", "literally", "pretty much"]

    return {
        "brand_id": brand_id,
        "goals": goals,
        "avoid_terms": avoid_terms,
        "prefer_terms": brand_keywords[:10],
        "style_rules": style_rules or ["Maintain current sentence structure"],
        "tone_direction": tone_label,
        "grounding_chunks": [],  # populated by caller with RAG results
    }
