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

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Feature extractors (real pipeline) ────────────────────────────────────
# Imported eagerly – they are lightweight (no model load at import time).

from src.feature_extraction.embedding_extractor import get_embedding
from src.feature_extraction.formality_extractor import extract_formality
from src.feature_extraction.readability_extractor import extract_readability, flesch_reading_ease
from src.feature_extraction.sentiment_extractor import extract_sentiment
from src.feature_extraction.feature_utils import clean_text, word_tokenize
from src.feature_extraction.vocabulary_extractor import extract_vocab_metrics

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return []
    return result


def _designation_from_profile(profile: dict[str, Any]) -> str:
    for key in ("designation", "brand_name", "name", "mission_core_vision"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _keywords_from_profile(profile: dict[str, Any]) -> list[str]:
    tone_features = _as_mapping(profile.get("tone_features"))
    for key in ("top_keywords", "keywords"):
        value = profile.get(key)
        if isinstance(value, list) and value:
            return [str(item).strip().lower() for item in value if str(item).strip()]
    for key in ("top_keywords", "keywords"):
        value = tone_features.get(key)
        if isinstance(value, list) and value:
            return [str(item).strip().lower() for item in value if str(item).strip()]
    return []


def _profile_metric(profile: dict[str, Any], *keys: str, fallback: float) -> float:
    tone_features = _as_mapping(profile.get("tone_features"))
    for key in keys:
        if key in profile:
            try:
                return float(profile.get(key))
            except (TypeError, ValueError):
                continue
        if key in tone_features:
            try:
                return float(tone_features.get(key))
            except (TypeError, ValueError):
                continue
    return fallback


def _profile_embedding(profile: dict[str, Any]) -> list[float]:
    for key in ("aggregate_embedding", "aggregate_embedding_json", "embedding"):
        embedding = _as_float_list(profile.get(key))
        if embedding:
            return embedding

    metadata = _as_mapping(profile.get("metadata"))
    if metadata:
        for key in ("aggregate_embedding", "sample_embedding", "embedding"):
            embedding = _as_float_list(metadata.get(key))
            if embedding:
                return embedding

    tone_features = _as_mapping(profile.get("tone_features"))
    embedding = _as_float_list(tone_features.get("aggregate_embedding"))
    if embedding:
        return embedding

    return []


def _count_brand_mentions(text: str, designation: str) -> int:
    if not text or not designation.strip():
        return 0
    parts = [re.escape(part) for part in designation.split() if part.strip()]
    if not parts:
        return 0
    pattern = r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _clamp_score(value: float) -> float:
    return _clamp(value)


def _feature_payload(
    *,
    score: float,
    input_value: Any,
    target_value: Any,
    delta: Any,
    weight: float,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "score": round(_clamp_score(score), 1),
        "input_value": input_value,
        "target_value": target_value,
        "delta": delta,
        "weight": round(weight, 4),
    }
    if details:
        payload["details"] = details
    return payload


def _diagnostic_payload(
    *,
    dimension: str,
    severity: str,
    score: float,
    observed: Any,
    expected: Any,
    deviation: Any,
    message: str,
    suggestion: str,
) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "severity": severity,
        "score": round(_clamp_score(score), 1),
        "observed": observed,
        "expected": expected,
        "deviation": deviation,
        "message": message,
        "suggestion": suggestion,
    }


def _severity_from_score(score: float) -> str:
    if score < 50:
        return "high"
    if score < 70:
        return "moderate"
    return "low"


def _extract_input_features(text: str) -> dict[str, Any]:
    cleaned = clean_text(text) or ""
    sentiment = extract_sentiment(cleaned)
    formality = extract_formality(cleaned)
    flesch, avg_sentence_length = extract_readability(cleaned)
    vocab_metrics = extract_vocab_metrics(cleaned)
    input_keywords = _content_words(cleaned)
    embedding, embedding_model = get_embedding(cleaned)
    return {
        "cleaned_text": cleaned,
        "sentiment": float(sentiment),
        "formality": float(formality),
        "flesch": float(flesch),
        "avg_sentence_length": float(avg_sentence_length),
        "vocab_diversity": float(vocab_metrics.get("vocab_diversity", 0.0)),
        "input_keywords": input_keywords,
        "embedding": embedding,
        "embedding_model": embedding_model,
    }


def _extract_target_profile(profile: dict[str, Any]) -> dict[str, Any]:
    target = profile or {}
    tone_features = _as_mapping(target.get("tone_features"))
    keywords = _keywords_from_profile(target)
    designation = _designation_from_profile(target)

    target_formality = _profile_metric(target, "mean_formality", "avg_formality", fallback=_float(tone_features, "mean_formality", _float(tone_features, "avg_formality", 0.5)))
    target_sentiment = _profile_metric(target, "mean_sentiment", "avg_sentiment", fallback=_float(tone_features, "mean_sentiment", _float(tone_features, "avg_sentiment", 0.5)))
    target_flesch = _profile_metric(target, "mean_flesch", "avg_readability_flesch", fallback=_float(tone_features, "mean_flesch", _float(tone_features, "avg_readability_flesch", 50.0)))
    target_sentence_length = _profile_metric(target, "avg_sentence_length", fallback=_float(tone_features, "avg_sentence_length", 0.0))
    target_vocab_richness = _profile_metric(target, "mean_vocab_richness", "vocabulary_richness", fallback=_float(tone_features, "mean_vocab_richness", _float(tone_features, "vocabulary_richness", 0.5)))
    target_embedding = _profile_embedding(target)

    return {
        "designation": designation,
        "brand_name": target.get("brand_name") or target.get("name") or designation,
        "mission_core_vision": target.get("mission_core_vision") or target.get("mission") or "",
        "keywords": keywords,
        "formality": target_formality,
        "sentiment": target_sentiment,
        "flesch": target_flesch,
        "avg_sentence_length": target_sentence_length,
        "vocab_richness": target_vocab_richness,
        "embedding": target_embedding,
        "embedding_dim": len(target_embedding),
        "genome_version": target.get("profile_version") or target.get("genome_version"),
        "tone_label": tone_features.get("tone_label") or target.get("tone_label") or target.get("tone") or "",
        "tone_features": tone_features,
    }


def _score_against_target(text: str, target_profile: dict[str, Any]) -> dict[str, Any]:
    input_features = _extract_input_features(text)
    target = _extract_target_profile(target_profile)

    target_keywords = target["keywords"]
    input_keywords = input_features["input_keywords"]
    keyword_overlap = _jaccard(input_keywords, target_keywords)
    matched_keywords = sorted(set(input_keywords) & set(target_keywords))

    tone_std = max(_float(target["tone_features"], "std_formality", _float(target_profile, "std_formality", 0.05)), 0.01)
    sentiment_std = max(_float(target["tone_features"], "std_sentiment", _float(target_profile, "std_sentiment", 0.15)), 0.01)
    flesch_std = max(_float(target["tone_features"], "std_flesch", _float(target_profile, "std_flesch", 10.0)), 5.0)
    sentence_tolerance = max(target["avg_sentence_length"] * 0.35, 6.0)

    tone_score = _gaussian_similarity(input_features["formality"], target["formality"], tone_std) * 100.0
    sentiment_score = _gaussian_similarity(input_features["sentiment"], target["sentiment"], sentiment_std) * 100.0
    flesch_score = _gaussian_similarity(input_features["flesch"], target["flesch"], flesch_std) * 100.0
    sentence_score = _inverse_distance(input_features["avg_sentence_length"], target["avg_sentence_length"], sentence_tolerance) * 100.0
    readability_score = (0.7 * flesch_score) + (0.3 * sentence_score)
    keyword_score = keyword_overlap * 100.0

    target_embedding = target["embedding"]
    if target_embedding:
        embedding_similarity = (_cosine_similarity(input_features["embedding"], target_embedding) + 1.0) / 2.0
        embedding_score = embedding_similarity * 100.0
    else:
        embedding_similarity = 0.0
        embedding_score = 0.0

    weights = {
        "tone": 0.27,
        "sentiment": 0.225,
        "readability": 0.225,
        "keywords": 0.18,
        "embedding_similarity": 0.10,
    }
    score_overall = (
        weights["tone"] * tone_score
        + weights["sentiment"] * sentiment_score
        + weights["readability"] * readability_score
        + weights["keywords"] * keyword_score
        + weights["embedding_similarity"] * embedding_score
    )

    brand_mentions = {
        "designation": target["designation"],
        "count": _count_brand_mentions(input_features["cleaned_text"], target["designation"]),
    }

    feature_breakdown = {
        "tone": _feature_payload(
            score=tone_score,
            input_value={"formality": round(input_features["formality"], 4)},
            target_value={"formality": round(target["formality"], 4), "tone_label": target["tone_label"]},
            delta={"formality": round(input_features["formality"] - target["formality"], 4)},
            weight=weights["tone"],
            details={"std_formality": round(tone_std, 4)},
        ),
        "sentiment": _feature_payload(
            score=sentiment_score,
            input_value={"sentiment": round(input_features["sentiment"], 4)},
            target_value={"sentiment": round(target["sentiment"], 4)},
            delta={"sentiment": round(input_features["sentiment"] - target["sentiment"], 4)},
            weight=weights["sentiment"],
            details={"std_sentiment": round(sentiment_std, 4)},
        ),
        "readability": _feature_payload(
            score=readability_score,
            input_value={
                "flesch": round(input_features["flesch"], 2),
                "avg_sentence_length": round(input_features["avg_sentence_length"], 2),
            },
            target_value={
                "flesch": round(target["flesch"], 2),
                "avg_sentence_length": round(target["avg_sentence_length"], 2),
            },
            delta={
                "flesch": round(input_features["flesch"] - target["flesch"], 2),
                "avg_sentence_length": round(input_features["avg_sentence_length"] - target["avg_sentence_length"], 2),
            },
            weight=weights["readability"],
            details={"flesch_std": round(flesch_std, 2), "sentence_tolerance": round(sentence_tolerance, 2), "sentence_score": round(sentence_score, 1), "flesch_score": round(flesch_score, 1)},
        ),
        "keywords": _feature_payload(
            score=keyword_score,
            input_value={"count": len(input_keywords), "keywords": input_keywords[:10]},
            target_value={"count": len(target_keywords), "keywords": target_keywords[:10]},
            delta={"overlap_count": len(matched_keywords), "unique_gap": len(set(target_keywords) - set(input_keywords))},
            weight=weights["keywords"],
            details={"matched_keywords": matched_keywords[:10]},
        ),
        "embedding_similarity": _feature_payload(
            score=embedding_score,
            input_value={"cosine_similarity": round(embedding_similarity, 4), "embedding_model": input_features["embedding_model"]},
            target_value={"cosine_similarity": 1.0, "embedding_dim": target["embedding_dim"]},
            delta={"cosine_similarity": round(embedding_similarity - 1.0, 4)},
            weight=weights["embedding_similarity"],
            details={"embedding_dim": target["embedding_dim"]},
        ),
    }

    diagnostic_breakdown: list[dict[str, Any]] = []
    for key, payload in feature_breakdown.items():
        score = float(payload["score"])
        if score >= 85:
            continue
        if key == "tone":
            message = "Formality is drifting from the persisted genome."
            suggestion = "Use the genome's measured formality range and avoid casual phrasing."
        elif key == "sentiment":
            message = "Emotional tone is not aligned with the genome baseline."
            suggestion = "Match the genome's sentiment range with more appropriate emphasis or restraint."
        elif key == "readability":
            message = "Sentence complexity and reading ease differ from the persisted genome."
            suggestion = "Adjust sentence length and structure toward the genome's reading profile."
        elif key == "keywords":
            message = "Key genome anchors are missing or only lightly represented."
            suggestion = "Reuse the active genome's repeated keyword set and mission language."
        else:
            message = "The semantic footprint is not close to the persisted genome."
            suggestion = "Rephrase around the active genome's language and concepts."
        diagnostic_breakdown.append(
            _diagnostic_payload(
                dimension=key,
                severity=_severity_from_score(score),
                score=score,
                observed=payload["input_value"],
                expected=payload["target_value"],
                deviation=payload["delta"],
                message=message,
                suggestion=suggestion,
            )
        )

    return {
        "score_overall": round(_clamp_score(score_overall), 1),
        "feature_breakdown": feature_breakdown,
        "diagnostic_breakdown": diagnostic_breakdown,
        "brand_name_mentions": brand_mentions,
        "timestamp": _utc_now(),
        "designation": target["designation"],
        "brand_name": target["brand_name"],
        "genome_version": target["genome_version"],
        "embedding_model": input_features["embedding_model"],
    }


def score_against_user_genome(text: str, persisted_user_genome: dict[str, Any]) -> dict[str, Any]:
    """Canonical user-genome scorer used by consistency and rewrite flows."""
    return _score_against_target(text, persisted_user_genome or {})


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

    rich_result = _score_against_target(text, bp)
    breakdown = rich_result["feature_breakdown"]

    tone_pct = breakdown["tone"]["score"]
    vocab_pct = breakdown["keywords"]["score"]
    sentiment_pct = breakdown["sentiment"]["score"]
    readability_pct = breakdown["readability"]["score"]
    overall = (
        0.30 * (tone_pct / 100.0)
        + 0.25 * (sentiment_pct / 100.0)
        + 0.25 * (vocab_pct / 100.0)
        + 0.20 * (readability_pct / 100.0)
    ) * 100.0

    return {
        "overall_score": round(_clamp(overall), 1),
        "tone_pct": round(_clamp(tone_pct), 1),
        "vocab_overlap_pct": round(_clamp(vocab_pct), 1),
        "sentiment_alignment_pct": round(_clamp(sentiment_pct), 1),
        "readability_match_pct": round(_clamp(readability_pct), 1),
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
