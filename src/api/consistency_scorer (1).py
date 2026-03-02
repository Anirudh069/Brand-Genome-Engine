# src/scoring/consistency_scorer.py
# Person C — Consistency Scorer
# Compares a new text's features to a brand profile and returns 5 scores.
# Field names are FROZEN — Person D's API depends on them.

import re
import math
from dataclasses import dataclass


# ── Frozen output contract (do NOT rename fields) ─────────────────────────────

@dataclass
class ScoreResult:
    overall_score: float           # 0–100
    tone_pct: float                # 0–100
    vocab_overlap_pct: float       # 0–100
    sentiment_alignment_pct: float # 0–100
    readability_match_pct: float   # 0–100


# ── Custom exceptions ─────────────────────────────────────────────────────────

class BrandProfileNotFoundError(Exception):
    pass

class EmbeddingDimensionError(Exception):
    pass

class FeatureExtractionError(Exception):
    pass


# ── Internal helpers ──────────────────────────────────────────────────────────

WORD_RE = re.compile(r"[a-zA-Z']+")
STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","at",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "by","from","you","we","they","he","she","i","our","your","their","its","not",
    "have","has","had","do","does","did","will","would","could","should","may",
    "can","all","been","more","also","than","into","which","about",
}


def _tokenize(text: str) -> list:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _content_words(text: str) -> list:
    return [w for w in _tokenize(text) if w not in STOPWORDS and len(w) >= 3]


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _jaccard(set_a: list, set_b: list) -> float:
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|
    Edge case: both empty → 0 (not 100).
    """
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _gaussian_similarity(value: float, mean: float, std: float) -> float:
    """
    exp(−((value − mean)² / (2 × std²)))
    Edge case: std < 0.01 → clamp to 0.01 to avoid division collapse.
    """
    std = max(std, 0.01)
    return math.exp(-((value - mean) ** 2) / (2 * std ** 2))


def _inverse_distance(value: float, mean: float, tolerance: float = 20.0) -> float:
    """
    max(0, 1 − |value − mean| / tolerance)
    Edge case: tolerance ≤ 0 → use 20.
    """
    tolerance = max(tolerance, 20.0)
    return max(0.0, 1.0 - abs(value - mean) / tolerance)


def _cosine_similarity(a: list, b: list) -> float:
    """
    Cosine similarity between two vectors.
    Edge case: zero vector → return 0.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise EmbeddingDimensionError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}"
        )
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _flesch_score(text: str) -> float:
    """Approximate Flesch Reading Ease (no external libs needed)."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = _tokenize(text)
    if not sentences or not words:
        return 50.0
    def syllables(word):
        return max(1, len(re.findall(r"[aeiou]+", word.lower())))
    total_syllables = sum(syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = total_syllables / len(words)
    return 206.835 - 1.015 * asl - 84.6 * asw


def _sentiment_proxy(text: str) -> float:
    """Rule-based sentiment proxy → range [-1, 1]."""
    positive = {
        "excellence","achieve","achievement","exceptional","extraordinary","remarkable",
        "enduring","precision","innovative","iconic","perfect","ultimate","heritage",
        "trust","superior","finest","premium","outstanding","dedicated","passion",
        "inspire","lead","legendary","timeless","craftsmanship","proud","champion",
        "victory","best","beautiful","elegant","luxury","prestigious",
    }
    negative = {
        "fail","failure","poor","cheap","bad","terrible","awful","inferior","weak",
        "broken","wrong","defect","problem","issue","concern","risk","loss","reject",
    }
    words = set(_tokenize(text))
    pos = len(words & positive)
    neg = len(words & negative)
    total = pos + neg
    if total == 0:
        return 0.1
    return (pos - neg) / total


# ── Main scoring function (API contract — field names frozen) ─────────────────

def score_consistency(text_features: dict, brand_profile: dict) -> ScoreResult:
    """
    Compare a new text's features against a brand profile.

    Parameters
    ----------
    text_features : dict
        Must contain at least: "text" (raw string).
        Optionally pre-computed: "sentiment_score", "flesch_reading_ease",
        "top_keywords", "embedding".
    brand_profile : dict
        Parsed profile_json from brand_profiles table.

    Returns
    -------
    ScoreResult with five float fields in range [0, 100].

    Raises
    ------
    FeatureExtractionError  – if a required feature contains NaN.
    BrandProfileNotFoundError – caller must catch if profile is missing.
    EmbeddingDimensionError – if embedding lengths differ.
    """
    # ── Guard: short-text edge case ───────────────────────────────────────────
    text = text_features.get("text", "") or ""
    words = _tokenize(text)
    if len(words) < 10:
        return ScoreResult(
            overall_score=0.0,
            tone_pct=0.0,
            vocab_overlap_pct=0.0,
            sentiment_alignment_pct=0.0,
            readability_match_pct=0.0,
        )

    # ── Extract / compute features ─────────────────────────────────────────────
    # Use pre-computed values if Person B provides them; else compute proxies.
    sentiment = text_features.get("sentiment_score")
    if sentiment is None:
        sentiment = _sentiment_proxy(text)

    flesch = text_features.get("flesch_reading_ease")
    if flesch is None:
        flesch = _flesch_score(text)

    text_keywords = text_features.get("top_keywords")
    if text_keywords is None:
        text_keywords = _content_words(text)

    embedding = text_features.get("embedding", [])

    # NaN guard
    for field_name, val in [("sentiment", sentiment), ("flesch", flesch)]:
        try:
            if math.isnan(float(val)):
                raise FeatureExtractionError(f"NaN in feature field: {field_name}")
        except (TypeError, ValueError):
            raise FeatureExtractionError(f"Non-numeric value in field: {field_name}")

    # ── Retrieve brand stats ───────────────────────────────────────────────────
    mean_sentiment   = float(brand_profile.get("mean_sentiment", 0.1))
    std_sentiment    = float(brand_profile.get("std_sentiment", 0.01))
    mean_flesch      = float(brand_profile.get("mean_flesch", 50.0))
    std_flesch       = float(brand_profile.get("std_flesch", 5.0))
    brand_keywords   = brand_profile.get("top_keywords", []) or []
    brand_embedding  = brand_profile.get("mean_embedding", []) or []

    # ── 1) Vocabulary Overlap (Jaccard) ────────────────────────────────────────
    vocab_overlap = _jaccard(text_keywords, brand_keywords)      # 0–1

    # ── 2) Sentiment Alignment (Gaussian) ──────────────────────────────────────
    sentiment_align = _gaussian_similarity(sentiment, mean_sentiment, std_sentiment)  # 0–1

    # ── 3) Readability Match (Inverse Distance) ────────────────────────────────
    tolerance = max(2 * std_flesch, 20.0)
    readability = _inverse_distance(flesch, mean_flesch, tolerance)   # 0–1

    # ── 4) Tone (Cosine Similarity of embeddings) ──────────────────────────────
    if brand_embedding and embedding:
        tone = max(0.0, _cosine_similarity(embedding, brand_embedding))   # 0–1
    else:
        # Fallback when embeddings not yet wired: use formality proxy comparison
        tone = 1.0 - abs(sentiment - mean_sentiment)  # crude stand-in
        tone = max(0.0, tone)

    # ── Overall (weighted average) ─────────────────────────────────────────────
    # Weights from spec:  Tone 0.30 | Sentiment 0.25 | Vocab 0.25 | Readability 0.20
    overall = (
        0.30 * tone
        + 0.25 * sentiment_align
        + 0.25 * vocab_overlap
        + 0.20 * readability
    ) * 100.0

    return ScoreResult(
        overall_score          = _clamp(overall),
        tone_pct               = _clamp(tone               * 100.0),
        vocab_overlap_pct      = _clamp(vocab_overlap      * 100.0),
        sentiment_alignment_pct= _clamp(sentiment_align    * 100.0),
        readability_match_pct  = _clamp(readability        * 100.0),
    )
