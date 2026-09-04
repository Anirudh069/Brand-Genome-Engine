"""
consistency.py — canonical brand-consistency scorer (Person C).

PUBLIC API — unchanged from the previous version. ``src/api/main.py`` imports
these two functions and nothing else, so the API and the frontend need no
modification:

    compute_consistency_score(text: str, brand_profile: dict) -> dict
    generate_edit_plan(text: str, brand_profile: dict) -> dict

Two functions are added, both optional for existing callers:

    generate_drift_report_dict(text, brand_profile) -> dict
    score_before_after_dict(original, rewritten, brand_profile) -> dict

WHAT CHANGED AND WHY
--------------------
The previous implementation scored authentic brand copy at 51/100 and casual
off-brand copy at 45/100 — a six-point separation. Measured against the real
Rolex profile, an unrelated sentence about pizza delivery also scored 45. The
same paragraph scored 51 against Rolex and 47 against Tissot, so the scorer
could not reliably tell two brands apart.

Four causes, all fixed in the modules this file now delegates to:

1. Brand vocabulary was chosen by raw frequency, which returns the nouns every
   watch brand shares — for Rolex, ``rolex, watch, case, oyster, time``. It is
   now chosen by TF-IDF across brands, blended with frequency, with proper
   nouns and specification vocabulary excluded by data-driven rules.

2. The brand's own name was the top-ranked keyword, so repeating it was the
   cheapest route to a high score. Name tokens are now excluded from the
   vocabulary at build time and stripped before scoring; mentions are counted
   and reported as a neutral signal, as the Phase 4A brief requires.

3. Sentiment was effectively ternary, producing a standard deviation wide
   enough that unrelated text still scored in the seventies. It is now a graded
   density measure, and the lexicon holds emotional words only — "precision"
   and "heritage" were being counted as both sentiment and vocabulary.

4. Tone, the heaviest-weighted metric, was a restatement of sentiment. It now
   measures formality, lexical variety and sentence length, none of which
   duplicate another metric.

Full derivations, weights and edge cases: docs/scoring_spec_v2.md

The output schema is unchanged and remains frozen.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.profiles.brand_profile_builder import (
    load_brand_profile,
    list_brands,
    is_genome_initialised,
    build_user_genome,
    BrandProfileNotFoundError,
)
from src.scoring.consistency_scorer import (
    ScoreResult,
    score_consistency,
    extract_text_features,
    resolve_weights,
    WEIGHT_PRESETS,
    EmbeddingDimensionError,
    FeatureExtractionError,
    MIN_WORDS,
)
from src.scoring.diagnostics import build_diagnostics
from src.scoring.drift_report import generate_drift_report
from src.scoring.edit_plan import generate_edit_plan as _generate_edit_plan

logger = logging.getLogger(__name__)

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/brand_data.db")

__all__ = [
    "compute_consistency_score",
    "generate_edit_plan",
    "generate_drift_report_dict",
    "score_before_after_dict",
    "build_diagnostics_dict",
    "ScoreResult",
    "BrandProfileNotFoundError",
    "EmbeddingDimensionError",
    "FeatureExtractionError",
    "MIN_WORDS",
    "WEIGHT_PRESETS",
    "resolve_weights",
    "build_user_genome",
    "load_brand_profile",
    "list_brands",
    "is_genome_initialised",
]

_ZERO_SCORE = {
    "overall_score": 0.0,
    "tone_pct": 0.0,
    "vocab_overlap_pct": 0.0,
    "sentiment_alignment_pct": 0.0,
    "readability_match_pct": 0.0,
}


# ── Public API ────────────────────────────────────────────────────────────────

def compute_consistency_score(text: str, brand_profile: dict,
                              preset: str | dict | None = None) -> dict[str, float]:
    """
    Score `text` against `brand_profile`.

    Returns exactly these keys, all floats clamped to [0, 100]:

        overall_score, tone_pct, vocab_overlap_pct,
        sentiment_alignment_pct, readability_match_pct

    `preset` selects a weight profile — "balanced", "tone_heavy" or
    "semantic_heavy" — or accepts a custom weight dict. Omit it and the
    genome's own stored preference is used, falling back to balanced. The
    sub-scores are identical across presets; only overall_score changes.

    Never raises: a malformed profile or feature returns the all-zero result and
    logs, because the API must answer rather than 500 on bad input.
    """
    if not brand_profile:
        logger.warning("compute_consistency_score called with an empty profile")
        return dict(_ZERO_SCORE)

    try:
        features = extract_text_features(text or "")
        result = score_consistency(features, brand_profile, preset=preset)
    except (FeatureExtractionError, EmbeddingDimensionError) as exc:
        logger.warning("Scoring failed for brand %s: %s",
                       brand_profile.get("brand_id"), exc)
        return dict(_ZERO_SCORE)

    return {
        "overall_score": round(result.overall_score, 1),
        "tone_pct": round(result.tone_pct, 1),
        "vocab_overlap_pct": round(result.vocab_overlap_pct, 1),
        "sentiment_alignment_pct": round(result.sentiment_alignment_pct, 1),
        "readability_match_pct": round(result.readability_match_pct, 1),
    }


def generate_edit_plan(text: str, brand_profile: dict,
                       db_path: str | None = None,
                       retriever=None) -> dict[str, Any]:
    """
    Build the rewrite instruction set for `text` against `brand_profile`.

    Returns the frozen EditPlan shape:

        brand_id, goals, avoid_terms, prefer_terms,
        style_rules, tone_direction, grounding_chunks

    plus `prompt` — the fully rendered LLM prompt. Callers that build their own
    prompt can ignore it; using it keeps prompt wording next to the logic that
    decided what belongs in it.

    `retriever` accepts Person B's FAISS retrieval function with the contract
    retriever(query_text, brand_id, k) -> list[str]. Without it, grounding
    chunks are retrieved from brand_chunks by lexical ranking.
    """
    db_path = db_path or SQLITE_DB_PATH

    if not brand_profile:
        return {
            "brand_id": None, "goals": [], "avoid_terms": [], "prefer_terms": [],
            "style_rules": [], "tone_direction": "", "grounding_chunks": [],
            "prompt": None,
        }

    text = text or ""
    features = extract_text_features(text)

    try:
        score = score_consistency(features, brand_profile)
    except (FeatureExtractionError, EmbeddingDimensionError):
        score = None

    diagnostics = build_diagnostics(text, brand_profile, db_path=db_path)
    drift = generate_drift_report(features, brand_profile, diagnostics, db_path=db_path)
    plan = _generate_edit_plan(drift, brand_profile, text, score,
                               db_path=db_path, retriever=retriever)

    payload = plan.to_dict()
    payload["prompt"] = plan.to_prompt(text, brand_profile.get("brand_name"))
    return payload


def generate_drift_report_dict(text: str, brand_profile: dict,
                               db_path: str | None = None) -> dict[str, Any]:
    """
    Structured explanation of how `text` departs from the brand voice.

        brand_id, drift_flags, sentiment_delta, readability_delta,
        missing_keywords, excess_keywords, summary

    New in Phase 4B — no drift report existed previously. `summary` is written
    as plain English suitable for both the UI and the LLM prompt.
    """
    db_path = db_path or SQLITE_DB_PATH
    if not brand_profile:
        return {"brand_id": None, "drift_flags": [], "sentiment_delta": 0.0,
                "readability_delta": 0.0, "missing_keywords": [],
                "excess_keywords": [], "summary": ""}
    features = extract_text_features(text or "")
    return generate_drift_report(features, brand_profile, db_path=db_path).to_dict()


def build_diagnostics_dict(text: str, brand_profile: dict,
                           db_path: str | None = None) -> dict[str, Any]:
    """
    Word-level diagnostics: aligned terms, missing terms, off-brand terms,
    pillar coverage, and the neutral brand-name mention count.
    """
    db_path = db_path or SQLITE_DB_PATH
    if not brand_profile:
        return {}
    return build_diagnostics(text or "", brand_profile, db_path=db_path).to_dict()


def score_before_after_dict(original_text: str, rewritten_text: str,
                            brand_profile: dict) -> dict[str, Any]:
    """
    Score both versions of a rewrite with the identical function.

    Returns {"before": {...}, "after": {...}}. Using one scorer for both halves
    is what makes the two sets of numbers comparable at all.
    """
    return {
        "before": compute_consistency_score(original_text, brand_profile),
        "after": compute_consistency_score(rewritten_text, brand_profile),
    }
