# src/scoring/rewrite_scoring.py
# Person C — Phase 4B deliverable 7: before/after scoring
#
# The orchestration layer Person D's endpoints call. Everything here is pure
# Python with no web framework and no LLM client: this module decides WHAT to
# say to the model and HOW to judge the answer; the endpoint owns the network
# call in between.
#
# Typical use inside POST /api/rewrite:
#
#     prep = prepare_rewrite(text, brand_id)
#     if prep["error"]:
#         return prep                      # short text / genome not initialised
#     rewritten = call_llm(prep["prompt"])         # Person D
#     result = finalise_rewrite(prep, rewritten)   # scores + logs + returns
#
# And inside POST /api/check-consistency:
#
#     result = check_consistency(text, brand_id)

from src.profiles.brand_profile_builder import (
    load_brand_profile,
    is_genome_initialised,
    BrandProfileNotFoundError,
    SQLITE_DB_PATH,
)
from src.scoring.consistency_scorer import (
    score_consistency,
    extract_text_features,
    MIN_WORDS,
)
from src.scoring.diagnostics import build_diagnostics
from src.scoring.drift_report import generate_drift_report
from src.scoring.edit_plan import generate_edit_plan
from src.scoring import analysis_log

ERR_TEXT_TOO_SHORT = "text_too_short"
ERR_GENOME_NOT_INITIALISED = "genome_not_initialised"
ERR_BRAND_NOT_FOUND = "brand_not_found"


def _word_count(text):
    return len((text or "").split())


def _guard(text, brand_id, db_path):
    """
    Shared precondition check.
    Returns (profile, error_string). Exactly one of the two is None.
    """
    if _word_count(text) < MIN_WORDS:
        return None, ERR_TEXT_TOO_SHORT

    try:
        profile = load_brand_profile(brand_id, db_path)
    except BrandProfileNotFoundError:
        return None, ERR_BRAND_NOT_FOUND

    if not is_genome_initialised(brand_id, db_path):
        return None, ERR_GENOME_NOT_INITIALISED

    return profile, None


# ── Consistency check (Phase 4A) ──────────────────────────────────────────────

def check_consistency(text, brand_id, db_path=SQLITE_DB_PATH, log=True):
    """
    Score one text and explain the result. Backs POST /api/check-consistency.

    Returns a dict matching the frozen response shape: the five score fields at
    the top level, plus diagnostics, plus `error` which is None on success.
    """
    profile, error = _guard(text, brand_id, db_path)
    if error:
        return {
            "brand_id": brand_id,
            "brand_name": None,
            "overall_score": None,
            "tone_pct": None,
            "vocab_overlap_pct": None,
            "sentiment_alignment_pct": None,
            "readability_match_pct": None,
            "diagnostics": None,
            "error": error,
        }

    features = extract_text_features(text)
    score = score_consistency(features, profile)
    diagnostics = build_diagnostics(text, profile, db_path=db_path)

    if log:
        analysis_log.log_analysis(
            brand_id, analysis_log.EVENT_CONSISTENCY,
            input_text=text, score_before=score,
            diagnostics=diagnostics, db_path=db_path,
        )

    return {
        "brand_id": brand_id,
        "brand_name": profile.get("brand_name"),
        "overall_score": round(score.overall_score, 1),
        "tone_pct": round(score.tone_pct, 1),
        "vocab_overlap_pct": round(score.vocab_overlap_pct, 1),
        "sentiment_alignment_pct": round(score.sentiment_alignment_pct, 1),
        "readability_match_pct": round(score.readability_match_pct, 1),
        "diagnostics": diagnostics.to_dict(),
        "error": None,
    }


# ── Rewrite: stage 1, everything before the LLM call ──────────────────────────

def prepare_rewrite(text, brand_id, k_grounding=3,
                    db_path=SQLITE_DB_PATH, retriever=None):
    """
    Score the input, diagnose it, and build the LLM prompt.

    The returned dict is passed straight back into finalise_rewrite() so that
    nothing is recomputed and the "before" score the user sees is provably the
    same object the "after" score is compared against.
    """
    profile, error = _guard(text, brand_id, db_path)
    if error:
        return {"error": error, "brand_id": brand_id, "original_text": text,
                "prompt": None, "score_before": None, "drift_report": None,
                "edit_plan": None, "profile": None}

    features = extract_text_features(text)
    score_before = score_consistency(features, profile)
    diagnostics = build_diagnostics(text, profile, db_path=db_path)
    drift = generate_drift_report(features, profile, diagnostics, db_path=db_path)
    plan = generate_edit_plan(drift, profile, text, score_before,
                              k_grounding=k_grounding, db_path=db_path,
                              retriever=retriever)

    return {
        "error": None,
        "brand_id": brand_id,
        "brand_name": profile.get("brand_name"),
        "original_text": text,
        "prompt": plan.to_prompt(text, profile.get("brand_name")),
        "score_before": score_before,
        "diagnostics": diagnostics,
        "drift_report": drift,
        "edit_plan": plan,
        "profile": profile,
    }


# ── Rewrite: stage 2, everything after the LLM call ───────────────────────────

def score_before_after(original_text, rewritten_text, brand_profile):
    """
    Score two versions of the same copy with the identical function.

    Using one scorer for both halves is the whole point: the before and after
    numbers are only comparable because nothing about the measurement changed
    between them.
    """
    before = score_consistency(extract_text_features(original_text), brand_profile)
    after = score_consistency(extract_text_features(rewritten_text), brand_profile)
    return before, after


def finalise_rewrite(prep, rewritten_text, llm_error=None,
                     db_path=SQLITE_DB_PATH, log=True):
    """
    Score the LLM's output, log the run, and assemble the API response.

    Matches the frozen /api/rewrite response shape.
    """
    if prep.get("error"):
        return {
            "brand_id": prep.get("brand_id"),
            "brand_name": None,
            "original_text": prep.get("original_text"),
            "rewritten_text": None,
            "suggestions": [],
            "grounding_chunks_used": [],
            "score_before": None,
            "score_after": None,
            "drift_report": None,
            "edit_plan": None,
            "error": prep["error"],
        }

    profile = prep["profile"]
    plan = prep["edit_plan"]
    score_before = prep["score_before"]

    score_after = None
    if rewritten_text and not llm_error:
        score_after = score_consistency(
            extract_text_features(rewritten_text), profile)

    if log:
        analysis_log.log_analysis(
            prep["brand_id"], analysis_log.EVENT_REWRITE,
            input_text=prep["original_text"],
            score_before=score_before, score_after=score_after,
            diagnostics=prep.get("diagnostics"),
            drift_report=prep.get("drift_report"),
            edit_plan=plan,
            extra={"llm_error": llm_error,
                   "n_grounding_chunks": len(plan.grounding_chunks)},
            db_path=db_path,
        )

    return {
        "brand_id": prep["brand_id"],
        "brand_name": prep.get("brand_name"),
        "original_text": prep["original_text"],
        "rewritten_text": rewritten_text,
        "suggestions": suggestions_from_plan(plan, prep["drift_report"]),
        "grounding_chunks_used": list(plan.grounding_chunks),
        "score_before": score_before.to_dict() if score_before else None,
        "score_after": score_after.to_dict() if score_after else None,
        "drift_report": prep["drift_report"].to_dict(),
        "edit_plan": plan.to_dict(),
        "error": llm_error,
    }


def suggestions_from_plan(plan, drift_report, limit=5):
    """
    The 3-5 bullet points shown in the UI's suggestions panel.

    Written for a human reader — the goals and style rules the plan sends to
    the model are phrased for a model, and reading them raw in the interface
    is not useful to a writer.
    """
    out = []

    if drift_report.excess_keywords:
        out.append(
            "Replace casual or off-brand wording — "
            + ", ".join(f"'{w}'" for w in drift_report.excess_keywords[:3])
            + " — with more measured vocabulary."
        )
    if plan.prefer_terms:
        out.append(
            "Introduce brand-anchored terms such as "
            + ", ".join(f"'{w}'" for w in plan.prefer_terms[:3]) + "."
        )
    if "tone_too_casual" in drift_report.drift_flags:
        out.append("Avoid contractions and colloquialisms; keep the register formal.")
    if "tone_too_formal" in drift_report.drift_flags:
        out.append("Loosen the register slightly — the copy reads stiffer than the brand.")
    if "readability_too_high" in drift_report.drift_flags:
        out.append(
            f"Increase sentence complexity — reading ease is "
            f"{abs(drift_report.readability_delta):.0f} points above the brand average."
        )
    if "readability_too_low" in drift_report.drift_flags:
        out.append(
            f"Simplify the sentence structure — reading ease is "
            f"{abs(drift_report.readability_delta):.0f} points below the brand average."
        )
    if "sentiment_too_positive" in drift_report.drift_flags:
        out.append("Tone down superlatives; state qualities as fact rather than excitement.")
    if "missing_pillar_coverage" in drift_report.drift_flags:
        out.append("Anchor the copy to at least one core messaging pillar.")

    if not out:
        out.append("The copy already matches the brand voice — no changes required.")

    return out[:limit]
