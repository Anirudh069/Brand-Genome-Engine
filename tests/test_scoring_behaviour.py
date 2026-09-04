# tests/test_scoring_behaviour.py
# Person C — behavioural tests.
#
# tests/test_scoring.py checks that each formula computes what the spec says it
# computes. These tests check that the SYSTEM reaches the right conclusion when
# run against the real brand database.
#
# The distinction matters: the first version of the scorer passed every unit
# test while scoring genuine Rolex copy at 27/100 and keyword spam at 82/100.
# The unit tests used a hand-written profile whose keywords and standard
# deviations bore no resemblance to the ones the builder actually produces.
# Every test below would have failed on that version.

import pytest

from src.profiles.brand_profile_builder import load_brand_profile, build_user_genome
from src.scoring.consistency_scorer import (
    score_consistency, extract_text_features, WEIGHT_PRESETS,
)
from src.scoring.diagnostics import build_diagnostics
from src.scoring.drift_report import generate_drift_report
from src.scoring.edit_plan import generate_edit_plan
from src.scoring.rewrite_scoring import (
    check_consistency, prepare_rewrite, finalise_rewrite, score_before_after,
    ERR_TEXT_TOO_SHORT, ERR_BRAND_NOT_FOUND,
)
from src.scoring import analysis_log

from tests.reference_texts import ON_BRAND, OFF_BRAND, UNRELATED, KEYWORD_SPAM, SHORT


def _score(text, profile):
    return score_consistency(extract_text_features(text), profile)


# ── The core claim: the scorer ranks real copy correctly ──────────────────────

def test_on_brand_beats_off_brand_overall(rolex):
    """Authentic brand copy must outscore casual off-brand copy overall."""
    on = _score(ON_BRAND, rolex)
    off = _score(OFF_BRAND, rolex)
    assert on.overall_score > off.overall_score, (
        f"on-brand {on.overall_score:.1f} did not beat off-brand {off.overall_score:.1f}")


def test_on_brand_beats_unrelated_text(rolex):
    """Brand copy must outscore text about an unrelated subject."""
    assert _score(ON_BRAND, rolex).overall_score > _score(UNRELATED, rolex).overall_score


def test_keyword_spam_does_not_beat_real_copy(rolex):
    """
    A bag of brand nouns must not outscore a real sentence.

    This is the regression test for the brand-name loophole: when the brand's
    own name sat in top_keywords, repeating it was the highest-scoring input
    the system accepted.
    """
    assert _score(ON_BRAND, rolex).overall_score > _score(KEYWORD_SPAM, rolex).overall_score


def test_brand_name_repetition_does_not_raise_score(rolex):
    """Adding brand-name mentions to a text must not change any score."""
    base = "The case is engineered for the depths and sealed against the sea for divers."
    padded = base + " Rolex. Rolex. Rolex. Rolex."
    a, b = _score(base, rolex), _score(padded, rolex)
    assert b.vocab_overlap_pct <= a.vocab_overlap_pct + 0.01, (
        "brand-name mentions must carry zero scoring weight")


def test_brand_copy_scores_highest_against_its_own_brand(live_db):
    """
    Rolex copy must score higher against the Rolex genome than against any
    competitor's. This is the property that makes the profiles meaningful
    rather than a generic 'good writing' detector.
    """
    own = _score(ON_BRAND, load_brand_profile("rolex", live_db)).overall_score
    for other in ("omega", "tissot", "tag_heuer", "cartier"):
        rival = _score(ON_BRAND, load_brand_profile(other, live_db)).overall_score
        assert own > rival, f"rolex copy scored {rival:.1f} on {other} vs {own:.1f} on rolex"


# ── Metric independence ───────────────────────────────────────────────────────

def test_tone_is_not_a_copy_of_sentiment(rolex):
    """
    Tone and sentiment must be independent signals.

    They were not: the tone fallback was `1 - |sentiment - brand_mean|`, which
    put 55% of the overall score on a single lexicon count. If the two metrics
    ever return the same value across a varied sample, they have collapsed
    into one again.
    """
    texts = [ON_BRAND, OFF_BRAND, UNRELATED, KEYWORD_SPAM]
    pairs = [(_score(t, rolex).tone_pct, _score(t, rolex).sentiment_alignment_pct)
             for t in texts]
    differences = [abs(a - b) for a, b in pairs]
    assert max(differences) > 10.0, (
        "tone and sentiment track each other too closely — check the tone fallback")


def test_unrelated_text_scores_low_on_vocabulary_and_tone(rolex):
    """Content-sensitive metrics must reject unrelated text even if its
    readability happens to land near the brand mean."""
    result = _score(UNRELATED, rolex)
    assert result.vocab_overlap_pct < 20.0
    assert result.tone_pct < 50.0


# ── Diagnostics ───────────────────────────────────────────────────────────────

def test_off_brand_terms_are_actually_foreign_to_the_brand(rolex, live_db):
    """
    Words the brand uses constantly must never be reported as off-brand — they
    would end up in the edit plan's avoid list, telling the LLM to stop using
    the brand's own vocabulary.
    """
    diagnostics = build_diagnostics(OFF_BRAND, rolex, db_path=live_db)
    known = set(rolex.get("common_vocab") or [])
    assert diagnostics.off_brand_terms, "expected some off-brand terms in casual copy"
    assert not (set(diagnostics.off_brand_terms) & known)


def test_name_mentions_are_counted_but_neutral(rolex, live_db):
    text = OFF_BRAND + " A Rolex is a Rolex."
    diagnostics = build_diagnostics(text, rolex, db_path=live_db)
    assert diagnostics.name_mentions >= 2
    assert "zero scoring weight" in diagnostics.name_mention_note


def test_aligned_terms_found_in_on_brand_copy(rolex, live_db):
    diagnostics = build_diagnostics(ON_BRAND, rolex, db_path=live_db)
    assert len(diagnostics.aligned_terms) >= 2


# ── Drift report ──────────────────────────────────────────────────────────────

def test_casual_copy_flags_tone_too_casual(rolex, live_db):
    report = generate_drift_report({"text": OFF_BRAND}, rolex, db_path=live_db)
    assert "tone_too_casual" in report.drift_flags
    assert report.summary


def test_drift_deltas_have_the_documented_sign(rolex, live_db):
    """readability_delta = input - brand_mean. Simple copy reads easier, so the
    delta must be positive."""
    report = generate_drift_report({"text": OFF_BRAND}, rolex, db_path=live_db)
    assert report.readability_delta > 0


def test_on_brand_copy_raises_few_flags(rolex, live_db):
    report = generate_drift_report({"text": ON_BRAND}, rolex, db_path=live_db)
    assert "tone_too_casual" not in report.drift_flags
    assert "missing_brand_keywords" not in report.drift_flags


# ── Edit plan ─────────────────────────────────────────────────────────────────

def test_edit_plan_is_actionable(rolex, live_db):
    report = generate_drift_report({"text": OFF_BRAND}, rolex, db_path=live_db)
    plan = generate_edit_plan(report, rolex, OFF_BRAND, db_path=live_db)
    assert plan.goals
    assert plan.avoid_terms
    assert plan.prefer_terms
    assert plan.style_rules
    assert plan.tone_direction
    assert not (set(plan.avoid_terms) & set(plan.prefer_terms)), (
        "a term must never be both preferred and avoided")


def test_edit_plan_retrieves_grounding_chunks(rolex, live_db):
    report = generate_drift_report({"text": OFF_BRAND}, rolex, db_path=live_db)
    plan = generate_edit_plan(report, rolex, OFF_BRAND, db_path=live_db)
    assert len(plan.grounding_chunks) == 3
    assert all(len(c) > 60 for c in plan.grounding_chunks)


def test_prompt_contains_the_original_text_and_the_rules(rolex, live_db):
    report = generate_drift_report({"text": OFF_BRAND}, rolex, db_path=live_db)
    plan = generate_edit_plan(report, rolex, OFF_BRAND, db_path=live_db)
    prompt = plan.to_prompt(OFF_BRAND, "Rolex")
    assert OFF_BRAND in prompt
    assert "Rolex" in prompt
    assert "Avoid these words" in prompt


def test_grounding_chunks_are_distinct(rolex, live_db):
    report = generate_drift_report({"text": ON_BRAND}, rolex, db_path=live_db)
    plan = generate_edit_plan(report, rolex, ON_BRAND, db_path=live_db)
    assert len(set(plan.grounding_chunks)) == len(plan.grounding_chunks)


# ── Before / after ────────────────────────────────────────────────────────────

def test_before_after_uses_one_scorer(rolex):
    """Scoring the same text twice through the before/after path must give
    identical numbers — otherwise before and after are not comparable."""
    before, after = score_before_after(ON_BRAND, ON_BRAND, rolex)
    assert before.to_dict() == after.to_dict()


def test_a_genuine_improvement_raises_the_score(rolex):
    """The Definition of Done requires score_after > score_before for a known
    off-brand input rewritten into brand voice."""
    rewritten = (
        "The Oyster Perpetual is conceived for daily wear yet engineered without "
        "compromise: a waterproof case sealed against the elements, a movement "
        "certified for precision, and a design essentially unchanged for decades."
    )
    before, after = score_before_after(OFF_BRAND, rewritten, rolex)
    assert after.overall_score > before.overall_score


# ── API-shaped entry points ───────────────────────────────────────────────────

def test_check_consistency_returns_the_frozen_shape(live_db):
    result = check_consistency(OFF_BRAND, "rolex", live_db)
    for key in ("brand_id", "brand_name", "overall_score", "tone_pct",
                "vocab_overlap_pct", "sentiment_alignment_pct",
                "readability_match_pct", "error"):
        assert key in result
    assert result["error"] is None


def test_short_text_returns_a_graceful_error(live_db):
    result = check_consistency(SHORT, "rolex", live_db)
    assert result["error"] == ERR_TEXT_TOO_SHORT
    assert result["overall_score"] is None


def test_unknown_brand_returns_a_graceful_error(live_db):
    assert check_consistency(OFF_BRAND, "not_a_brand", live_db)["error"] == ERR_BRAND_NOT_FOUND


def test_rewrite_response_has_every_frozen_field(live_db):
    prep = prepare_rewrite(OFF_BRAND, "rolex", db_path=live_db)
    result = finalise_rewrite(prep, "A rewritten sentence about a sealed waterproof case.",
                              db_path=live_db, log=False)
    for key in ("brand_id", "brand_name", "original_text", "rewritten_text",
                "suggestions", "grounding_chunks_used", "score_before",
                "score_after", "error"):
        assert key in result
    assert 3 <= len(result["suggestions"]) <= 5


# ── Logging contract ──────────────────────────────────────────────────────────

def test_log_payload_always_has_the_same_keys(rolex, live_db):
    payload = analysis_log.build_log_payload(
        "rolex", analysis_log.EVENT_CONSISTENCY, score_before=_score(ON_BRAND, rolex))
    for key in ("schema_version", "event_type", "brand_id", "scores",
                "diagnostics", "drift_report", "edit_plan", "extra", "logged_at"):
        assert key in payload
    assert payload["scores"]["after"] is None


def test_invalid_event_type_is_rejected():
    with pytest.raises(ValueError):
        analysis_log.build_log_payload("rolex", "not_an_event")


def test_counters_move_when_runs_are_logged(live_db):
    before = analysis_log.get_counters(live_db)
    check_consistency(OFF_BRAND, "rolex", live_db)
    after = analysis_log.get_counters(live_db)
    assert after["copies_analysed"] == before["copies_analysed"] + 1


def test_logging_survives_a_broken_database():
    """A failure in analytics must never break the request that triggered it."""
    assert analysis_log.log_analysis("rolex", analysis_log.EVENT_CONSISTENCY,
                                     db_path="/nonexistent/path/db.sqlite") == -1


# ── Weight presets (Phase 4A deliverable 2) ───────────────────────────────────

def test_presets_change_only_the_overall_score(rolex):
    """
    A preset re-weights the blend. It must never change what the individual
    metrics measured — otherwise the breakdown shown in the UI would depend on
    a dropdown, and two users would see different sub-scores for one text.
    """
    features = extract_text_features(ON_BRAND)
    results = {name: score_consistency(features, rolex, preset=name)
               for name in WEIGHT_PRESETS}
    subs = {
        (round(r.tone_pct, 6), round(r.vocab_overlap_pct, 6),
         round(r.sentiment_alignment_pct, 6), round(r.readability_match_pct, 6))
        for r in results.values()
    }
    assert len(subs) == 1, "sub-scores must be identical across presets"
    assert len({round(r.overall_score, 4) for r in results.values()}) > 1


def test_every_preset_sums_to_one():
    for name, weights in WEIGHT_PRESETS.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9, f"{name} does not sum to 1"


def test_semantic_heavy_rewards_vocabulary(rolex):
    """A text strong on vocabulary should score higher under semantic_heavy
    than under tone_heavy — that is what choosing the preset is for."""
    features = extract_text_features(ON_BRAND)
    semantic = score_consistency(features, rolex, preset="semantic_heavy")
    tonal = score_consistency(features, rolex, preset="tone_heavy")
    assert semantic.vocab_overlap_pct > semantic.tone_pct
    assert semantic.overall_score > tonal.overall_score


def test_unknown_preset_falls_back_to_balanced(rolex):
    """A bad preset name from the UI must not take the endpoint down."""
    features = extract_text_features(ON_BRAND)
    fallback = score_consistency(features, rolex, preset="not_a_preset")
    balanced = score_consistency(features, rolex, preset="balanced")
    assert fallback.overall_score == balanced.overall_score


def test_custom_weights_are_normalised(rolex):
    """Weights that do not sum to 1 must not inflate the score."""
    features = extract_text_features(ON_BRAND)
    result = score_consistency(
        features, rolex,
        preset={"tone": 2, "sentiment": 2, "vocab": 2, "readability": 2})
    assert 0.0 <= result.overall_score <= 100.0


def test_profile_stored_preset_is_used(rolex):
    """A preset saved on the genome applies without the caller passing it."""
    profile = dict(rolex)
    profile["weight_preset"] = "semantic_heavy"
    features = extract_text_features(ON_BRAND)
    assert (score_consistency(features, profile).overall_score
            == score_consistency(features, rolex, preset="semantic_heavy").overall_score)


# ── User brand genome (Phase 4A deliverable 1) ────────────────────────────────

_USER_MISSION = (
    "We build instruments for people who measure their lives in decades. Every "
    "component is engineered to outlast its owner, and every calibre is tested "
    "before it leaves the workshop."
)

_USER_SNIPPETS = [
    "The movement is assembled by hand and adjusted across five positions before it leaves the workshop.",
    "Our archive holds every drawing since 1946, and each restoration begins in the archive.",
    "A case is not finished until it has survived three hundred hours of testing.",
    "We do not release a calibre until that calibre has run for a year without adjustment.",
    "Patience is a material. We treat patience as one.",
    "The bracelet is milled from a single billet, because a join is a point of failure.",
    "Our guarantee is measured in generations. The movement outlives the owner.",
]


def _user_genome(db, **kwargs):
    return build_user_genome(
        "Meridian Instruments", _USER_MISSION, _USER_SNIPPETS,
        tone_label="Formal", db_path=db, persist=False, **kwargs)


def test_user_genome_keywords_come_from_the_text(live_db):
    """
    The genome must be measured, not assembled from constants. The previous
    implementation hardcoded ["precision", "legacy", "craftsmanship"], so the
    Consistency Check page scored against a placeholder.
    """
    genome = _user_genome(live_db)
    assert genome["top_keywords"], "no keywords extracted"
    assert set(genome["top_keywords"]) != {"precision", "legacy", "craftsmanship"}
    corpus = " ".join([_USER_MISSION] + _USER_SNIPPETS).lower()
    for word in genome["top_keywords"]:
        assert word in corpus, f"'{word}' is not in the submitted text"


def test_user_genome_formality_is_measured_not_a_dropdown(live_db):
    """Formality must come from the writing, not from the declared tone."""
    genome = _user_genome(live_db)
    assert genome["declared_tone"] == "Formal"
    assert genome["mean_formality"] not in (0.4, 0.7), "looks like the old constant"
    assert 0.0 <= genome["mean_formality"] <= 1.0


def test_user_genome_has_every_field_the_scorer_needs(live_db):
    genome = _user_genome(live_db)
    for field in ("mean_sentiment", "std_sentiment", "mean_flesch", "std_flesch",
                  "mean_formality", "std_formality", "mean_vocab_richness",
                  "std_vocab_richness", "mean_sentence_length",
                  "std_sentence_length", "top_keywords", "brand_name_tokens",
                  "sentiment_scale"):
        assert field in genome, f"missing {field}"


def test_user_genome_std_floors_applied(live_db):
    """Eight short texts cannot support a narrow spread — the floors prevent a
    genome that rejects perfectly good copy."""
    from src.profiles.brand_profile_builder import USER_STD_FLOORS
    genome = _user_genome(live_db)
    assert genome["std_sentiment"] >= USER_STD_FLOORS["sentiment"]
    assert genome["std_flesch"] >= USER_STD_FLOORS["flesch"]
    assert genome["std_formality"] >= USER_STD_FLOORS["formality"]


def test_user_genome_separates_on_brand_from_off_brand(live_db):
    genome = _user_genome(live_db)
    on = ("Each calibre is adjusted by hand and tested for three hundred hours "
          "before it leaves the workshop, because a join is a point of failure.")
    on_score = score_consistency(extract_text_features(on), genome).overall_score
    off_score = score_consistency(extract_text_features(OFF_BRAND), genome).overall_score
    assert on_score > off_score + 15, (
        f"user genome barely separates copy: {on_score:.1f} vs {off_score:.1f}")


def test_user_genome_rejects_too_little_text(live_db):
    with pytest.raises(ValueError):
        build_user_genome("X", "Short mission.", [], db_path=live_db, persist=False)


def test_user_genome_persists_and_reloads(live_db):
    from src.profiles.brand_profile_builder import USER_BRAND_ID
    build_user_genome("Meridian Instruments", _USER_MISSION, _USER_SNIPPETS,
                      db_path=live_db, persist=True)
    reloaded = load_brand_profile(USER_BRAND_ID, live_db)
    assert reloaded["brand_name"] == "Meridian Instruments"
    assert reloaded["snippetsCount"] == 7


def test_user_genome_carries_the_weight_preset(live_db):
    genome = _user_genome(live_db, weight_preset="semantic_heavy")
    assert genome["weight_preset"] == "semantic_heavy"
