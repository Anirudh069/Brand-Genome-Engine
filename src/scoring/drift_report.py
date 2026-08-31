# src/scoring/drift_report.py
# Person C — Phase 4B deliverable 5: drift report generator
#
# Answers "why is this off-brand, and by how much?" in a structured form.
# DriftReport field names are FROZEN — Person D's rewrite endpoint returns them
# and the UI renders them.

from dataclasses import dataclass, field, asdict

from src.profiles.brand_profile_builder import (
    flesch_score,
    sentiment_proxy,
    formality_proxy,
    mean_sentence_length,
    SQLITE_DB_PATH,
)
from src.scoring.diagnostics import build_diagnostics

# A difference is worth telling a writer about when it is both (a) large in
# the metric's own units and (b) large relative to how much the brand itself
# varies. A brand that writes very consistently should flag smaller deviations
# than one that ranges widely, but no brand should flag noise — hence the
# absolute floors below, used as:
#
#     threshold = max(absolute_floor, 0.75 * brand_std)
#
SENTIMENT_FLAG = 0.25      # points on a -1..1 scale
READABILITY_FLAG = 10.0    # Flesch points
FORMALITY_FLAG = 0.07      # points on a 0..1 scale
SENTENCE_LEN_FLAG = 5.0    # words per sentence

STD_FRACTION = 0.75


def _threshold(floor, brand_std):
    return max(floor, STD_FRACTION * abs(float(brand_std or 0.0)))


@dataclass
class DriftReport:
    """Structured explanation of how a text departs from a brand profile."""
    brand_id: str
    drift_flags: list = field(default_factory=list)
    sentiment_delta: float = 0.0      # input_sentiment - brand_mean_sentiment
    readability_delta: float = 0.0    # input_flesch - brand_mean_flesch
    missing_keywords: list = field(default_factory=list)
    excess_keywords: list = field(default_factory=list)
    summary: str = ""

    def to_dict(self):
        return asdict(self)


def _flag_list(sentiment_delta, readability_delta, formality_delta,
               length_delta, diagnostics, profile):
    """
    Machine-readable drift flags.

    Deltas are signed and the flag names say which direction the text went, so
    the edit plan can act on them without re-deriving anything.
    """
    flags = []

    t_sent = _threshold(SENTIMENT_FLAG, profile.get("std_sentiment"))
    t_read = _threshold(READABILITY_FLAG, profile.get("std_flesch"))
    t_form = _threshold(FORMALITY_FLAG, profile.get("std_formality"))
    t_len = _threshold(SENTENCE_LEN_FLAG, profile.get("std_sentence_length"))

    if sentiment_delta > t_sent:
        flags.append("sentiment_too_positive")
    elif sentiment_delta < -t_sent:
        flags.append("sentiment_too_negative")

    if readability_delta > t_read:
        flags.append("readability_too_high")     # too easy / too plain
    elif readability_delta < -t_read:
        flags.append("readability_too_low")      # too dense

    if formality_delta < -t_form:
        flags.append("tone_too_casual")
    elif formality_delta > t_form:
        flags.append("tone_too_formal")

    if length_delta < -t_len:
        flags.append("sentences_too_short")
    elif length_delta > t_len:
        flags.append("sentences_too_long")

    # missing_terms is always non-empty — it is simply the brand vocabulary the
    # text did not use, and no short piece of copy uses all fifteen words. The
    # flag fires on genuine absence: fewer than two brand terms present at all.
    if len(diagnostics.aligned_terms) < 2:
        flags.append("missing_brand_keywords")

    # Likewise, a single untouched pillar is normal. The flag fires when the
    # copy misses the majority of them.
    if len(diagnostics.missing_pillar_terms) >= 3:
        flags.append("missing_pillar_coverage")

    return flags


def _summary_sentence(brand_name, flags, diagnostics):
    """
    One or two sentences of plain English.

    This string is fed to the LLM inside the rewrite prompt, so it is written
    as an instruction-friendly description rather than a UI label.
    """
    if not flags:
        return (f"The input text is consistent with the {brand_name} voice on every "
                f"measured dimension.")

    phrases = {
        "sentiment_too_positive": "noticeably more enthusiastic",
        "sentiment_too_negative": "noticeably more negative",
        "readability_too_high": "much easier to read",
        "readability_too_low": "considerably denser",
        "tone_too_casual": "more casual",
        "tone_too_formal": "more formal",
        "sentences_too_short": "written in shorter sentences",
        "sentences_too_long": "written in longer sentences",
    }
    said = [phrases[f] for f in flags if f in phrases]

    parts = []
    if said:
        if len(said) == 1:
            joined = said[0]
        else:
            joined = ", ".join(said[:-1]) + " and " + said[-1]
        parts.append(f"The input text is {joined} than the {brand_name} brand voice.")

    if diagnostics.off_brand_terms:
        parts.append(
            "It leans on off-brand vocabulary such as "
            + ", ".join(f"'{w}'" for w in diagnostics.off_brand_terms[:3]) + "."
        )
    if diagnostics.missing_terms:
        parts.append(
            "It omits characteristic brand terms such as "
            + ", ".join(f"'{w}'" for w in diagnostics.missing_terms[:3]) + "."
        )
    if diagnostics.missing_pillar_terms:
        parts.append(
            "It does not touch the "
            + ", ".join(diagnostics.missing_pillar_terms[:2])
            + " messaging pillar(s)."
        )

    return " ".join(parts)


def generate_drift_report(text_features, brand_profile,
                          diagnostics=None, db_path=SQLITE_DB_PATH):
    """
    Build a DriftReport for one text against one brand profile.

    Parameters
    ----------
    text_features : dict   must contain "text"; pre-computed features are used
                           when present, exactly as in score_consistency
    brand_profile : dict   parsed profile_json
    diagnostics   : Diagnostics, optional — recomputed if not supplied

    Returns
    -------
    DriftReport
    """
    text = text_features.get("text", "") or ""
    brand_id = brand_profile.get("brand_id", "")
    brand_name = brand_profile.get("brand_name", brand_id)

    sentiment = text_features.get("sentiment_score")
    if sentiment is None:
        sentiment = sentiment_proxy(text)
    flesch = text_features.get("flesch_reading_ease")
    if flesch is None:
        flesch = flesch_score(text)
    formality = text_features.get("formality")
    if formality is None:
        formality = formality_proxy(text)
    sent_len = text_features.get("sentence_length")
    if sent_len is None:
        sent_len = mean_sentence_length(text)

    sentiment_delta = float(sentiment) - float(brand_profile.get("mean_sentiment", 0.0))
    readability_delta = float(flesch) - float(brand_profile.get("mean_flesch", 50.0))
    formality_delta = float(formality) - float(brand_profile.get("mean_formality", 0.5))
    length_delta = float(sent_len) - float(brand_profile.get("mean_sentence_length", 20.0))

    if diagnostics is None:
        diagnostics = build_diagnostics(text, brand_profile, db_path=db_path)

    flags = _flag_list(sentiment_delta, readability_delta, formality_delta,
                       length_delta, diagnostics, brand_profile)

    return DriftReport(
        brand_id=brand_id,
        drift_flags=flags,
        sentiment_delta=round(sentiment_delta, 4),
        readability_delta=round(readability_delta, 2),
        missing_keywords=list(diagnostics.missing_terms),
        excess_keywords=list(diagnostics.off_brand_terms),
        summary=_summary_sentence(brand_name, flags, diagnostics),
    )
