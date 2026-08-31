# src/scoring/edit_plan.py
# Person C — Phase 4B deliverable 6: edit plan generator
#
# Turns a DriftReport into an actionable instruction set. This object IS the
# prompt context Person D's /api/rewrite endpoint feeds to the LLM, so every
# field is written to be read by a model as well as displayed in the UI.
#
# EditPlan field names are FROZEN.

from dataclasses import dataclass, field, asdict

from src.profiles.brand_profile_builder import SQLITE_DB_PATH
from src.scoring.grounding import retrieve_grounding_chunks

# Style rules attached when a given drift flag is present. Phrased as
# imperatives because they are handed straight to the LLM.
FLAG_STYLE_RULES = {
    "tone_too_casual": [
        "Use formal, measured sentence structures",
        "Avoid contractions",
        "Address the reader in the third person rather than as 'you'",
        "Prefer Latinate vocabulary over colloquial equivalents",
    ],
    "tone_too_formal": [
        "Loosen the register — shorter, more direct sentences",
        "Allow contractions where they read naturally",
    ],
    "readability_too_high": [
        "Increase sentence complexity and clause depth",
        "Replace simple words with more precise, specific alternatives",
    ],
    "readability_too_low": [
        "Break long sentences into shorter ones",
        "Reduce subordinate clauses",
    ],
    "sentiment_too_positive": [
        "Remove superlatives and exclamation",
        "State qualities as fact rather than enthusiasm",
    ],
    "sentiment_too_negative": [
        "Reframe limitations as deliberate choices",
    ],
    "sentences_too_short": [
        "Combine related statements into fuller sentences",
    ],
    "sentences_too_long": [
        "Split multi-clause sentences at natural boundaries",
    ],
}


@dataclass
class EditPlan:
    """Actionable rewrite instructions derived from a DriftReport."""
    brand_id: str
    goals: list = field(default_factory=list)
    avoid_terms: list = field(default_factory=list)
    prefer_terms: list = field(default_factory=list)
    style_rules: list = field(default_factory=list)
    tone_direction: str = ""
    grounding_chunks: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def to_prompt(self, original_text, brand_name=None):
        """
        Render the plan as an LLM prompt.

        Provided so that Person D's endpoint does not have to invent prompt
        wording — if the prompt changes, it changes here, next to the logic
        that decided what should go into it.
        """
        name = brand_name or self.brand_id
        lines = [
            f"Rewrite the following copy so that it matches the {name} brand voice.",
            "",
            "Goals:",
        ]
        lines += [f"  - {g}" for g in self.goals] or ["  - Improve brand consistency"]

        if self.style_rules:
            lines += ["", "Style rules:"] + [f"  - {r}" for r in self.style_rules]
        if self.prefer_terms:
            lines += ["", "Prefer this vocabulary where it fits naturally: "
                      + ", ".join(self.prefer_terms)]
        if self.avoid_terms:
            lines += ["Avoid these words: " + ", ".join(self.avoid_terms)]
        if self.tone_direction:
            lines += ["", f"Target tone: {self.tone_direction}"]
        if self.grounding_chunks:
            lines += ["", f"Examples of authentic {name} copy — match this voice, "
                          "do not copy the content:"]
            lines += [f"  {i + 1}. \"{c}\"" for i, c in enumerate(self.grounding_chunks)]

        lines += [
            "",
            "Preserve the original meaning and every factual claim. Do not invent "
            "product features, prices, dates or specifications.",
            "Return only the rewritten copy.",
            "",
            "Original copy:",
            original_text,
        ]
        return "\n".join(lines)


def _goals(drift_report, brand_profile, score_result=None):
    """
    Concrete, measurable goals. Each names the current value and the target so
    the rewrite has something to aim at and the before/after comparison has
    something to verify.
    """
    goals = []
    flags = set(drift_report.drift_flags)

    mean_sentiment = float(brand_profile.get("mean_sentiment", 0.0))
    mean_flesch = float(brand_profile.get("mean_flesch", 50.0))
    mean_formality = float(brand_profile.get("mean_formality", 0.5))

    if "tone_too_casual" in flags:
        goals.append(
            f"Increase formality toward the brand mean of {mean_formality:.2f}")
    if "tone_too_formal" in flags:
        goals.append(
            f"Relax formality toward the brand mean of {mean_formality:.2f}")

    if "readability_too_high" in flags:
        goals.append(
            f"Reduce Flesch reading ease by about {abs(drift_report.readability_delta):.0f} "
            f"points, toward the brand mean of {mean_flesch:.0f}")
    if "readability_too_low" in flags:
        goals.append(
            f"Raise Flesch reading ease by about {abs(drift_report.readability_delta):.0f} "
            f"points, toward the brand mean of {mean_flesch:.0f}")

    if "sentiment_too_positive" in flags:
        goals.append(
            f"Lower the emotional register toward the brand mean of {mean_sentiment:+.2f}")
    if "sentiment_too_negative" in flags:
        goals.append(
            f"Lift the emotional register toward the brand mean of {mean_sentiment:+.2f}")

    if drift_report.missing_keywords:
        goals.append(
            "Introduce brand-anchored vocabulary: "
            + ", ".join(f"'{w}'" for w in drift_report.missing_keywords[:4]))

    if "missing_pillar_coverage" in flags:
        goals.append("Reference at least one core messaging pillar")

    if score_result is not None:
        goals.insert(0, f"Raise the overall consistency score from "
                        f"{score_result.overall_score:.0f}/100")

    if not goals:
        goals.append("Preserve the current voice — no drift detected")

    return goals


def generate_edit_plan(drift_report, brand_profile, original_text="",
                       score_result=None, k_grounding=3,
                       db_path=SQLITE_DB_PATH, retriever=None):
    """
    Build an EditPlan from a DriftReport.

    Parameters
    ----------
    drift_report  : DriftReport
    brand_profile : dict — parsed profile_json
    original_text : str — used to retrieve topically relevant grounding chunks
    score_result  : ScoreResult, optional — lets the plan state the starting score
    retriever     : Person B's FAISS retrieval callable, optional

    Returns
    -------
    EditPlan
    """
    brand_id = brand_profile.get("brand_id", "")
    flags = set(drift_report.drift_flags)

    style_rules = []
    for flag in drift_report.drift_flags:
        for rule in FLAG_STYLE_RULES.get(flag, []):
            if rule not in style_rules:
                style_rules.append(rule)

    target_len = brand_profile.get("mean_sentence_length")
    if target_len:
        style_rules.append(
            f"Aim for roughly {int(round(float(target_len)))} words per sentence")

    tone_label = brand_profile.get("tone_label", "neutral")
    direction = {
        "authoritative": "authoritative, measured and confident",
        "formal": "formal, restrained and precise",
        "warm": "warm and approachable without being casual",
        "conversational": "direct and conversational",
        "neutral": "neutral and factual",
    }.get(tone_label, tone_label)

    if "tone_too_casual" in flags:
        direction += " — noticeably less casual than the input"
    elif "tone_too_formal" in flags:
        direction += " — a little lighter than the input"

    grounding = retrieve_grounding_chunks(
        original_text or drift_report.summary,
        brand_id,
        k=k_grounding,
        brand_profile=brand_profile,
        db_path=db_path,
        retriever=retriever,
    )

    return EditPlan(
        brand_id=brand_id,
        goals=_goals(drift_report, brand_profile, score_result),
        avoid_terms=list(drift_report.excess_keywords)[:6],
        prefer_terms=list(drift_report.missing_keywords)[:6],
        style_rules=style_rules,
        tone_direction=direction,
        grounding_chunks=grounding,
    )
