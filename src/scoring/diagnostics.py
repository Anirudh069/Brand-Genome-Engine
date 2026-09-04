# src/scoring/diagnostics.py
# Person C — Phase 4A deliverable 3: additional diagnostics
#
# The five numbers from consistency_scorer answer "how on-brand is this?".
# This module answers "which words made it that way?" — the part a writer can
# actually act on, and the input the drift report and edit plan are built from.
#
# Contents:
#   * aligned_terms        — brand vocabulary the text already uses
#   * missing_terms        — brand vocabulary the text is missing
#   * off_brand_terms      — prominent words in the text that are foreign to the brand
#   * pillar_coverage      — presence of each messaging pillar
#   * missing_pillar_terms — pillars the text does not touch
#   * name_mentions        — brand-name mentions, tracked NEUTRALLY (zero weight)

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field, asdict

from src.profiles.brand_profile_builder import (
    _content_words,
    _tokenize,
    SQLITE_DB_PATH,
)

# ── Messaging pillars ─────────────────────────────────────────────────────────
#
# The five pillars are fixed by the Phase 4A brief. The keyword set for each is
# NOT hardcoded: these seeds are used to find the words the brand's own corpus
# actually uses around that theme, so each brand gets its own pillar vocabulary.
# If Person B exports auto-derived pillar sets, pass them into
# derive_pillar_keywords(overrides=...) and they take precedence.

PILLARS = ["Sustainability", "Precision", "Heritage", "Value", "Innovation"]

PILLAR_SEEDS = {
    "Sustainability": [
        "sustainable", "sustainability", "environment", "environmental",
        "responsible", "recycled", "ethical", "carbon", "planet", "future",
        "conservation", "ocean", "preserve", "renewable",
    ],
    "Precision": [
        "precision", "precise", "accuracy", "accurate", "chronometer",
        "certified", "tolerance", "calibration", "tested", "movement",
        "mechanism", "engineering", "technical", "performance",
    ],
    "Heritage": [
        "heritage", "history", "historic", "tradition", "traditional",
        "founded", "legacy", "generations", "archive", "origins", "century",
        "decades", "since", "enduring", "timeless",
    ],
    "Value": [
        "value", "quality", "accessible", "affordable", "investment", "worth",
        "durable", "reliable", "everyday", "practical", "guarantee",
        "warranty", "service",
    ],
    "Innovation": [
        "innovation", "innovative", "new", "pioneering", "pioneer", "advance",
        "advanced", "patent", "patented", "breakthrough", "developed",
        "technology", "modern", "research", "first",
    ],
}


@dataclass
class Diagnostics:
    """Word-level explanation of a consistency score."""
    brand_id: str
    aligned_terms: list = field(default_factory=list)
    missing_terms: list = field(default_factory=list)
    off_brand_terms: list = field(default_factory=list)
    pillar_coverage: dict = field(default_factory=dict)
    missing_pillar_terms: list = field(default_factory=list)
    name_mentions: int = 0
    name_mention_note: str = ""

    def to_dict(self):
        return asdict(self)


# ── Pillar keyword derivation ─────────────────────────────────────────────────

def derive_pillar_keywords(brand_id, db_path=SQLITE_DB_PATH, overrides=None):
    """
    Return {pillar: [keyword, ...]} for one brand.

    Keeps only the seed terms that genuinely appear in this brand's corpus, and
    adds any word that co-occurs with a seed in the same text at least three
    times. A brand that never writes about sustainability ends up with an empty
    Sustainability set rather than a set of words it has never used.

    `overrides` accepts Person B's auto-derived pillar sets and wins outright.
    """
    if overrides:
        return {p: list(overrides.get(p, [])) for p in PILLARS}

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT text FROM brand_texts WHERE brand_id = ?", (brand_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return {p: [] for p in PILLARS}
    finally:
        conn.close()

    texts = [r[0] or "" for r in rows]
    n_texts = max(1, len(texts))

    corpus_counts = Counter()
    doc_freq = Counter()
    for t in texts:
        words = _content_words(t)
        corpus_counts.update(words)
        doc_freq.update(set(words))

    # Words the brand uses everywhere ("watch", "time") and its own name are
    # not evidence of any particular pillar — they would make every text look
    # like it covers Heritage and Value simply for mentioning the product.
    # The 0.22 threshold was tuned on the watch corpus: it removes "watch",
    # "time" and "swiss" while keeping genuinely thematic vocabulary.
    ubiquitous = {w for w, df in doc_freq.items() if df / n_texts > 0.22}
    name_tokens = set(_tokenize(brand_id.replace("_", " ")))

    result = {}
    for pillar, seeds in PILLAR_SEEDS.items():
        present = [s for s in seeds if corpus_counts.get(s, 0) > 0]
        if not present:
            result[pillar] = []
            continue

        seed_set = set(present)
        seed_docs = 0
        co_occurring = Counter()
        for t in texts:
            words = set(_content_words(t))
            if words & seed_set:
                seed_docs += 1
                co_occurring.update(w for w in words if w not in seed_set)

        extras = []
        for w, c in co_occurring.most_common(120):
            if len(extras) >= 6:
                break
            if w in ubiquitous or w in name_tokens or corpus_counts[w] < 3:
                continue
            # Lift: does this word appear in pillar texts more often than in
            # the corpus at large? Below 1.3 it is just a common word.
            lift = (c / max(1, seed_docs)) / (doc_freq[w] / n_texts)
            if lift >= 1.3:
                extras.append(w)

        result[pillar] = present + extras

    return result


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _prominent_words(text, limit=40):
    """Content words of the text, most frequent first."""
    counts = Counter(_content_words(text))
    return [w for w, _ in counts.most_common(limit)]


def count_name_mentions(text, profile):
    """
    Count brand-name mentions.

    Reported for transparency only — this number never enters any score. The
    Phase 4A brief requires name mentions to be a neutral signal, and the
    scorer strips name tokens before measuring vocabulary precisely so that
    repeating the brand name cannot inflate a result.
    """
    tokens = set(profile.get("brand_name_tokens") or [])
    if not tokens:
        return 0
    return sum(1 for w in _tokenize(text) if w in tokens)


def build_diagnostics(text, profile, pillar_keywords=None, db_path=SQLITE_DB_PATH):
    """
    Produce the word-level Diagnostics for one text against one brand profile.
    """
    brand_id = profile.get("brand_id", "")
    brand_keywords = profile.get("top_keywords") or []
    name_tokens = set(profile.get("brand_name_tokens") or [])

    text_words = [w for w in _content_words(text) if w not in name_tokens]
    text_set = set(text_words)
    brand_set = set(brand_keywords)

    aligned = [w for w in brand_keywords if w in text_set]
    missing = [w for w in brand_keywords if w not in text_set][:8]

    # A word counts as off-brand only if the brand's whole corpus never uses
    # it — not merely if it is outside the top 15. Otherwise ordinary words the
    # brand writes constantly ("watch", "wear") get reported as off-brand and
    # end up in the edit plan's avoid list, which would tell the LLM to stop
    # using the brand's own vocabulary.
    known_vocab = set(profile.get("common_vocab") or []) | brand_set

    # Off-brand terms: words the text leans on that the brand's vocabulary does
    # not contain. Ordered by prominence in the text so the most conspicuous
    # offenders surface first.
    off_brand = [w for w in _prominent_words(text)
                 if w not in known_vocab and w not in name_tokens and len(w) >= 4][:8]

    if pillar_keywords is None:
        pillar_keywords = derive_pillar_keywords(brand_id, db_path=db_path)

    coverage = {}
    for pillar, words in pillar_keywords.items():
        if not words:
            coverage[pillar] = 0.0
            continue
        hits = len(text_set & set(words))
        coverage[pillar] = round(min(1.0, hits / 3.0), 3)   # 3 hits = full marks

    missing_pillars = [p for p, v in coverage.items() if v == 0.0 and pillar_keywords.get(p)]

    mentions = count_name_mentions(text, profile)

    return Diagnostics(
        brand_id=brand_id,
        aligned_terms=aligned,
        missing_terms=missing,
        off_brand_terms=off_brand,
        pillar_coverage=coverage,
        missing_pillar_terms=missing_pillars,
        name_mentions=mentions,
        name_mention_note=(
            f"{mentions} brand-name mention(s) detected. Tracked for reporting only — "
            f"name mentions carry zero scoring weight."
        ),
    )
