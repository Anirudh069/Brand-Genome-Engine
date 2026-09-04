# src/profiles/brand_profile_builder.py
# Person C — Brand Profile Builder
#
# Reads brand_texts from SQLite, computes a per-brand "genome" (style statistics
# + characteristic vocabulary), and writes it to the brand_profiles table.
#
# Design notes (see docs/scoring_spec.md for full rationale):
#   * Characteristic vocabulary is selected by TF-IDF ACROSS brands, not by raw
#     frequency. Raw frequency surfaces words every watch brand uses ("watch",
#     "time", "case"); TF-IDF surfaces what makes THIS brand sound like itself.
#   * Proper nouns (brand names, product lines) are excluded from the voice
#     vocabulary and tracked separately as a neutral signal, per the Phase 4A
#     requirement that brand-name mentions carry no scoring weight.
#   * Sentiment is a graded density score, not a ternary ratio, so that
#     std_sentiment describes a real distribution the scorer can compare against.

import sqlite3
import json
import re
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

# ── Path to DB (same file Person A owns) ──────────────────────────────────────
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "data/brand_data.db")

PROFILE_VERSION = 2          # bumped: v1 used raw-frequency keywords
MIN_TEXTS_PER_BRAND = 10     # below this the profile is unreliable — skip + warn

# ── Tokenisation ──────────────────────────────────────────────────────────────

WORD_RE = re.compile(r"[A-Za-z']+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "as", "at", "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "by", "from", "you", "we", "they", "he", "she",
    "i", "our", "your", "their", "its", "not", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "can", "all",
    "more", "also", "than", "into", "which", "about", "when", "where", "what",
    "who", "how", "there", "here", "then", "them", "his", "her", "him", "each",
    "over", "such", "only", "other", "any", "both", "through", "between",
    "after", "before", "while", "during", "under", "again", "most", "some",
    "very", "own", "same", "just", "now", "one", "two", "first", "new", "make",
    "made", "well", "even", "still", "back", "way", "many", "much",
    # Function words that survive a length filter but carry no brand signal.
    # They matter most for the user genome, which is built from ~8 short texts
    # where a single occurrence is enough to rank a word highly.
    "until", "because", "since", "though", "although", "however", "therefore",
    "whether", "upon", "within", "without", "across", "among", "toward",
    "towards", "unless", "already", "another", "itself", "themselves",
    "every", "being", "having", "doing", "get", "got", "put", "take", "taken",
    "come", "comes", "give", "given", "goes", "going", "want", "need", "let",
}


def _tokenize(text):
    """Lowercase word list."""
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _tokenize_cased(text):
    """Word list preserving original casing — needed for proper-noun detection."""
    return WORD_RE.findall(text or "")


def _content_words(text):
    """Non-stopword words of length >= 3."""
    return [w for w in _tokenize(text) if w not in STOPWORDS and len(w) >= 3]


# ── Basic statistics ──────────────────────────────────────────────────────────

def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ── Style features (one value per text) ───────────────────────────────────────

def _sentences(text):
    parts = re.split(r"[.!?]+", text or "")
    return [s.strip() for s in parts if s.strip()]


def _syllables(word):
    """Rough syllable count: number of vowel groups, minimum 1."""
    return max(1, len(re.findall(r"[aeiou]+", word.lower())))


def flesch_score(text):
    """
    Flesch Reading Ease.
        206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    Higher = easier to read. Returns 50.0 (neutral) for empty input.
    """
    sents = _sentences(text)
    words = _tokenize(text)
    if not sents or not words:
        return 50.0
    asl = len(words) / len(sents)
    asw = sum(_syllables(w) for w in words) / len(words)
    return 206.835 - 1.015 * asl - 84.6 * asw


def vocab_richness(text, window=100):
    """
    Type-token ratio measured over a fixed window of `window` words.

    Plain TTR falls as text gets longer (a 500-word text repeats "the" more
    often than a 50-word one), which makes long and short texts incomparable.
    Measuring over a fixed window removes that length bias.
    Range 0-1.
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    words = words[:window]
    return len(set(words)) / len(words)


def formality_proxy(text):
    """
    Formality register, 0-1. Three signals, equally weighted:
      * ratio of long (>= 7 char) content words — Latinate vocabulary
      * absence of contractions ("don't", "it's")
      * absence of second-person address ("you", "your")
    Luxury brand copy is long-worded, contraction-free and third-person;
    casual copy is the opposite.
    """
    content = _content_words(text)
    if not content:
        return 0.5
    long_ratio = sum(1 for w in content if len(w) >= 7) / len(content)

    words = _tokenize(text)
    n = max(1, len(words))
    contractions = len(re.findall(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", text or "", re.I))
    second_person = sum(1 for w in words if w in {"you", "your", "yours"})

    contraction_score = max(0.0, 1.0 - (contractions / n) * 20.0)
    person_score = max(0.0, 1.0 - (second_person / n) * 15.0)

    # Weighted 0.6 / 0.2 / 0.2 rather than equally. Most copy contains no
    # contractions and no second-person address, so those two components sit
    # at 1.0 almost always; averaging them equally with long_ratio squeezed
    # every text into the 0.67-0.85 band and left the metric unable to
    # separate casual copy from formal copy at all.
    return max(0.0, min(1.0,
                        0.6 * long_ratio + 0.2 * contraction_score + 0.2 * person_score))


# ── Sentiment ─────────────────────────────────────────────────────────────────

# Sentiment lexicon — EMOTIONAL VALENCE ONLY.
#
# Deliberately excludes brand-value vocabulary ("precision", "heritage",
# "craft", "quality", "innovation", "expertise"). Those words are topical, not
# emotional: they belong to the vocabulary metric, which measures them already.
# When they sat in this lexicon, a well-written luxury paragraph registered as
# wildly more positive than the brand's own corpus and scored 0.5% on sentiment
# alignment — the two metrics were double-counting the same words and then
# punishing the text for it.
POSITIVE_LEXICON = {
    "beautiful", "beauty", "love", "loved", "lovely", "happy", "happiness",
    "delight", "delighted", "delightful", "joy", "joyful", "proud", "pride",
    "exciting", "excited", "excitement", "inspiring", "inspired", "inspire",
    "celebrate", "celebrated", "celebration", "admire", "admired", "stunning",
    "spectacular", "wonderful", "amazing", "awesome", "great", "fantastic",
    "brilliant", "charming", "elegant", "elegance", "graceful", "warm",
    "welcome", "welcoming", "favourite", "favorite", "enjoy", "enjoyed",
    "pleasure", "pleasing", "fun", "cool", "easy", "effortless", "comfortable",
    "friendly", "generous", "bold", "daring", "confident", "triumph",
    "victory", "success", "successful", "thrilling", "remarkable",
    "extraordinary", "exceptional", "perfect", "finest", "best",
}

NEGATIVE_LEXICON = {
    "fail", "failure", "failed", "poor", "cheap", "bad", "terrible", "awful",
    "inferior", "weak", "broken", "wrong", "defect", "defective", "problem",
    "problems", "issue", "issues", "concern", "concerned", "risk", "risky",
    "loss", "reject", "rejected", "difficult", "difficulty", "flaw", "flawed",
    "lack", "lacking", "disappointing", "disappointed", "mediocre", "dull",
    "boring", "ugly", "harsh", "unfortunate", "sadly", "sad", "worry",
    "worried", "fear", "afraid", "struggle", "struggled", "painful",
}


def sentiment_proxy(text):
    """
    Graded sentiment density in [-1, 1].

        raw = (positive_hits - negative_hits) / sqrt(content_word_count)
        sentiment = tanh(raw)

    The earlier version used (pos - neg) / (pos + neg), which collapses to
    almost exactly -1, 0 or +1 for real copy. That produced a std_sentiment
    around 0.43, which made the scorer's Gaussian so wide that unrelated text
    still scored 75%. Dividing by sqrt(length) instead gives a continuous,
    length-fair value with a distribution worth comparing against.
    """
    content = _content_words(text)
    if not content:
        return 0.0
    pos = sum(1 for w in content if w in POSITIVE_LEXICON)
    neg = sum(1 for w in content if w in NEGATIVE_LEXICON)
    raw = (pos - neg) / math.sqrt(len(content))
    return math.tanh(raw)


def mean_sentence_length(text):
    sents = _sentences(text)
    words = _tokenize(text)
    if not sents or not words:
        return 0.0
    return len(words) / len(sents)


# ── Characteristic vocabulary (TF-IDF, proper nouns removed) ──────────────────

def _detect_proper_nouns(texts, threshold=0.6):
    """
    A word is treated as a proper noun if it appears capitalised in more than
    `threshold` of its occurrences, ignoring sentence-initial position.

    Brand names and product lines ("Rolex", "Oyster", "Perpetual") are proper
    nouns. They are excluded from the voice vocabulary because the Phase 4A
    spec requires brand-name mentions to be a NEUTRAL signal carrying no
    scoring weight — otherwise repeating the brand name is the cheapest way to
    fake a high score.
    """
    upper = Counter()
    total = Counter()
    for text in texts:
        for sent in _sentences(text):
            tokens = _tokenize_cased(sent)
            for i, tok in enumerate(tokens):
                if i == 0:                      # sentence-initial caps mean nothing
                    continue
                low = tok.lower()
                total[low] += 1
                if tok[0].isupper():
                    upper[low] += 1
    return {w for w, n in total.items() if n >= 2 and upper[w] / n > threshold}


NUMBER_RE = re.compile(r"\d")


def _detect_spec_register(texts, threshold=0.4, window=3):
    """
    Detect specification-sheet vocabulary: words that mostly appear next to
    numbers ("approx 39 mm", "thickness 9.6", "totaling 2.5 carats").

    Some brands' scraped corpora contain product spec tables. Those words are
    distinctive by TF-IDF but say nothing about voice, so they would poison
    the vocabulary. Detecting them by proximity to digits is data-driven —
    no hardcoded blocklist, so it adapts if the corpus changes.
    """
    near = Counter()
    total = Counter()
    for text in texts:
        tokens = WORD_RE.split(text or "")     # keeps numeric/punctuation gaps
        words = _tokenize_cased(text)
        raw = re.findall(r"[A-Za-z']+|\d+[\d.,]*", text or "")
        for i, tok in enumerate(raw):
            if not tok[0].isalpha():
                continue
            low = tok.lower()
            total[low] += 1
            lo = max(0, i - window)
            hi = min(len(raw), i + window + 1)
            if any(NUMBER_RE.match(raw[j]) for j in range(lo, hi) if j != i):
                near[low] += 1
    return {w for w, n in total.items() if n >= 3 and near[w] / n > threshold}


def compute_distinctive_keywords(by_brand, k=15):
    """
    Rank each brand's vocabulary by TF-IDF across the other brands.

        tf_sublinear = 1 + log(count of word in brand)
        idf          = log(n_brands / n_brands containing word)
        score        = tf_sublinear * (idf + 0.5)

    Raw frequency answers "what does this brand talk about most", which for
    every watch brand is "watch", "time", "case". TF-IDF answers "what does
    this brand say that the others don't", which is the actual voice signal.

    Sublinear tf stops a single very frequent word dominating; the +0.5 floor
    on idf keeps shared-but-central vocabulary from being discarded entirely
    when a word appears in every brand (idf = 0).

    The returned vocabulary is a blend of two lists:
      * the top `k_distinctive` words by TF-IDF — what sets this brand apart
      * the top `k_frequent` words by raw count — what it says most often

    TF-IDF alone drifts toward rare niche vocabulary and misses the central
    words a brand repeats constantly; raw frequency alone returns the category
    nouns every competitor shares. Together they describe a brand the way a
    tone-of-voice guideline does: "we sound like this, and we talk about that."

    Excluded from the vocabulary:
      * proper nouns — brand and product names, tracked neutrally instead
      * specification-sheet words — measurement vocabulary, no voice signal
      * words appearing fewer than 3 times — noise, not a pattern

    Returns ({brand_key: [keyword, ...]}, proper_noun_set).
    """
    brand_counts = {}
    doc_freq = Counter()
    proper_nouns = set()
    spec_words = set()

    for texts in by_brand.values():
        proper_nouns |= _detect_proper_nouns(texts)
        spec_words |= _detect_spec_register(texts)

    for key, texts in by_brand.items():
        counts = Counter()
        for t in texts:
            counts.update(_content_words(t))
        brand_counts[key] = counts
        for w in counts:
            doc_freq[w] += 1

    n_brands = max(1, len(by_brand))
    k_distinctive = int(round(k * 2 / 3))
    k_frequent = k - k_distinctive
    keywords = {}

    for key, counts in brand_counts.items():
        brand_id, brand_name = key
        brand_tokens = {brand_id.lower()} | set(_tokenize(brand_name))
        excluded = proper_nouns | spec_words | brand_tokens

        eligible = [(w, c) for w, c in counts.items() if w not in excluded and c >= 3]

        by_tfidf = sorted(
            eligible,
            key=lambda wc: (1.0 + math.log(wc[1]))
            * ((math.log(n_brands / doc_freq[wc[0]]) if doc_freq[wc[0]] else 0.0) + 0.5),
            reverse=True,
        )
        # The frequency tier skips words that appear in EVERY brand's corpus.
        # "watch", "time", "dial", "design" are category nouns, not brand
        # vocabulary — including them lets any watch-shaped text score, and
        # rewards keyword stuffing.
        by_count = sorted(
            [wc for wc in eligible if doc_freq[wc[0]] < n_brands],
            key=lambda wc: wc[1],
            reverse=True,
        )

        selected = [w for w, _ in by_tfidf[:k_distinctive]]
        for w, _ in by_count:
            if len(selected) >= k:
                break
            if w not in selected:
                selected.append(w)

        keywords[key] = selected

    return keywords, proper_nouns


def _tone_label(mean_formality, mean_sentiment):
    """Human-readable tone label from the two register dimensions."""
    if mean_formality >= 0.60 and mean_sentiment >= 0.30:
        return "authoritative"
    if mean_formality >= 0.60:
        return "formal"
    if mean_sentiment >= 0.30:
        return "warm"
    if mean_formality < 0.40:
        return "conversational"
    return "neutral"


# ── Main builder ──────────────────────────────────────────────────────────────

def build_brand_profiles(db_path=SQLITE_DB_PATH, verbose=True):
    """
    Read brand_texts, compute per-brand profiles, write to brand_profiles.
    Returns the number of profiles written.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS brand_profiles (
        brand_id     TEXT PRIMARY KEY,
        brand_name   TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        built_at     TEXT NOT NULL DEFAULT (datetime('now')),
        version      INTEGER NOT NULL DEFAULT 1,
        n_texts      INTEGER NOT NULL
    )
    """)
    conn.commit()

    # Carry forward anything an earlier build produced that this module does not
    # recompute — above all mean_embedding, Person B's 384-dimension vector.
    # Rebuilding the style statistics must never discard his work.
    preserved = {}
    try:
        for old_id, old_json in cur.execute(
                "SELECT brand_id, profile_json FROM brand_profiles").fetchall():
            old = json.loads(old_json)
            preserved[old_id] = {
                key: old[key]
                for key in ("mean_embedding", "embedding_status", "snippets",
                            "snippetsCount")
                if key in old
            }
    except (sqlite3.OperationalError, ValueError):
        preserved = {}

    cur.execute("SELECT brand_id, brand_name, text FROM brand_texts")
    rows = cur.fetchall()

    by_brand = defaultdict(list)
    for brand_id, brand_name, text in rows:
        by_brand[(brand_id, brand_name)].append(text or "")

    keywords_by_brand, proper_nouns = compute_distinctive_keywords(by_brand)

    # A wider vocabulary snapshot per brand. Not used for scoring — the
    # diagnostics use it to tell "this word is foreign to the brand" apart from
    # "this word simply is not in the top 15", so that ordinary words the brand
    # uses all the time never get reported as off-brand.
    common_vocab = {}
    for key, texts in by_brand.items():
        counts = Counter()
        for t in texts:
            counts.update(_content_words(t))
        common_vocab[key] = [w for w, _ in counts.most_common(400)]

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = 0
    skipped = []

    for key, texts in by_brand.items():
        brand_id, brand_name = key

        if len(texts) < MIN_TEXTS_PER_BRAND:
            skipped.append((brand_id, len(texts)))
            continue

        sentiments = [sentiment_proxy(t) for t in texts]
        flesches = [flesch_score(t) for t in texts]
        richness = [vocab_richness(t) for t in texts]
        formality = [formality_proxy(t) for t in texts]
        sent_lens = [mean_sentence_length(t) for t in texts]

        profile = {
            "brand_id": brand_id,
            "brand_name": brand_name,
            "mean_sentiment": round(_mean(sentiments), 4),
            "std_sentiment": round(max(_std(sentiments), 0.01), 4),
            "mean_flesch": round(_mean(flesches), 2),
            "std_flesch": round(max(_std(flesches), 0.01), 2),
            "mean_vocab_richness": round(_mean(richness), 4),
            "std_vocab_richness": round(max(_std(richness), 0.01), 4),
            "mean_formality": round(_mean(formality), 4),
            "std_formality": round(max(_std(formality), 0.01), 4),
            "mean_sentence_length": round(_mean(sent_lens), 2),
            "std_sentence_length": round(max(_std(sent_lens), 0.5), 2),
            "top_keywords": keywords_by_brand.get(key, []),
            "common_vocab": common_vocab.get(key, []),
            "brand_name_tokens": sorted({brand_id.lower()} | set(_tokenize(brand_name))),
            "tone_label": _tone_label(_mean(formality), _mean(sentiments)),
            "built_at": built_at,
            "version": PROFILE_VERSION,
            "sentiment_scale": "signed",   # see note below
            "n_texts": len(texts),
        }

        # Keep Person B's embedding and anything else an earlier build wrote.
        profile.update(preserved.get(brand_id, {}))
        profile.setdefault("mean_embedding", [])

        # Compatibility aliases for the API and the frontend, which read these
        # names. `avg_sentiment` is deliberately remapped: this module measures
        # sentiment on a signed -1..1 scale, while the Analytics page multiplies
        # avg_sentiment by 100 and expects 0..1. `sentiment_scale` above tells
        # the scorer which convention `mean_sentiment` uses, so a profile built
        # by the older script is still read correctly.
        profile["avg_sentiment"] = round((profile["mean_sentiment"] + 1.0) / 2.0, 4)
        profile["avg_formality"] = profile["mean_formality"]
        profile["avg_readability_flesch"] = profile["mean_flesch"]
        profile["vocabulary_richness"] = profile["mean_vocab_richness"]

        cur.execute("""
        INSERT OR REPLACE INTO brand_profiles
            (brand_id, brand_name, profile_json, built_at, version, n_texts)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (brand_id, brand_name, json.dumps(profile), built_at,
              PROFILE_VERSION, len(texts)))
        written += 1

    conn.commit()
    conn.close()

    if verbose:
        print(f"[brand_profile_builder] Built {written} profiles (v{PROFILE_VERSION}) -> {db_path}")
        print(f"[brand_profile_builder] Excluded {len(proper_nouns)} proper nouns from voice vocabulary")
        for brand_id, n in skipped:
            print(f"[brand_profile_builder] WARNING: skipped '{brand_id}' — only {n} texts "
                  f"(minimum {MIN_TEXTS_PER_BRAND})")

    return written


# ── User brand genome (Phase 4A deliverable 1) ────────────────────────────────
#
# The competitor genomes above are built from 45 texts each. The user's genome
# is built from their mission plus exactly 7 snippets — 8 short texts. That is a
# small sample, so every standard deviation gets a generous floor: with 8
# samples a narrow spread is far more likely to be an accident of the sample
# than a real fact about the brand, and a scorer that trusted it would reject
# perfectly good copy. The floors make the user's genome forgiving where the
# evidence is thin, which is the honest behaviour.

USER_BRAND_ID = "user_brand"
MIN_USER_TEXTS = 3

# Minimum spread per dimension for a genome built from ~8 texts.
USER_STD_FLOORS = {
    "sentiment": 0.18,
    "flesch": 12.0,
    "vocab_richness": 0.08,
    "formality": 0.10,
    "sentence_length": 4.0,
}


def _competitor_doc_freq(db_path):
    """
    How many competitor brands use each word — the reference point that makes
    the user's distinctive vocabulary distinctive.
    """
    doc_freq = Counter()
    n_brands = 0
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT brand_id, text FROM brand_texts WHERE brand_id != ?",
            (USER_BRAND_ID,)).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        return doc_freq, 0

    by_brand = defaultdict(set)
    for brand_id, text in rows:
        by_brand[brand_id].update(_content_words(text or ""))
    n_brands = len(by_brand)
    for words in by_brand.values():
        doc_freq.update(words)
    return doc_freq, n_brands


def extract_user_keywords(texts, brand_name, db_path, k=15):
    """
    Characteristic vocabulary for the user's brand.

    Scored by the same TF-IDF logic as the competitor genomes, using the
    competitor corpus as the reference: a word the user repeats that no
    competitor uses is highly characteristic; a word every watch brand uses is
    not. Falls back to plain frequency when no competitor corpus is available.
    """
    counts = Counter()
    for text in texts:
        counts.update(_content_words(text))
    if not counts:
        return []

    spec_words = _detect_spec_register(texts)
    brand_tokens = set(_tokenize(brand_name or ""))
    excluded = spec_words | brand_tokens

    doc_freq, n_brands = _competitor_doc_freq(db_path)

    def rank(candidates):
        scored = []
        for word, count in candidates:
            tf = 1.0 + math.log(count)
            if n_brands:
                idf = (math.log(n_brands / doc_freq[word]) if doc_freq.get(word)
                       else math.log(n_brands))
            else:
                idf = 0.0
            scored.append((word, tf * (idf + 0.5)))
        scored.sort(key=lambda wc: wc[1], reverse=True)
        return [w for w, _ in scored]

    eligible = [(w, c) for w, c in counts.items() if w not in excluded]

    # A word used once is a word; a word used twice is a choice. With only ~8
    # short texts every term is rare, so raw TF-IDF promotes whatever happens to
    # be unusual. Recurrence across the user's own writing is the better signal
    # of intent, so repeated words are ranked first and single-use words only
    # fill the remaining slots.
    repeated = rank([wc for wc in eligible if wc[1] >= 2])
    single = rank([wc for wc in eligible if wc[1] < 2])

    keywords = repeated[:k]
    for word in single:
        if len(keywords) >= k:
            break
        keywords.append(word)
    return keywords


def build_user_genome(brand_name, mission, snippets, tone_label=None,
                      db_path=SQLITE_DB_PATH, weight_preset=None,
                      extra=None, persist=True):
    """
    Build the user's brand genome from their mission and sample copy.

    This is what the Consistency Check page scores against, so it must be
    measured from the user's actual writing. The previous implementation stored
    three hardcoded keywords and a formality value picked from a dropdown, which
    meant the main screen of the application was scoring copy against a
    placeholder.

    Parameters
    ----------
    brand_name : str
    mission    : str          mission / core vision statement
    snippets   : list[str]    sample copy, 7 expected
    tone_label : str          the user's own description of their tone; kept as
                              `declared_tone` and used only if too little text
                              is supplied to measure one
    weight_preset : str       "balanced" | "tone_heavy" | "semantic_heavy"
    extra      : dict         merged in last — e.g. centroid_embedding
    persist    : bool         write to brand_profiles

    Returns the profile dict. Raises ValueError if there is too little text.
    """
    snippets = [s for s in (snippets or []) if s and s.strip()]
    texts = [t for t in ([mission] if mission and mission.strip() else []) + snippets]

    if len(texts) < MIN_USER_TEXTS:
        raise ValueError(
            f"Need at least {MIN_USER_TEXTS} texts to build a genome "
            f"(mission + snippets); got {len(texts)}."
        )

    sentiments = [sentiment_proxy(t) for t in texts]
    flesches = [flesch_score(t) for t in texts]
    richness = [vocab_richness(t) for t in texts]
    formality = [formality_proxy(t) for t in texts]
    sent_lens = [mean_sentence_length(t) for t in texts]

    def spread(values, floor_key):
        return round(max(_std(values), USER_STD_FLOORS[floor_key]), 4)

    keywords = extract_user_keywords(texts, brand_name, db_path)
    measured_tone = _tone_label(_mean(formality), _mean(sentiments))

    profile = {
        "brand_id": USER_BRAND_ID,
        "brand_name": brand_name,
        "name": brand_name,                 # compatibility with the frontend
        "mission": mission,
        "mean_sentiment": round(_mean(sentiments), 4),
        "std_sentiment": spread(sentiments, "sentiment"),
        "mean_flesch": round(_mean(flesches), 2),
        "std_flesch": spread(flesches, "flesch"),
        "mean_vocab_richness": round(_mean(richness), 4),
        "std_vocab_richness": spread(richness, "vocab_richness"),
        "mean_formality": round(_mean(formality), 4),
        "std_formality": spread(formality, "formality"),
        "mean_sentence_length": round(_mean(sent_lens), 2),
        "std_sentence_length": spread(sent_lens, "sentence_length"),
        "top_keywords": keywords,
        "common_vocab": [w for w, _ in Counter(
            w for t in texts for w in _content_words(t)).most_common(200)],
        "brand_name_tokens": sorted(set(_tokenize(brand_name or "")) | {USER_BRAND_ID}),
        "mean_embedding": [],
        "tone_label": measured_tone,
        "declared_tone": tone_label,
        "weight_preset": weight_preset or "balanced",
        "sentiment_scale": "signed",
        "snippets": snippets,
        "snippetsCount": len(snippets),
        "n_texts": len(texts),
        "version": PROFILE_VERSION,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Aliases the API and frontend read. avg_sentiment is remapped to 0..1
    # because the Analytics page multiplies it by 100.
    profile["avg_sentiment"] = round((profile["mean_sentiment"] + 1.0) / 2.0, 4)
    profile["avg_formality"] = profile["mean_formality"]
    profile["avg_readability_flesch"] = profile["mean_flesch"]
    profile["vocabulary_richness"] = profile["mean_vocab_richness"]

    if extra:
        profile.update(extra)

    if persist:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS brand_profiles (
            brand_id     TEXT PRIMARY KEY,
            brand_name   TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            built_at     TEXT NOT NULL DEFAULT (datetime('now')),
            version      INTEGER NOT NULL DEFAULT 1,
            n_texts      INTEGER NOT NULL
        )
        """)
        cur.execute("""
        INSERT OR REPLACE INTO brand_profiles
            (brand_id, brand_name, profile_json, built_at, version, n_texts)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (USER_BRAND_ID, brand_name, json.dumps(profile),
              profile["built_at"], PROFILE_VERSION, len(texts)))
        conn.commit()
        conn.close()

    return profile


# ── Profile loading (used by the scorer and by Person D's endpoints) ──────────

class BrandProfileNotFoundError(Exception):
    """Raised when a brand has no row in brand_profiles — i.e. genome not initialised."""


def load_brand_profile(brand_id, db_path=SQLITE_DB_PATH):
    """
    Load one brand profile as a dict.
    Raises BrandProfileNotFoundError if the genome has not been built yet.
    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT profile_json FROM brand_profiles WHERE brand_id = ?", (brand_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        raise BrandProfileNotFoundError(
            f"brand_profiles table does not exist — run brand_profile_builder first "
            f"(requested '{brand_id}')"
        )
    finally:
        conn.close()

    if row is None:
        raise BrandProfileNotFoundError(
            f"No genome found for brand '{brand_id}'. Initialise it on the Genome Setup page."
        )
    return json.loads(row[0])


def list_brands(db_path=SQLITE_DB_PATH):
    """Return [{'brand_id':..., 'brand_name':...}] for GET /api/brands."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT brand_id, brand_name FROM brand_profiles ORDER BY brand_name"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [{"brand_id": r[0], "brand_name": r[1]} for r in rows]


def is_genome_initialised(brand_id, db_path=SQLITE_DB_PATH):
    """True if a usable genome exists for this brand."""
    try:
        profile = load_brand_profile(brand_id, db_path)
    except BrandProfileNotFoundError:
        return False
    return bool(profile.get("top_keywords")) and profile.get("n_texts", 0) >= MIN_TEXTS_PER_BRAND


if __name__ == "__main__":
    build_brand_profiles()
