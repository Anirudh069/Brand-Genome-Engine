# src/profiles/brand_profile_builder.py
# Person C — Brand Profile Builder
# Reads brand_texts from SQLite, computes per-brand averages,
# writes results to brand_profiles table in the same DB.

import sqlite3
import json
import re
import math
from collections import Counter
from datetime import datetime, timezone

# ── Path to DB (same file Person A owns) ──────────────────────────────────────
SQLITE_DB_PATH = "data/brand_data.db"

# ── Helpers ───────────────────────────────────────────────────────────────────
WORD_RE = re.compile(r"[a-zA-Z']+")

STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","at",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "by","from","you","we","they","he","she","i","our","your","their","its","not",
    "have","has","had","do","does","did","will","would","could","should","may",
    "can","its","we","all","been","more","also","than","into","which","about",
}


def _tokenize(text: str) -> list:
    """Split text into lowercase words."""
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _content_words(text: str) -> list:
    """Return non-stopword words of length ≥ 3."""
    return [w for w in _tokenize(text) if w not in STOPWORDS and len(w) >= 3]


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _flesch_score(text: str) -> float:
    """Approximate Flesch Reading Ease without external libraries."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = _tokenize(text)
    if not sentences or not words:
        return 50.0  # neutral default
    # Count syllables (rough approximation)
    def syllables(word):
        word = word.lower()
        count = len(re.findall(r"[aeiou]+", word))
        return max(1, count)
    total_syllables = sum(syllables(w) for w in words)
    asl = len(words) / len(sentences)           # avg sentence length
    asw = total_syllables / len(words)          # avg syllables per word
    return 206.835 - 1.015 * asl - 84.6 * asw


def _vocab_richness(text: str) -> float:
    """Type-token ratio (unique words / total words). Range 0–1."""
    words = _tokenize(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _formality_proxy(text: str) -> float:
    """
    Simple formality proxy: ratio of long/Latinate words (≥7 chars)
    to all content words. Longer words ≈ more formal register.
    """
    content = _content_words(text)
    if not content:
        return 0.5
    long_words = sum(1 for w in content if len(w) >= 7)
    return min(1.0, long_words / len(content))


def _sentiment_proxy(text: str) -> float:
    """
    Very simple rule-based sentiment: ratio of positive-leaning words
    minus negative-leaning words, normalised to [-1, 1].
    Uses a tiny seed lexicon sufficient for brand copy.
    """
    positive = {
        "excellence","achieve","achievement","exceptional","extraordinary","remarkable",
        "enduring","precision","innovative","iconic","perfect","ultimate","award",
        "heritage","trust","superior","finest","premium","outstanding","dedicated",
        "passion","inspire","lead","legendary","timeless","craftsmanship","proud",
        "champion","victory","best","beautiful","elegant","luxury","prestigious",
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
        return 0.1  # brand copy tends slightly positive
    return (pos - neg) / total


def _top_keywords(texts: list, k: int = 10) -> list:
    counts = Counter()
    for t in texts:
        for w in _content_words(t):
            counts[w] += 1
    return [w for w, _ in counts.most_common(k)]


def _tone_label(mean_formality: float, mean_sentiment: float) -> str:
    """Map simple numeric proxies to a human tone label."""
    if mean_formality >= 0.55 and mean_sentiment >= 0.3:
        return "authoritative"
    if mean_formality >= 0.55:
        return "formal"
    if mean_sentiment >= 0.3:
        return "motivational"
    return "neutral"


# ── Main builder ──────────────────────────────────────────────────────────────

def build_brand_profiles(db_path: str = SQLITE_DB_PATH) -> None:
    """
    Read brand_texts from SQLite, compute per-brand profiles,
    write to brand_profiles table.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create brand_profiles table if it doesn't exist yet
    cur.execute("""
    CREATE TABLE IF NOT EXISTS brand_profiles (
        brand_id   TEXT PRIMARY KEY,
        brand_name TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        built_at   TEXT NOT NULL DEFAULT (datetime('now')),
        version    INTEGER NOT NULL DEFAULT 1,
        n_texts    INTEGER NOT NULL
    )
    """)
    conn.commit()

    # Load all texts
    cur.execute("SELECT brand_id, brand_name, text FROM brand_texts")
    rows = cur.fetchall()

    # Group by brand
    by_brand: dict = {}
    for brand_id, brand_name, text in rows:
        by_brand.setdefault((brand_id, brand_name), []).append(text or "")

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for (brand_id, brand_name), texts in by_brand.items():
        sentiments   = [_sentiment_proxy(t) for t in texts]
        flesches     = [_flesch_score(t) for t in texts]
        richnesses   = [_vocab_richness(t) for t in texts]
        formalities  = [_formality_proxy(t) for t in texts]

        mean_sentiment   = _mean(sentiments)
        std_sentiment    = max(_std(sentiments), 0.01)
        mean_flesch      = _mean(flesches)
        std_flesch       = max(_std(flesches), 0.01)
        mean_vocab       = _mean(richnesses)
        std_vocab        = max(_std(richnesses), 0.01)
        mean_formality   = _mean(formalities)
        std_formality    = max(_std(formalities), 0.01)
        keywords         = _top_keywords(texts, k=10)
        tone             = _tone_label(mean_formality, mean_sentiment)

        profile = {
            "brand_id":          brand_id,
            "brand_name":        brand_name,
            "mean_sentiment":    round(mean_sentiment, 4),
            "std_sentiment":     round(std_sentiment, 4),
            "mean_flesch":       round(mean_flesch, 2),
            "std_flesch":        round(std_flesch, 2),
            "mean_vocab_richness": round(mean_vocab, 4),
            "std_vocab_richness":  round(std_vocab, 4),
            "mean_formality":    round(mean_formality, 4),
            "std_formality":     round(std_formality, 4),
            "top_keywords":      keywords,
            "mean_embedding":    [],   # placeholder; upgraded when Person B's embeddings available
            "tone_label":        tone,
            "built_at":          built_at,
            "version":           1,
        }

        cur.execute("""
        INSERT OR REPLACE INTO brand_profiles
            (brand_id, brand_name, profile_json, built_at, version, n_texts)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (brand_id, brand_name, json.dumps(profile), built_at, 1, len(texts)))

    conn.commit()
    conn.close()
    print(f"[brand_profile_builder] Built profiles for {len(by_brand)} brands → {db_path}")


if __name__ == "__main__":
    build_brand_profiles()
