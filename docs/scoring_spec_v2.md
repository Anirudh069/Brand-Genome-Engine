# Scoring Specification — Person C

**Project:** Brand Genome Engine (Tabula Rasa)
**Owner:** Person C — profiles, scoring, diagnostics, drift, edit plan
**Version:** 2.0
**Supersedes:** 1.0 (2026-03-01)

---

## 1. What this module does

Person A stores the brand corpus. Person B turns text into features and
embeddings. Person D builds the interface. This module is the judgement layer
in between: it decides what "on-brand" means numerically, explains its verdict
in words a writer can act on, and turns that explanation into instructions for
the rewrite model.

Four things are produced:

| Output | Function | Consumed by |
|---|---|---|
| Brand profile (the "genome") | `build_brand_profiles()` | scorer, `/api/profile/rebuild` |
| Five consistency scores | `score_consistency()` | `/api/check-consistency`, score card |
| Drift report | `generate_drift_report()` | `/api/rewrite`, drift panel |
| Edit plan | `generate_edit_plan()` | the LLM prompt inside `/api/rewrite` |

---

## 2. Frozen contracts

These names are fixed. Changing one breaks Person D's UI and requires written
agreement from both owners.

```python
@dataclass
class ScoreResult:
    overall_score: float            # 0–100
    tone_pct: float                 # 0–100
    vocab_overlap_pct: float        # 0–100
    sentiment_alignment_pct: float  # 0–100
    readability_match_pct: float    # 0–100

@dataclass
class DriftReport:
    brand_id: str
    drift_flags: list[str]
    sentiment_delta: float
    readability_delta: float
    missing_keywords: list[str]
    excess_keywords: list[str]
    summary: str

@dataclass
class EditPlan:
    brand_id: str
    goals: list[str]
    avoid_terms: list[str]
    prefer_terms: list[str]
    style_rules: list[str]
    tone_direction: str
    grounding_chunks: list[str]
```

---

## 3. The brand profile

`brand_profile_builder.py` reads `brand_texts` and writes one row per brand to
`brand_profiles`. `profile_json` holds:

| Field | Meaning |
|---|---|
| `mean_sentiment`, `std_sentiment` | emotional register, −1 to 1 |
| `mean_flesch`, `std_flesch` | Flesch Reading Ease |
| `mean_vocab_richness`, `std_vocab_richness` | type-token ratio over a 100-word window |
| `mean_formality`, `std_formality` | formality register, 0 to 1 |
| `mean_sentence_length`, `std_sentence_length` | words per sentence |
| `top_keywords` | 15 characteristic words |
| `common_vocab` | 400 most-used words — diagnostics only, never scored |
| `brand_name_tokens` | name tokens, excluded from every score |
| `mean_embedding` | populated when Person B's embeddings arrive; `[]` until then |
| `tone_label` | human label derived from formality + sentiment |
| `version`, `built_at`, `n_texts` | provenance |

A brand with fewer than **10 texts** is skipped with a warning rather than
given an unreliable profile.

### 3.1 Characteristic vocabulary

Selected as **10 words by TF-IDF across brands + 5 by raw frequency**:

```
tf_sublinear = 1 + log(count of word in this brand)
idf          = log(n_brands / n_brands containing the word)
score        = tf_sublinear × (idf + 0.5)
```

Rationale: raw frequency alone answers "what does this brand talk about most",
which for ten watch brands is *watch, time, case* — words that describe the
category, not the brand. TF-IDF alone drifts to rare niche vocabulary and
misses the words a brand repeats constantly. The blend describes a brand the
way a tone-of-voice guideline does. The frequency tier additionally skips any
word present in **every** brand's corpus, so category nouns cannot enter
through the back door.

Three exclusions, all data-driven rather than hardcoded lists:

1. **Proper nouns** — a word capitalised in more than 60% of its non-sentence-initial
   occurrences. Removes brand and product names (*Rolex, Oyster, Perpetual*).
2. **Specification-sheet words** — a word appearing within three tokens of a
   number in more than 40% of its occurrences. Removes *approx, thickness,
   carats* where a brand's corpus contains product tables.
3. **Words appearing fewer than three times** — noise, not a pattern.

### 3.2 Feature definitions

**Flesch Reading Ease** — `206.835 − 1.015 × (words/sentences) − 84.6 × (syllables/words)`.
Syllables counted as vowel groups, minimum one. Empty input returns 50.

**Vocabulary richness** — type-token ratio over the first 100 words. The window
is required: plain TTR falls as text lengthens, so a 500-word page and a
50-word caption are otherwise not comparable.

**Formality** — `0.6 × long_word_ratio + 0.2 × contraction_score + 0.2 × person_score`,
where long words are ≥ 7 characters, and the two scores fall as contractions
and second-person address rise. The 0.6/0.2/0.2 weighting is deliberate: most
copy contains no contractions and no "you", so those components sit at 1.0
nearly always and an equal average compressed every text into 0.67–0.85.

**Sentiment** — `tanh((positive_hits − negative_hits) / sqrt(content_word_count))`.

The lexicon covers **emotional valence only**. Brand-value words — *precision,
heritage, craft, quality, innovation* — are deliberately excluded: they are
topical, and the vocabulary metric already measures them. When they sat in the
sentiment lexicon the two metrics double-counted the same words, and a
well-written luxury paragraph registered as far more positive than the brand's
own corpus and scored 0.5% on sentiment alignment.

---

## 4. The five scores

### 4.1 Vocabulary overlap — `vocab_overlap_pct`

```
hits     = |unique content words of text ∩ top_keywords|
expected = min(|top_keywords|, max(5, unique_content_words / 3))
coverage = min(1, hits / expected) × 100
```

Brand-name tokens are stripped from the text before comparison.

**Why not Jaccard.** Version 1.0 specified `|A ∩ B| / |A ∪ B|`. The two sets are
different sizes by design — a profile holds 15 words, an input may hold 200 —
and Jaccard divides by the union, so a text using *every* brand keyword
perfectly still scored about 7%. Coverage asks the question a brand manager
asks: how much of our vocabulary is in this copy? The `expected` denominator
scales with length so a 30-word caption is not held to the same absolute hit
count as a 300-word page, and floors at 5 so a very short text cannot reach
100% on two lucky words.

Edge cases: empty text or empty keyword list → 0 (an empty match is not a match).

### 4.2 Sentiment alignment — `sentiment_alignment_pct`

```
σ = max(std_sentiment, 0.15)
alignment = exp(−((s − μ)² / (2σ²))) × 100
```

Returns 100 when the text's sentiment sits exactly on the brand mean, decaying
with distance and scaled by the brand's own variability.

The 0.15 floor on σ exists because a Gaussian falls to ~1% at three standard
deviations: a brand whose corpus happens to be emotionally uniform would
otherwise score almost every real text at zero, turning a measure of alignment
into a near-binary gate.

Edge cases: NaN or non-numeric sentiment → `FeatureExtractionError`.

### 4.3 Readability match — `readability_match_pct`

```
tolerance = max(2 × std_flesch, 15)
match = max(0, 1 − |f − μ_f| / tolerance) × 100
```

A brand that varies its reading level gets a wider tolerance band. The floor of
15 Flesch points stops a very consistent brand from becoming impossible to match.

Edge cases: text with no sentences or no words → Flesch defaults to 50.

### 4.4 Tone — `tone_pct`

**With embeddings** (once Person B's `mean_embedding` is populated):

```
tone = max(0, cosine_similarity(e_text, e_brand_mean)) × 100
```

**Without embeddings** — agreement on three register dimensions, each a
Gaussian similarity against the brand's own mean and spread:

```
tone = 0.50 × G(formality) + 0.30 × G(vocab_richness) + 0.20 × G(sentence_length)
```

**Why this fallback changed.** Version 1.0 used
`tone = 1 − |sentiment − brand_mean_sentiment|`, which is the sentiment metric a
second time. Because tone carries the heaviest weight (0.30) and sentiment
carries 0.25, that placed 55% of the overall score on a single lexicon count —
and since `mean_embedding` is empty until Person B's pipeline lands, the
fallback was the only path ever executed. A sentence about pizza delivery
scored 68 on tone against the Rolex profile. Formality, lexical variety and
sentence length are what actually separate *"This timepiece embodies enduring
precision"* from *"this watch is super easy to wear"*, and none of them
duplicate a signal measured elsewhere.

Edge cases: zero vector → 0. Dimension mismatch → `EmbeddingDimensionError`.

### 4.5 Overall score and weight presets

```
overall = 0.30 × tone
        + 0.25 × sentiment_alignment
        + 0.25 × vocab_overlap
        + 0.20 × readability_match
```

| Metric | Weight | Rationale |
|---|---|---|
| Tone | 0.30 | Register is the most recognisable part of a brand voice and the hardest to fake |
| Sentiment | 0.25 | Emotional pitch is what readers notice first when copy is off |
| Vocabulary | 0.25 | Shared terminology is the most objective, most checkable signal |
| Readability | 0.20 | Reading level matters but varies legitimately across formats |

Clamped to [0, 100].

#### Preset weight profiles

Not every brand weighs the same things, so the blend is switchable. Every preset
sums to 1.0, and the four sub-scores are identical across presets — only
`overall_score` changes, so the breakdown a user sees never depends on a
dropdown.

| Preset | Tone | Sentiment | Vocabulary | Readability |
|---|---|---|---|---|
| `balanced` (default) | 0.30 | 0.25 | 0.25 | 0.20 |
| `tone_heavy` | 0.45 | 0.25 | 0.15 | 0.15 |
| `semantic_heavy` | 0.30 | 0.15 | 0.40 | 0.15 |

`tone_heavy` suits a brand whose identity is how it speaks rather than what it
says — a distinctive voice survives a change of subject, so vocabulary matters
less. `semantic_heavy` suits a brand built on owned terminology, where using a
competitor's word for something is the real error.

A preset can be passed per call, or stored on the genome as `weight_preset` so a
brand keeps its weighting without every caller having to remember it. A custom
weight dict is accepted and normalised to sum to 1.0; an unknown preset name
falls back to `balanced` rather than raising, because a bad value from the UI
must not take the endpoint down.

---

## 5. Neutral signals

Brand-name mentions are **counted and reported, never scored**. Name tokens are
removed from the text before the vocabulary metric runs, and they are excluded
from `top_keywords` at build time by the proper-noun filter. Without both
guards, repeating the brand name is the cheapest possible route to a high
score — in version 1.0 the string *"Rolex Rolex Rolex watch oyster perpetual…"*
scored 82/100, higher than any real sentence.

---

## 5b. The user's brand genome

The competitor genomes are built from 45 texts each. The user's genome is built
from their mission plus exactly 7 snippets — 8 short texts — by
`build_user_genome()`.

Everything is measured from the submitted text: keywords, formality, sentiment,
readability, lexical variety and sentence length. The user's declared tone is
kept as `declared_tone` for display but never used as a measurement.

Two things differ from the competitor path:

**Keyword selection uses the competitor corpus as its reference.** A word the
user repeats that no competitor uses is highly characteristic; a word every
watch brand uses is not. Within the user's own text, words used more than once
rank ahead of words used once — with only 8 short texts every term is rare, so
raw TF-IDF would promote whatever happens to be unusual, while recurrence is a
better signal of intent. A word used once is a word; a word used twice is a
choice.

**Every standard deviation carries a floor**, larger than the competitor path
uses:

| Dimension | Floor |
|---|---|
| sentiment | 0.18 |
| Flesch | 12.0 |
| vocabulary richness | 0.08 |
| formality | 0.10 |
| sentence length | 4.0 |

With 8 samples, a narrow spread is far more likely to be an accident of the
sample than a fact about the brand. A genome that trusted it would reject
perfectly good copy. The floors make the user's genome forgiving exactly where
the evidence is thin.

Fewer than 3 texts raises `ValueError`; the API returns
`error: "insufficient_text"` rather than building a genome it cannot justify.

---

## 6. Diagnostics

`build_diagnostics()` returns:

| Field | Definition |
|---|---|
| `aligned_terms` | brand keywords present in the text |
| `missing_terms` | brand keywords absent (top 8) |
| `off_brand_terms` | prominent text words absent from `common_vocab` **and** `top_keywords` |
| `pillar_coverage` | per pillar, `min(1, hits/3)` |
| `missing_pillar_terms` | pillars with zero coverage |
| `name_mentions` | count, neutral |

`off_brand_terms` is checked against the brand's whole 400-word vocabulary, not
just the top 15. Otherwise ordinary words the brand writes constantly are
reported as off-brand and land in the edit plan's avoid list — telling the LLM
to stop using the brand's own vocabulary.

### 6.1 Messaging pillars

Pillars are fixed by the brief: Sustainability, Precision, Heritage, Value,
Innovation. Their keyword sets are **derived per brand from the corpus**, not
hardcoded. Seed terms are kept only if the brand actually uses them, then
expanded with words that co-occur with a seed at a lift of ≥ 1.3 over their
base rate. Words appearing in more than 22% of the brand's texts are excluded
as ubiquitous. `derive_pillar_keywords(overrides=...)` accepts Person B's
auto-derived sets, which take precedence.

---

## 7. Drift report

Deltas are signed, computed as `input − brand_mean`.

A flag fires when a delta exceeds `max(absolute_floor, 0.75 × brand_std)`:

| Flag | Floor |
|---|---|
| `sentiment_too_positive` / `sentiment_too_negative` | 0.25 |
| `readability_too_high` / `readability_too_low` | 10.0 Flesch points |
| `tone_too_casual` / `tone_too_formal` | 0.07 formality points |
| `sentences_too_short` / `sentences_too_long` | 5.0 words |
| `missing_brand_keywords` | fewer than 2 brand terms present |
| `missing_pillar_coverage` | 3 or more pillars untouched |

The relative component matters: a brand that writes consistently should flag
smaller deviations than one that ranges widely. The absolute floors stop any
brand flagging noise.

`missing_brand_keywords` fires on genuine absence rather than on a non-empty
`missing_terms` list — no short piece of copy uses all fifteen brand words, so
the latter would fire every time and mean nothing.

---

## 8. Edit plan

Each drift flag maps to a set of imperative style rules. Goals name the current
value and the target, so the rewrite has something to aim at and the before/after
comparison has something to verify. `avoid_terms` comes from `excess_keywords`,
`prefer_terms` from `missing_keywords`, and the two can never intersect.

`grounding_chunks` are three excerpts retrieved from `brand_chunks`, ranked

```
0.6 × topical overlap with the input + 0.4 × brand-vocabulary density
```

and de-duplicated at 0.6 Jaccard so the model sees three different examples.
Relevance outweighs exemplarity because an example about a different product
teaches the model the wrong content.

Retrieval is pluggable: `retrieve_grounding_chunks(..., retriever=fn)` uses
Person B's FAISS retrieval when supplied, with the contract
`retriever(query_text, brand_id, k) -> list[str]`. The lexical ranking above is
the fallback so the pipeline runs before the index exists. A retriever that
raises falls through to the fallback rather than failing the request.

`EditPlan.to_prompt()` renders the plan as the LLM prompt, so prompt wording
lives beside the logic that decided what belongs in it.

---

## 9. Before / after scoring

`score_before_after()` runs the identical scorer over the original and the
rewritten text. Using one function for both halves is the entire point: the
numbers are only comparable because nothing about the measurement changed
between them. Both are written to `analysis_history` in a single row.

---

## 10. analysis_history logging contract

One row per run. `diagnostics_json` holds the full payload so a past run can be
re-displayed without recomputation; `pre_score` and `post_score` stay flat and
indexable for the counters.

```json
{
  "schema_version": 1,
  "event_type": "consistency" | "rewrite" | "benchmark",
  "brand_id": "rolex",
  "scores": { "before": { ...ScoreResult }, "after": { ...ScoreResult } | null },
  "diagnostics": { ...Diagnostics } | null,
  "drift_report": { ...DriftReport } | null,
  "edit_plan":   { ...EditPlan }   | null,
  "extra": {},
  "logged_at": "2026-08-30T10:16:37Z"
}
```

Every key is always present; a value is `null` when it does not apply, so the
caller never has to check which event type produced a row.

Counters:

```sql
-- copies_analysed
SELECT COUNT(*) FROM analysis_history WHERE event_type IN ('consistency','rewrite');

-- avg_consistency
SELECT AVG(COALESCE(post_score, pre_score)) FROM analysis_history
WHERE COALESCE(post_score, pre_score) IS NOT NULL;

-- deviations_fixed
SELECT COUNT(*) FROM analysis_history
WHERE event_type='rewrite' AND post_score IS NOT NULL
  AND pre_score IS NOT NULL AND post_score > pre_score;
```

`init_history_table()` adds any missing columns to an existing
`analysis_history` rather than creating a competing table, so Person A's
earlier schema is migrated in place and her rows are preserved.

A logging failure is swallowed and reported as `-1`. A user losing their
rewrite because an analytics insert failed is the worse outcome.

---

## 11. Error handling

| Situation | Behaviour |
|---|---|
| Text under 10 words | all scores 0; API returns `error: "text_too_short"` |
| Brand has no profile row | `BrandProfileNotFoundError`; API returns `error: "brand_not_found"` |
| Profile exists but is unusable | API returns `error: "genome_not_initialised"` |
| Embedding dimension mismatch | `EmbeddingDimensionError` |
| NaN, infinity or non-numeric feature | `FeatureExtractionError` naming the field |
| `std_sentiment` below 0.15 | floored to 0.15 |
| `std_flesch` small | tolerance floored at 15 points |
| Empty `mean_embedding` | style-based tone fallback (§4.4) |
| Brand has no chunks | `grounding_chunks` returns `[]`; callers must handle it |
| Analytics write fails | logged as a warning, request unaffected |

The scorer and the profile builder import the **same** feature functions. When
each module kept its own copy of the Flesch and sentiment functions they
drifted, and a text was being compared against a mean computed by different
maths.

---

## 12. Testing

`tests/test_scoring.py` — 15 unit tests, one or more per metric plus an
integration test. Verifies the formulas compute what this document says.

`tests/test_scoring_behaviour.py` — 27 behavioural tests run against a real
copy of the brand database with profiles built by the real builder. Verifies
the system reaches the right conclusion:

- authentic brand copy outscores casual off-brand copy
- authentic brand copy outscores unrelated text
- keyword spam does not beat a real sentence
- repeating the brand name never raises a score
- Rolex copy scores highest against the Rolex genome, ahead of all nine competitors
- tone and sentiment do not track each other
- off-brand terms are genuinely foreign to the brand
- a real rewrite raises the overall score
- every frozen API field is present in every response shape

The split is deliberate. Version 1.0 passed all 15 unit tests while scoring
authentic Rolex copy at 27/100 and keyword spam at 82/100, because the tests
used a hand-written profile whose keywords and standard deviations bore no
resemblance to the ones the builder produces. Unit tests prove the arithmetic;
only behavioural tests on real data prove the judgement.

---

*All field names, weights, thresholds and error types in this document match
the implementation exactly. Any change requires written agreement from Person C
and Person D.*
