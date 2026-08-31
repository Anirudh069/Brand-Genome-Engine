# Evaluation Results — Scoring Layer (Person C)

Reproduce with:

```bash
python scripts/brand_attribution_eval.py --holdout 0.3 --seed 42
```

---

## 1. Does the scorer separate on-brand from off-brand copy?

Four fixed inputs, scored against the Rolex genome.

| Input | Previous version | Current | 
|---|---|---|
| Authentic Rolex copy | 51.0 | **76.2** |
| Casual off-brand copy | 45.0 | **25.8** |
| Unrelated subject (pizza delivery) | 45.1 | 46.6 |
| Brand-name keyword spam | 36.8 | 47.1 |
| **Separation, good vs. bad** | **6.0 points** | **50.4 points** |

The previous version scored deliberately off-brand copy within six points of
authentic brand copy, and scored an unrelated sentence the same as off-brand
copy. It was not distinguishing brand voice; it was returning a number.

## 2. Does one brand's genome actually differ from another's?

The same Rolex paragraph, unmodified, scored against every brand genome.

| Genome | Previous | Current |
|---|---|---|
| **Rolex** | **51.0** | **76.2** |
| Omega | 34.6 | 49.6 |
| TAG Heuer | — | 44.9 |
| Cartier | 46.0 | 37.3 |
| Tissot | 47.2 | 36.8 |
| **Margin over next-best** | **3.8 points** | **26.6 points** |

## 3. Blind attribution — the held-out experiment

The tests above use inputs chosen by the author. This one does not.

**Method.** Hold out 30% of every brand's texts. Rebuild all ten genomes from
the remaining 70% only, so the held-out copy has never been seen. Score each
held-out paragraph against all ten genomes and record whether the correct brand
ranks first. This is brand-level authorship attribution.

**Setup.** 10 brands · 387 usable texts (≥ 25 words) · 270 train / 117 held-out ·
seed 42 · random-guess baseline 10%.

| Metric | Result | Baseline |
|---|---|---|
| Top-1 accuracy | **31.6%** (37/117) | 10% |
| Top-3 accuracy | **54.7%** (64/117) | 30% |

Three times the baseline on top-1, and the correct brand appears in the top
three for more than half of unseen paragraphs.

### Per-brand accuracy

| Brand | Top-1 |
|---|---|
| Breitling | 69.2% |
| Audemars Piguet | 61.5% |
| Cartier | 53.8% |
| Tissot | 41.7% |
| IWC Schaffhausen | 20.0% |
| Hublot | 15.4% |
| Patek Philippe | 15.4% |
| Omega | 12.5% |
| Rolex | 8.3% |
| TAG Heuer | 0.0% |

The spread is itself a finding. Brands with a distinctive editorial register are
identified reliably; brands whose corpus is largely product specification and
generic company history are not, because those texts contain little voice to
detect. This is a property of the source data, not of the scorer — the corpus
was scraped from public brand sites and its composition varies by brand.

### Ablation — which signal carries the attribution?

Same held-out texts, ranked by each metric alone.

| Ranking signal | Top-1 accuracy |
|---|---|
| Vocabulary only | **47.0%** |
| Vocabulary 0.6 + tone 0.4 | 39.3% |
| Overall score (production weights) | 31.6% |
| Tone only | 23.1% |
| Readability only | 17.9% |
| Sentiment only | 8.5% |
| *Random baseline* | *10.0%* |

Two things follow.

**Vocabulary is the strongest identifier**, at nearly five times baseline. What a
brand chooses to talk about is the most brand-specific signal available.

**The production score is deliberately not the best attributor**, and that is
correct. `overall_score` answers *"is this on-brand?"*, which is not the same
question as *"which brand wrote this?"*. Judging on-brand-ness has to account
for register and reading level — dimensions every luxury watch brand shares, and
which therefore carry little identifying information. Optimising the production
weights for attribution accuracy would improve this table and make the product
worse.

Sentiment alone sits at baseline, which is the honest reading: emotional pitch
is close to uniform across ten luxury watch brands. It earns its 0.25 weight by
catching copy that is *wrong* for a brand, not by telling brands apart.

---

## 4. Limitations

- The tone metric currently uses a style proxy (formality, lexical variety,
  sentence length). Person B's 384-dimension embeddings are stored in each
  profile and the cosine path is implemented and tested; it activates when
  embeddings are computed for input text at scoring time.
- Sentiment is lexicon-based and does not handle irony.
- Corpus composition varies by brand, which caps attribution accuracy for
  brands whose scraped texts are mostly specification tables.
- 117 held-out samples is a small evaluation set; the per-brand figures carry
  wide error margins and should be read as indicative.
