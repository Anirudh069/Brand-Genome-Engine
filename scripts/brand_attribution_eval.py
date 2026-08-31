#!/usr/bin/env python3
"""
scripts/brand_attribution_eval.py — Person C evaluation harness.

THE EXPERIMENT
--------------
Hold out a portion of every brand's texts. Rebuild the brand genomes from the
remaining texts only, so the held-out copy has never been seen. Then, for each
held-out paragraph, score it against ALL brand genomes and ask: does the
correct brand come top?

This is brand-level authorship attribution — the same shape as the classic
stylometry problem of identifying an author from their writing style. Random
guessing across ten brands scores 10%.

It matters because it is the difference between "we built a scorer" and "we
measured whether the scorer works". The scores in a demo are chosen by the
demonstrator; these are not.

Usage
-----
    python scripts/brand_attribution_eval.py
    python scripts/brand_attribution_eval.py --holdout 0.3 --seed 42
    python scripts/brand_attribution_eval.py --json results.json

Output
------
  * top-1 and top-3 accuracy
  * per-brand accuracy
  * a confusion matrix showing which brands get mistaken for which
"""

import argparse
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.profiles.brand_profile_builder import build_brand_profiles, MIN_TEXTS_PER_BRAND
from src.scoring.consistency_scorer import score_consistency, extract_text_features

MIN_WORDS_FOR_EVAL = 25   # below this, style statistics are too noisy to attribute


def load_texts(db_path):
    """Return {brand_id: [(text_id, brand_name, text), ...]}."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT text_id, brand_id, brand_name, text FROM brand_texts"
    ).fetchall()
    conn.close()

    by_brand = defaultdict(list)
    for text_id, brand_id, brand_name, text in rows:
        if text and len(text.split()) >= MIN_WORDS_FOR_EVAL:
            by_brand[brand_id].append((text_id, brand_name, text))
    return by_brand


def split(by_brand, holdout, rng):
    """Split each brand's texts into (train_ids, test_items)."""
    train_ids, test_items = [], []
    for brand_id, items in by_brand.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * holdout)))
        # Never starve a brand below the minimum needed for a usable profile.
        n_test = min(n_test, max(0, len(shuffled) - MIN_TEXTS_PER_BRAND))
        test_items.extend((brand_id, t) for t in shuffled[:n_test])
        train_ids.extend(t[0] for t in shuffled[n_test:])
    return train_ids, test_items


def build_holdout_db(source_db, train_ids):
    """Copy the database, keep only training texts, rebuild profiles."""
    tmpdir = tempfile.mkdtemp(prefix="bge_eval_")
    path = os.path.join(tmpdir, "brand_data.db")
    shutil.copy(source_db, path)

    conn = sqlite3.connect(path)
    keep = set(train_ids)
    all_ids = [r[0] for r in conn.execute("SELECT text_id FROM brand_texts")]
    drop = [(i,) for i in all_ids if i not in keep]
    conn.executemany("DELETE FROM brand_texts WHERE text_id = ?", drop)
    conn.commit()
    conn.close()

    build_brand_profiles(path, verbose=False)
    return tmpdir, path


def load_profiles(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT brand_id, profile_json FROM brand_profiles").fetchall()
    conn.close()
    return {r[0]: json.loads(r[1]) for r in rows}


def rank_brands(text, profiles):
    """Score `text` against every profile, best first."""
    features = extract_text_features(text)
    ranked = []
    for brand_id, profile in profiles.items():
        try:
            ranked.append((brand_id, score_consistency(features, profile).overall_score))
        except Exception:
            ranked.append((brand_id, 0.0))
    ranked.sort(key=lambda r: r[1], reverse=True)
    return ranked


# Ranking signals compared in the ablation below.
#
# The production overall_score answers "is this on-brand?", which is not the same
# question as "which brand wrote this?". Attribution rewards whatever is most
# brand-specific; on-brand judgement also has to care about register and reading
# level, which every luxury watch brand shares. Reporting both is the honest way
# to describe what each metric is good for — and every signal individually
# beating the 10% baseline is evidence that none of the five is dead weight.
RANKING_SIGNALS = {
    "overall (production)": lambda r: r.overall_score,
    "vocabulary only": lambda r: r.vocab_overlap_pct,
    "vocab 0.6 + tone 0.4": lambda r: 0.6 * r.vocab_overlap_pct + 0.4 * r.tone_pct,
    "tone only": lambda r: r.tone_pct,
    "sentiment only": lambda r: r.sentiment_alignment_pct,
    "readability only": lambda r: r.readability_match_pct,
}


def ablation(test_items, profiles, n_brands):
    """Top-1 accuracy for each ranking signal, on the same held-out texts."""
    results = {name: [0, 0] for name in RANKING_SIGNALS}
    for true_brand, (_, _, text) in test_items:
        features = extract_text_features(text)
        scored = {}
        for brand_id, profile in profiles.items():
            try:
                scored[brand_id] = score_consistency(features, profile)
            except Exception:
                continue
        if not scored:
            continue
        for name, key_fn in RANKING_SIGNALS.items():
            ranked = sorted(scored.items(), key=lambda kv: key_fn(kv[1]), reverse=True)
            results[name][1] += 1
            if ranked[0][0] == true_brand:
                results[name][0] += 1
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/brand_data.db")
    parser.add_argument("--holdout", type=float, default=0.3,
                        help="fraction of each brand's texts held out (default 0.3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", help="write full results to this path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_brand = load_texts(args.db)
    brands = sorted(by_brand)

    print("=" * 76)
    print("BRAND ATTRIBUTION EVALUATION")
    print("=" * 76)
    print(f"  brands              : {len(brands)}")
    print(f"  usable texts        : {sum(len(v) for v in by_brand.values())} "
          f"(>= {MIN_WORDS_FOR_EVAL} words)")
    print(f"  holdout             : {args.holdout:.0%}   seed: {args.seed}")

    train_ids, test_items = split(by_brand, args.holdout, rng)
    print(f"  training texts      : {len(train_ids)}")
    print(f"  held-out texts      : {len(test_items)}")
    print(f"  random-guess baseline: {100.0 / len(brands):.1f}%")

    tmpdir, holdout_db = build_holdout_db(args.db, train_ids)
    try:
        profiles = load_profiles(holdout_db)
        profiles = {b: p for b, p in profiles.items() if b in brands}

        top1 = top3 = 0
        per_brand = defaultdict(lambda: [0, 0])          # brand -> [correct, total]
        confusion = defaultdict(lambda: defaultdict(int))  # true -> predicted -> n

        for true_brand, (_, _, text) in test_items:
            ranked = rank_brands(text, profiles)
            predicted = ranked[0][0]
            per_brand[true_brand][1] += 1
            confusion[true_brand][predicted] += 1
            if predicted == true_brand:
                top1 += 1
                per_brand[true_brand][0] += 1
            if true_brand in [b for b, _ in ranked[:3]]:
                top3 += 1

        n = max(1, len(test_items))
        print()
        print("-" * 76)
        print(f"  TOP-1 ACCURACY : {top1}/{n} = {100.0 * top1 / n:.1f}%")
        print(f"  TOP-3 ACCURACY : {top3}/{n} = {100.0 * top3 / n:.1f}%")
        print(f"  (random guessing would be {100.0 / len(brands):.1f}% / "
              f"{300.0 / len(brands):.1f}%)")
        print("-" * 76)

        print("\n  PER-BRAND TOP-1 ACCURACY")
        for brand in brands:
            correct, total = per_brand[brand]
            if not total:
                continue
            pct = 100.0 * correct / total
            bar = "#" * int(pct / 4)
            print(f"    {brand:16s} {correct:3d}/{total:<3d} {pct:5.1f}%  {bar}")

        print("\n  CONFUSION MATRIX  (rows = true brand, columns = predicted)")
        short = {b: b[:6] for b in brands}
        print("    " + " " * 16 + "".join(f"{short[b]:>7s}" for b in brands))
        for true_brand in brands:
            row = "".join(
                f"{confusion[true_brand][pred] or '.':>7}" for pred in brands)
            print(f"    {true_brand:16s}{row}")

        print("\n  Read the diagonal: those are correct attributions. Off-diagonal")
        print("  clusters show which brands genuinely sound alike.")

        abl = ablation(test_items, profiles, len(brands))
        print("\n  ABLATION — top-1 accuracy by ranking signal")
        for name, (correct, total) in abl.items():
            if not total:
                continue
            pct = 100.0 * correct / total
            bar = "#" * int(pct / 3)
            print(f"    {name:24s} {correct:3d}/{total:<4d} {pct:5.1f}%  {bar}")
        print(f"    {'random baseline':24s} {'':9s}{100.0 / len(brands):5.1f}%")

        if args.json:
            with open(args.json, "w") as fh:
                json.dump({
                    "top1_accuracy": top1 / n,
                    "top3_accuracy": top3 / n,
                    "n_test": len(test_items),
                    "n_train": len(train_ids),
                    "holdout": args.holdout,
                    "seed": args.seed,
                    "baseline": 1.0 / len(brands),
                    "per_brand": {b: dict(zip(("correct", "total"), per_brand[b]))
                                  for b in brands},
                    "confusion": {t: dict(confusion[t]) for t in brands},
                }, fh, indent=2)
            print(f"\n  Full results written to {args.json}")
        print()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
