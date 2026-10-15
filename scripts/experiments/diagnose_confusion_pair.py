"""
diagnose_confusion_pair.py — investigates ONE specific confusion pair
(default: qa misclassified as data-engineer, 25.3% of qa instances per
the confusion_matrix_full.csv breakdown) at the level of ACTUAL skill
content, not just aggregate counts.

WHY THIS EXISTS: check_class_stability.py and the confusion-matrix
diagnostic in train_role_classifier.py both tell you THAT qa is
frequently confused with data-engineer and roughly HOW MUCH — but not
WHY. This script answers why, two ways:

  1. DESCRIPTIVE: what are the most common skill_tags in each class,
     and how much do they overlap? A high overlap in generic tags
     (python, sql, etl) versus a near-absence of QA-specific tags
     (selenium, pytest, test-automation) in qa's own top tags would be
     a concrete, checkable explanation.

  2. ROW-LEVEL: of the qa rows that ACTUALLY got misclassified as
     data-engineer in cross-validation, what do their skill_tags
     actually look like — and how do they differ from the qa rows the
     model got RIGHT? This is the part an aggregate confusion count
     can never show you: whether misclassified rows are a distinct,
     identifiable subgroup (e.g. "QA rows that happen to list SQL/ETL
     skills and nothing QA-specific") or just random noise.

IMPORTANT: this script reuses load_and_prepare / build_pool / make_model
/ compute_sqrt_balanced_weights from train_role_classifier.py directly —
it does NOT reimplement feature engineering, to guarantee this analysis
reflects the EXACT same model as production, not a subtly different one.

USAGE
  python -m scripts.experiments.diagnose_confusion_pair
  python -m scripts.experiments.diagnose_confusion_pair --true qa --confused-as data-engineer
"""
import argparse
import ast
from collections import Counter

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from training.train_catboost import build_pool, make_model
from careercompass.ml.common import (
    DATA_CSV, load_and_prepare, compute_sqrt_balanced_weights, TARGET,
)

N_FOLDS = 3
SEED = 42
TOP_N_TAGS = 15


def top_tags(df: pd.DataFrame, role_slug: str, n: int = TOP_N_TAGS) -> list:
    """Most common skill_tags across every row of one class, with the
    fraction of that class's rows each tag appears in (not just raw
    count) — a tag in 80% of qa rows is a much stronger signal than one
    in 80% of a class with only 20 rows total."""
    rows = df[df[TARGET] == role_slug]
    counter = Counter()
    for tags in rows["skill_tags_parsed"]:
        counter.update(set(tags))  # set(): count PRESENCE per row, not repeats within a row
    total = len(rows)
    return [(tag, cnt, cnt / total) for tag, cnt in counter.most_common(n)]


def print_tag_table(label: str, tags: list):
    print(f"\n  Top {len(tags)} tags in '{label}':")
    for tag, cnt, frac in tags:
        print(f"    {tag:30s} {cnt:4d} rows  ({frac:.0%})")


def get_misclassified_rows(df: pd.DataFrame, true_class: str, confused_as: str) -> tuple:
    """
    Runs the SAME 3-fold CV as train_role_classifier.py's
    run_cross_validation(), but keeps out-of-fold predictions attached
    to the ORIGINAL dataframe rows (by index) so we can go back and
    inspect the actual skill_tags of specific misclassified rows —
    something the aggregate confusion matrix can't give you.

    Returns (misclassified_rows_df, correctly_classified_rows_df) —
    both subsets of the true_class's rows, split by whether THIS
    specific fold-run predicted them as confused_as or correctly.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = pd.Series(index=df.index, dtype=object)

    for train_idx, val_idx in skf.split(df, df[TARGET]):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        weights = compute_sqrt_balanced_weights(train_df)
        model = make_model(class_weights=weights)
        model.fit(build_pool(train_df))
        preds = model.predict(build_pool(val_df)).flatten()
        oof_preds.iloc[val_idx] = preds

    true_rows = df[df[TARGET] == true_class].copy()
    true_rows["predicted"] = oof_preds.loc[true_rows.index]

    misclassified = true_rows[true_rows["predicted"] == confused_as]
    correct = true_rows[true_rows["predicted"] == true_class]
    return misclassified, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", default="qa", help="the true class being confused (default: qa)")
    ap.add_argument("--confused-as", default="data-engineer", help="what it's being predicted as (default: data-engineer)")
    args = ap.parse_args()

    print(f"Loading {DATA_CSV}...")
    df = load_and_prepare(DATA_CSV)
    df["skill_tags_parsed"] = df["skill_tags"].apply(ast.literal_eval)
    print(f"{len(df)} rows, {df[TARGET].nunique()} classes\n")

    print("=" * 70)
    print(f"PART 1 — DESCRIPTIVE: top skill_tags in '{args.true}' vs '{args.confused_as}'")
    print("=" * 70)
    true_tags = top_tags(df, args.true)
    confused_tags = top_tags(df, args.confused_as)
    print_tag_table(args.true, true_tags)
    print_tag_table(args.confused_as, confused_tags)

    true_tag_set = {t for t, _, _ in true_tags}
    confused_tag_set = {t for t, _, _ in confused_tags}
    overlap = true_tag_set & confused_tag_set
    print(f"\n  Overlap in TOP {TOP_N_TAGS} tags between the two classes: {sorted(overlap) if overlap else 'none'}")
    print(f"  ({len(overlap)}/{TOP_N_TAGS} of {args.true}'s top tags also appear in {args.confused_as}'s top tags)")

    print("\n" + "=" * 70)
    print(f"PART 2 — ROW-LEVEL: which specific '{args.true}' rows get predicted as '{args.confused_as}'?")
    print("=" * 70)
    print("(This re-runs 3-fold CV once to get out-of-fold predictions per row — takes a moment)\n")
    misclassified, correct = get_misclassified_rows(df, args.true, args.confused_as)

    print(f"  {len(misclassified)} of {len(misclassified) + len(correct)} '{args.true}' rows in this run "
          f"were misclassified as '{args.confused_as}' "
          f"({len(misclassified) / (len(misclassified) + len(correct)):.1%})")

    if len(misclassified):
        print(f"\n  Skill tags of MISCLASSIFIED '{args.true}' rows (predicted as '{args.confused_as}'):")
        for _, row in misclassified.head(10).iterrows():
            print(f"    {row['skill_tags_parsed']}")

        misclassified_tag_counter = Counter()
        for tags in misclassified["skill_tags_parsed"]:
            misclassified_tag_counter.update(set(tags))
        print(f"\n  Most common tags AMONG MISCLASSIFIED rows specifically:")
        for tag, cnt in misclassified_tag_counter.most_common(10):
            print(f"    {tag:30s} {cnt} of {len(misclassified)} misclassified rows")

    if len(correct):
        correct_tag_counter = Counter()
        for tags in correct["skill_tags_parsed"]:
            correct_tag_counter.update(set(tags))
        print(f"\n  Most common tags AMONG CORRECTLY-CLASSIFIED '{args.true}' rows (for comparison):")
        for tag, cnt in correct_tag_counter.most_common(10):
            print(f"    {tag:30s} {cnt} of {len(correct)} correct rows")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print(f"  - If misclassified rows are missing tags that correctly-classified rows")
    print(f"    commonly have (e.g. testing-specific tools), that's actionable: add/")
    print(f"    emphasize those distinguishing tags in future {args.true} training data.")
    print(f"  - If misclassified rows look SIMILAR to correctly-classified ones (no")
    print(f"    clear tag-level difference), the confusion may be genuinely ambiguous")
    print(f"    at the skill-tag level — a feature-engineering limitation, not a data")
    print(f"    volume one, and may need a different feature entirely to resolve.")


if __name__ == "__main__":
    main()