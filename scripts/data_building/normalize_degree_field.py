"""
normalize_degree_field.py — fixes inconsistent degree_field formatting
found in tier3_expanded_v2.csv's output: mixed case and mixed separator
style ("computer science" vs "computer-science" vs "Business Administration")
that CatBoost sees as DISTINCT categorical values, not one, silently
fragmenting whatever signal degree_field carries.

ROOT CAUSE: tier3_expand.py's AUGMENTATION_PROMPT asked for degree_field
as "short field name" without specifying a format convention, unlike
skill_tags (which explicitly requires "lowercase, hyphenated"). Gemini
filled that gap inconsistently across calls. This script cleans data
already generated; the prompt itself should also be fixed going forward
(see the one-line addition noted at the bottom of this file) so future
runs don't reproduce the same inconsistency.

USAGE
  python normalize_degree_field.py final_dataset_v4.csv
  Overwrites in place by default — pass --output to write elsewhere.
"""
import argparse
import re

import pandas as pd


def normalize_degree_field(raw) -> str:
    if pd.isna(raw) or not str(raw).strip():
        return "unknown"
    s = str(raw).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)       # spaces/underscores -> hyphens
    s = re.sub(r"[^a-z0-9\-]", "", s)   # drop stray punctuation
    s = re.sub(r"-+", "-", s).strip("-")  # collapse repeats, trim edges
    return s or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("--output", default=None, help="defaults to overwriting input_csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if "degree_field" not in df.columns:
        raise SystemExit(f"No 'degree_field' column found in {args.input_csv}")

    before = df["degree_field"].astype(str).copy()
    df["degree_field"] = df["degree_field"].apply(normalize_degree_field)

    changed = (before.str.lower().str.strip() != df["degree_field"]).sum()
    print(f"Normalized {changed}/{len(df)} rows.")
    print("\nBefore -> after examples (first 10 changed rows):")
    diff_mask = before.str.lower().str.strip() != df["degree_field"]
    for b, a in list(zip(before[diff_mask], df["degree_field"][diff_mask]))[:10]:
        print(f"  {b!r:35s} -> {a!r}")

    print(f"\n{df['degree_field'].nunique()} unique degree_field values after normalization "
          f"(compare against nunique() on the ORIGINAL column to see how much fragmentation this fixed).")

    out_path = args.output or args.input_csv
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()