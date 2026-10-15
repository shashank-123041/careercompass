"""
find_thin_qa_rows.py — pulls every 'qa' row that lacks ANY recognized
QA-tool identity tag, for manual inspection.

WHY: diagnose_confusion_pair.py's output showed the qa->data-engineer
confusion isn't general vocabulary overlap (only 2/15 top tags shared
between the classes) — it's a specific SUBGROUP of qa rows that carry
no testing-tool identity at all (no selenium/testng/jenkins/java/etc.)
and instead lean on generic tags like 'database', which is dominant in
data-engineer (85% of that class). This script surfaces every row
matching that pattern across the WHOLE dataset (not just the ones that
happened to get misclassified in one particular 3-fold split), so you
can look at them yourself and decide what's actually going on — this
script does not guess at that; only inspection of the real source rows
(original resume text, if Tier 1, or the generation prompt, if
synthetic) can determine whether a given row is:
  (a) an extraction failure (the person likely used testing tools but
      they weren't captured from their resume text),
  (b) a mislabeling (this profile isn't really a QA/test-automation
      role at all), or
  (c) a genuinely distinct, legitimate QA sub-population (e.g. manual
      testers, QA leads, or ERP/business-systems testers who don't
      personally write automation code) that the current single 'qa'
      label doesn't distinguish from automation-focused QA — the same
      kind of situation MERGED_LABEL_MAP already handles for
      ml-cluster/mobile.

This script only SURFACES the rows — it makes no changes to the
dataset and recommends none. Deciding (a)/(b)/(c) requires a human
who can see the real underlying source data, which this environment
does not have access to.

USAGE
  python -m scripts.experiments.find_thin_qa_rows
  python -m scripts.experiments.find_thin_qa_rows --role qa --identity-tags selenium,testng,jenkins,java,cypress,playwright,postman,jmeter,manual-testing,test-automation,qtp,appium
"""
import argparse
import ast

import pandas as pd

from careercompass.ml.common import DATA_CSV, load_and_prepare, TARGET

DEFAULT_IDENTITY_TAGS = {
    "selenium", "testng", "jenkins", "java", "cypress", "playwright",
    "postman", "jmeter", "manual-testing", "test-automation", "qtp",
    "appium", "junit", "cucumber", "soapui", "loadrunner",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="qa")
    ap.add_argument("--identity-tags", default=None,
                     help="comma-separated list; defaults to a broad QA-tool identity set")
    args = ap.parse_args()

    identity_tags = (
        set(t.strip() for t in args.identity_tags.split(","))
        if args.identity_tags else DEFAULT_IDENTITY_TAGS
    )

    print(f"Loading {DATA_CSV}...")
    df = load_and_prepare(DATA_CSV)
    df["skill_tags_parsed"] = df["skill_tags"].apply(ast.literal_eval)

    role_rows = df[df[TARGET] == args.role].copy()
    print(f"\n{len(role_rows)} total '{args.role}' rows in the dataset.")
    print(f"Identity tag set being checked against: {sorted(identity_tags)}\n")

    def has_identity_tag(tags):
        return bool(set(tags) & identity_tags)

    role_rows["has_identity_tag"] = role_rows["skill_tags_parsed"].apply(has_identity_tag)
    thin = role_rows[~role_rows["has_identity_tag"]]
    solid = role_rows[role_rows["has_identity_tag"]]

    print(f"{len(thin)}/{len(role_rows)} rows ({len(thin)/len(role_rows):.1%}) carry NONE of the "
          f"identity tags above \u2014 these are the rows most likely contributing to the "
          f"qa\u2192data-engineer confusion.\n")

    print("=" * 70)
    print(f"THIN ROWS (no QA identity tag) \u2014 review these yourself:")
    print("=" * 70)
    display_cols = [c for c in ["source", "original_category", "original_text_preview"] if c in thin.columns]
    for idx, row in thin.iterrows():
        preview = ""
        for c in display_cols:
            val = row.get(c, "")
            if pd.notna(val) and str(val).strip():
                preview += f" [{c}={str(val)[:80]}]"
        print(f"  row {idx}: {row['skill_tags_parsed']}{preview}")

    print("\n" + "=" * 70)
    print("FOR COMPARISON \u2014 a few 'solid' rows (DO carry an identity tag):")
    print("=" * 70)
    for idx, row in solid.head(5).iterrows():
        print(f"  row {idx}: {row['skill_tags_parsed']}")

    if "source" in thin.columns:
        print("\n" + "=" * 70)
        print("BREAKDOWN OF THIN ROWS BY SOURCE (real vs which synthetic batch):")
        print("=" * 70)
        print(thin["source"].value_counts().to_string())
        print("\nIf thin rows cluster heavily in ONE source/batch, that batch's extraction")
        print("or generation process is the specific thing worth fixing \u2014 not the qa")
        print("class in general.")

    print("\n" + "=" * 70)
    print("WHAT TO DO WITH THIS LIST")
    print("=" * 70)
    print("For each thin row, decide:")
    print("  (a) Extraction failure (Tier 1 real resume) \u2192 re-check the original resume")
    print("      text; the person may genuinely have QA tool experience that wasn't captured.")
    print("  (b) Mislabeled \u2192 this profile may not belong in 'qa' at all; consider")
    print("      recategorizing or removing it.")
    print("  (c) Legitimate distinct QA sub-population (manual/ERP/business-systems")
    print("      testers) \u2192 consider whether 'qa' should be split the same way")
    print("      'mobile' and 'ml-cluster' already are for the model, with its own")
    print("      distinguishing tags (e.g. 'manual-testing', 'uat', 'test-case-design')")
    print("      added during a FUTURE, targeted augmentation pass \u2014 not more generic")
    print("      qa rows like the ones already tried.")


if __name__ == "__main__":
    main()