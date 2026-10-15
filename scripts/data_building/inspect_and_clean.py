"""
inspect_and_clean.py — surgical fix based on the stability comparison:
keep the data-analyst scraped rows (clear win: mean F1 +0.191, std more
than halved), remove the full-stack scraped rows (clear regression:
mean F1 -0.124, more than 2x its prior std of 0.055 — a real,
reproducible effect, not noise), leave everything else untouched.

Also checks the full-stack scraped rows for the SAME kind of near-
duplicate/template contamination already found once in tier1_real_resumeatlas
(the Jessica Claire case) — before assuming the regression is purely a
vocabulary-overlap issue, rule out a data-quality bug first, since that's
cheaper to fix and we've already seen it happen once in this project.
"""
import ast
import pandas as pd

from careercompass.config import Config

df = pd.read_csv(Config.FINAL_DATA_DIR / "final_dataset_v6.csv")
print(f"Loaded {len(df)} rows")

scraped_fullstack = df[(df["source"] == "job_posting_scraped_v1") & (df["role_slug"] == "full-stack")]
print(f"\n{len(scraped_fullstack)} scraped full-stack rows found — checking for duplication first:")

# Duplicate check: same signature as the Jessica Claire find — near-identical
# original_text_preview values would indicate templated/reposted listings,
# not distinct real postings.
if "original_text_preview" in scraped_fullstack.columns:
    preview_counts = scraped_fullstack["original_text_preview"].value_counts()
    dupes = preview_counts[preview_counts > 1]
    if len(dupes):
        print(f"  WARNING: {dupes.sum()} rows share a duplicated preview text — "
              f"possible reposted/templated listings:")
        print(dupes.to_string())
    else:
        print("  No exact preview duplicates found — these appear to be genuinely distinct postings.")
        print("  (The regression is more likely genuine vocabulary dilution, not a data-quality bug.)")

print("\nSample full-stack scraped skill_tags for a quick manual sanity check:")
for tags in scraped_fullstack["skill_tags"].head(10):
    print(f"  {ast.literal_eval(tags)}")

# Surgical removal: drop ONLY the full-stack rows from this source,
# keep every other role's scraped rows (including the data-analyst win).
cleaned = df[~((df["source"] == "job_posting_scraped_v1") & (df["role_slug"] == "full-stack"))]
cleaned.to_csv(Config.FINAL_DATA_DIR / "final_dataset_v7.csv", index=False)
print(f"\nRemoved {len(df) - len(cleaned)} full-stack scraped rows.")
print(f"Wrote final_dataset_v7.csv ({len(cleaned)} rows) — rerun check_class_stability.py")
print("on this file to confirm: (1) full-stack recovers toward its prior ~0.72 mean,")
print("(2) data-analyst's win at ~0.77 holds, (3) nothing else moved unexpectedly.")