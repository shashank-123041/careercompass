"""
generate_shadow_data_from_holdout.py — generates real shadow-mode
comparison data without needing live users, by running CATBOOST and SVM
over REAL, HELD-OUT rows from candidate_job_role_dataset.csv that were
never incorporated into final_dataset_v7.csv (training coverage is
checked explicitly below, not assumed from pilot_extract_skills.py's
original 2-per-category pilot sample size, since a larger batch was run
afterward per that script's own docstring).

WHY THIS IS A MEANINGFUL SUBSTITUTE FOR LIVE TRAFFIC, NOT A WORKAROUND:
these are REAL, human-written skill/qualification profiles the project
already has on disk — not synthetic tag lists, not another call to
diagnose_shadow_logging.py's single test vector. Testing on genuinely
held-out real rows is a legitimate proxy for "new users" when none are
available, closer in kind to the offline validation's own tier2_synthetic
rows than to a fabricated test case.

SCOPE, DELIBERATELY NARROW: only candidate_job_role_dataset.csv (tier2) is
used here. Its `skills` column is already structured text, so building
ml_features needs only the normalization logic ALREADY WRITTEN in
pilot_extract_skills.py (normalize_tier2_skills, normalize_degree_level,
TIER2_MAP) — reused directly, not reimplemented, and requires NO
extraction model call at all (no Gemini, no auto_extract.py), so this
costs nothing and runs in seconds regardless of batch size.
UpdatedResumeDataSet.csv (tier1, raw resume text) is NOT included here —
extracting ml_features from it would need auto_extract.py's fallback
extractor, whose actual implementation wasn't available to verify when
this script was written. Extend this script to cover tier1 once that
file's real interface is confirmed.

DEDUPLICATION: a tier2_synthetic row's original_text_preview in
final_dataset_v7.csv was set to the row's own raw `skills` string (see
pilot_extract_skills.py's normalize path) — so an EXACT STRING MATCH
against that column is a reliable way to detect "already used in
training," not a guess.

TAGGING: logged rows carry context_snapshot.holdout_batch=True (a
DIFFERENT tag from diagnose_shadow_logging.py's diagnostic=True) so
shadow_log_analysis.py's existing diagnostic-row filter does NOT exclude
these — they ARE meant to be treated as real comparison signal. The tag
exists purely so you can filter them out later if/when genuine organic
traffic arrives and you want to look at that subset alone.

USAGE
  python -m scripts.shadow.generate_shadow_data_from_holdout
  python -m scripts.shadow.generate_shadow_data_from_holdout --limit 50
"""
import argparse
import json

import pandas as pd

from careercompass.db import init_db, get_conn
from careercompass.config import Config
from careercompass.db import interaction_operations as interaction_ops
from careercompass.ml.common import DATA_CSV, load_and_prepare
from scripts.data_building.pilot_extract_skills import TIER2_MAP, normalize_tier2_skills, normalize_degree_level

import careercompass.ml.catboost_inference as role_clf_catboost  # needs requirements-ml.txt (catboost)
from careercompass.ml import svm_inference

CANDIDATE_CSV = str(Config.TIER2_RAW_CSV)


def find_already_used_skills_strings() -> set:
    """Returns the set of raw `skills` strings already present in
    final_dataset_v7.csv's original_text_preview column — these are the
    tier2 rows already consumed during training, to be excluded."""
    df = load_and_prepare(DATA_CSV)
    used = set(df["original_text_preview"].dropna().astype(str))
    return used


def load_holdout_rows(limit: int = None) -> pd.DataFrame:
    df2 = pd.read_csv(CANDIDATE_CSV)
    df2 = df2[df2["job_role"].isin(TIER2_MAP.keys())].copy()
    df2["experience_level"] = df2["experience_level"].str.rstrip(",").str.strip()
    df2["role_slug"] = df2["job_role"].map(TIER2_MAP)

    already_used = find_already_used_skills_strings()
    holdout = df2[~df2["skills"].isin(already_used)].copy()

    print(f"{len(df2)} rows match the taxonomy, {len(df2) - len(holdout)} already "
          f"used in training (exact skills-string match), {len(holdout)} genuinely held out.")

    if limit:
        holdout = holdout.sample(n=min(limit, len(holdout)), random_state=42)
    return holdout


def row_to_ml_features(row) -> dict:
    return {
        "skill_tags": normalize_tier2_skills(row["skills"]),
        "degree_level": normalize_degree_level(row["qualification"]),
        "degree_field": row["qualification"],
        "years_experience": row["experience_level"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="cap the number of held-out rows processed (default: all)")
    args = ap.parse_args()

    init_db()
    holdout = load_holdout_rows(args.limit)

    if holdout.empty:
        print("No held-out rows found — either everything in this dataset was already "
              "used in training, or no rows matched the role taxonomy. Nothing to do.")
        return

    print(f"\nProcessing {len(holdout)} held-out rows through both models...\n")

    n_logged, n_both_available, n_agree = 0, 0, 0
    for i, (_, row) in enumerate(holdout.iterrows()):
        ml_features = row_to_ml_features(row)
        if not ml_features["skill_tags"]:
            continue

        cb_result = role_clf_catboost.predict_role_for_user(ml_features)
        cb_rankings = role_clf_catboost.predict_role_rankings(ml_features)
        svm_result = svm_inference.predict_svm(ml_features)

        fake_user_id = f"holdout_batch_{row['candidate_id']}"

        try:
            svm_rankings = svm_inference.predict_svm_rankings(ml_features)

            def top_n(rankings, n=3):
                return [r["role_slug"] for r in (rankings or [])[:n]]

            cb_top3 = top_n(cb_rankings)
            svm_top3 = top_n(svm_rankings)

            interaction_ops.log_event(
                user_id=fake_user_id, item_type="role_classifier_shadow", item_id="svm_v1",
                event_type="shadow_prediction", source="svm_shadow_v1",
                context_snapshot={
                    "catboost": {
                        "available": cb_result.get("available"),
                        "predicted_role": cb_result.get("predicted_role"),
                        "resolved_role": cb_result.get("resolved_role"),
                        "confidence": cb_result.get("confidence"),
                        "is_confident": cb_result.get("is_confident"),
                    },
                    "svm": {
                        "available": svm_result.get("available"),
                        "predicted_role": svm_result.get("predicted_role"),
                        "resolved_role": svm_result.get("resolved_role"),
                        "confidence": svm_result.get("confidence"),
                        "is_confident": svm_result.get("is_confident"),
                    },
                    "agree": (
                        cb_result.get("resolved_role") == svm_result.get("resolved_role")
                        if cb_result.get("available") and svm_result.get("available")
                        else None
                    ),
                    "top3": {
                        "catboost_top3": cb_top3, "svm_top3": svm_top3,
                        "top3_overlap_count": len(set(cb_top3) & set(svm_top3)),
                        "catboost_top1_in_svm_top3": cb_result.get("resolved_role") in svm_top3,
                        "svm_top1_in_catboost_top3": svm_result.get("resolved_role") in cb_top3,
                    },
                    "holdout_batch": True,
                    "true_role_slug": row["role_slug"],  # kept for your own reference —
                                                           # NOT used by shadow_log_analysis.py,
                                                           # but lets you check which model
                                                           # was actually RIGHT, not just which
                                                           # two agreed
                },
            )
            n_logged += 1
            if cb_result.get("available") and svm_result.get("available"):
                n_both_available += 1
                if cb_result.get("resolved_role") == svm_result.get("resolved_role"):
                    n_agree += 1

        except Exception as e:
            print(f"  row {i}: failed to log — {e}")

        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(holdout)} processed")

    print(f"\nLogged {n_logged} shadow rows from held-out real data.")
    if n_both_available:
        print(f"Top-1 agreement on this batch: {n_agree}/{n_both_available} ({n_agree/n_both_available:.1%})")
    print("\nRun shadow_log_analysis.py now — these rows are tagged holdout_batch=True")
    print("(not diagnostic=True), so they WILL be included in its analysis.")


if __name__ == "__main__":
    main()