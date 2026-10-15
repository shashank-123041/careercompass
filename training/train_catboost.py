"""
train_role_classifier.py — CatBoost role/roadmap-selection model (Phase 2).
CANONICAL VERSION — supersedes all earlier _v2/_v3 variants from this session.

DESIGN DECISIONS (see conversation history for full reasoning)
  - Two-tier feature representation: skill_tags -> CatBoost TEXT feature;
    degree_level/degree_field -> CATEGORICAL; years_experience -> NUMERIC
    (missing -> NaN, CatBoost handles natively).
  - "source" column is explicitly EXCLUDED as a feature — including it would
    let the model shortcut on "which dataset did this come from" instead of
    learning skill->role patterns, a shortcut that doesn't exist on a real
    student's resume.
  - class weighting: SQRT-BALANCED (see compute_sqrt_balanced_weights()),
    not auto_class_weights='Balanced'. Direct comparison across 4 schemes
    (none / balanced / sqrt_balanced / capped_balanced — see
    tune_class_weights.py) showed full 'Balanced' weighting gave backend
    (651 rows, one of the largest classes) a recall of only 0.61 despite
    0.89 precision — the model rarely GUESSED backend even though it was
    usually right when it did, consistent with inverse-frequency
    balancing giving huge classes disproportionately low per-example
    loss weight. sqrt-balanced (a softer compromise) improved macro F1
    (0.686->0.738), backend recall, AND the small classes' average F1
    simultaneously on identical data/folds — not a trade-off, a
    straightforward win. Class sizes range from 16 (data-analyst) to 660
    (data-engineer), a ~41x ratio, so SOME reweighting is still needed —
    just not the full inverse-frequency version.
  - Evaluation: Stratified K-Fold CV (k=3), not a single split — several
    classes are still small enough (16-50 rows) that a single split could
    leave very few test examples. NOTE: a repeated-CV stability check
    (see check_class_stability.py) found several small classes' F1 has a
    natural std of 0.10+ across different fold splits ALONE — treat any
    single 3-fold run's per-class numbers for classes under ~50 rows as a
    rough estimate, not a precise one, and prefer repeated CV before
    reacting to an apparent regression in one of them.
  - PER-ROLE CONFIDENCE THRESHOLDS (new in this version): the 9 Tier 3
    classes (TIER3_ROLES below) are 100% Claude-generated, web-search-
    grounded synthetic data with ZERO real resume grounding — a disclosed,
    deliberate limitation, not an oversight. They get a stricter confidence
    bar (TIER3_CONFIDENCE_THRESHOLD) before being trusted over the Phase-A
    heuristic. This is "coverage with calibrated humility" per the approved
    plan, not full exclusion (Option B) and not unqualified trust (naive
    Option A). Retrain these specific classes from real logged interactions
    once usage data accumulates — the interaction-logging architecture
    already designed for this exists specifically to close this gap later.
  - MERGED_LABEL_MAP (new in this version): "ml-cluster" and "mobile" are
    NOT real Config.ROLE_SLUGS values — they're data-scarcity merges
    (ai-engineer/ai-data-scientist/machine-learning -> ml-cluster;
    android/ios -> mobile) made necessary by what the source datasets could
    support. The serving endpoint MUST disambiguate these before showing a
    result to a user or routing to a roadmap page — there is no roadmap.sh
    page for "ml-cluster". Included here as documented metadata so the
    disambiguation requirement travels with the model, not just the chat log.
  - KNOWN GAP, NOT YET ADDRESSED: Config.ROLE_SLUGS also contains
    "devops-engineer" and "server-side-game-developer" — neither has any
    training data (missed during the original Tier 3 exclusion list, only
    caught while reviewing the full role list for this version). They look
    like near-duplicates of "devops" and "game-developer" respectively.
    Until a decision is made, they are UNROUTABLE by this model — the
    serving endpoint should send them straight to the heuristic, or map
    them to their apparent near-duplicate at the application layer.

USAGE
  python -m training.train_catboost
  Reads Config.FINAL_DATASET_CSV (data/final/final_dataset_v7.csv), prints
  CV results, trains + saves models/role_classifier.cbm + role_classifier_meta.json.
  (CatBoost is retained for rollback and for the architecture comparison;
  the production classifier is the SVM — see training/train_svm.py.)
"""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from careercompass.config import Config
from careercompass.ml.common import (
    DATA_CSV, CONFIDENCE_THRESHOLD, TIER3_ROLES, TIER3_CONFIDENCE_THRESHOLD,
    UNROUTABLE_ROLE_SLUGS_NO_DATA, MERGED_LABEL_MAP,
    TEXT_FEATURE, CAT_FEATURES, NUM_FEATURES, TARGET,
    load_and_prepare, compute_sqrt_balanced_weights, threshold_for_role,
)

MODEL_OUT = str(Config.ROLE_CLASSIFIER_MODEL_PATH)
META_OUT = str(Config.ROLE_CLASSIFIER_META_PATH)

N_FOLDS = 3
# Shared constants and helpers (thresholds, Tier-3 roles, merged-label map, feature
# names, load_and_prepare, compute_sqrt_balanced_weights, threshold_for_role)
# live in careercompass/ml/common.py so the serving layer doesn't import this script.


def build_pool(df: pd.DataFrame) -> Pool:
    feature_cols = [TEXT_FEATURE] + CAT_FEATURES + NUM_FEATURES
    return Pool(
        data=df[feature_cols], label=df[TARGET],
        text_features=[TEXT_FEATURE], cat_features=CAT_FEATURES,
    )


def make_model(class_weights: dict = None) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=150, depth=4, learning_rate=0.08,
        loss_function="MultiClass", class_weights=class_weights,
        verbose=False, random_seed=42,
    )


def run_cross_validation(df: pd.DataFrame):
    print(f"Running {N_FOLDS}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds = pd.Series(index=df.index, dtype=object)
    fold_f1s, fold_accs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[TARGET])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        weights = compute_sqrt_balanced_weights(train_df)  # per-fold, from train_df only — no val leakage
        model = make_model(class_weights=weights)
        model.fit(build_pool(train_df))
        preds = model.predict(build_pool(val_df)).flatten()
        oof_preds.iloc[val_idx] = preds
        acc = accuracy_score(val_df[TARGET], preds)
        f1 = f1_score(val_df[TARGET], preds, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold+1}: accuracy={acc:.3f}  macro_f1={f1:.3f}  (val size={len(val_df)})")

    print(f"\nCV mean accuracy: {np.mean(fold_accs):.3f} (+/- {np.std(fold_accs):.3f})")
    print(f"CV mean macro F1: {np.mean(fold_f1s):.3f} (+/- {np.std(fold_f1s):.3f})")
    print("\nOut-of-fold per-class report:")
    print(classification_report(df[TARGET], oof_preds, zero_division=0))
    return oof_preds


def train_final_model(df: pd.DataFrame) -> CatBoostClassifier:
    print("\nTraining FINAL model on 100% of data...")
    weights = compute_sqrt_balanced_weights(df)
    model = make_model(class_weights=weights)
    model.fit(build_pool(df))
    return model


def print_confusion_diagnostics(y_true, y_pred, focus_roles: list = None,
                                 output_csv: str = None):
    """
    Prints the full labeled confusion matrix and, when focus_roles is given,
    a per-role "what is it actually being predicted as instead" breakdown —
    the thing precision/recall alone can't tell you. A role with low recall
    could be losing predictions to ONE dominant confusable class or being
    spread thin across many; those need different fixes (targeted data for
    the specific confusable pair vs. broader augmentation), so seeing WHICH
    it is matters, not just that recall is low.

    Also writes the full matrix to output_csv (if given) since a 22x22
    matrix is unwieldy to read in a terminal — easier to open in a
    spreadsheet and scan for the largest off-diagonal values yourself.
    """
    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    if output_csv:
        cm_df.to_csv(output_csv)
        print(f"\nFull confusion matrix ({len(classes)}x{len(classes)}) written to {output_csv}")

    if focus_roles:
        print("\n--- Focused confusion breakdown ---")
        for role in focus_roles:
            if role not in cm_df.index:
                print(f"\n{role}: not present in this fold's labels — skipping")
                continue
            row = cm_df.loc[role].drop(role, errors="ignore")
            total = cm_df.loc[role].sum()
            correct = cm_df.loc[role, role] if role in cm_df.columns else 0
            top_confusions = row[row > 0].sort_values(ascending=False).head(5)

            print(f"\n{role}  (true count={total}, correctly predicted={correct}, "
                  f"recall={correct/total:.2f})" if total else f"\n{role}: no true instances")
            if top_confusions.empty:
                print("  No misclassifications — every instance predicted correctly.")
            else:
                print("  Most often misclassified AS:")
                for pred_role, count in top_confusions.items():
                    print(f"    -> {pred_role:25s} {count:4d}  ({count/total:.1%} of all '{role}' instances)")


def main():
    from careercompass.ml.catboost_inference import predict_with_confidence  # smoke tests only
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare(DATA_CSV)
    print(f"Loaded {len(df)} rows, {df[TARGET].nunique()} classes")
    print(df[TARGET].value_counts().sort_values(), "\n")

    oof_preds = run_cross_validation(df)

    FOCUS_ROLES = ["full-stack", "mobile", "backend", "data-analyst", "qa", "ml-cluster"]
    print_confusion_diagnostics(
        df[TARGET], oof_preds,
        focus_roles=FOCUS_ROLES,
        output_csv=str(Config.RESULTS_DIR / "confusion_matrix_full.csv"),
    )

    final_model = train_final_model(df)
    final_model.save_model(MODEL_OUT)

    meta = {
        "classes": list(final_model.classes_),
        "confidence_threshold_default": CONFIDENCE_THRESHOLD,
        "confidence_threshold_tier3": TIER3_CONFIDENCE_THRESHOLD,
        "tier3_roles": sorted(TIER3_ROLES),
        "merged_label_map": MERGED_LABEL_MAP,
        "unroutable_role_slugs_no_data": UNROUTABLE_ROLE_SLUGS_NO_DATA,
        "text_feature": TEXT_FEATURE,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "n_training_rows": len(df),
        "class_weighting_scheme": "sqrt_balanced",  # see compute_sqrt_balanced_weights() —
                                                      # replaced auto_class_weights='Balanced'
                                                      # after tune_class_weights.py showed it
                                                      # beats 'Balanced' on macro F1, backend
                                                      # recall, AND small-class F1 simultaneously
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model to {MODEL_OUT}, metadata to {META_OUT}")

    print("\n--- Smoke test: default-threshold role ---")
    print(predict_with_confidence(final_model, ["python","django","react","postgresql"],
                                   "bachelors","computer-science",3))

    print("\n--- Smoke test: Tier 3 role (ux-design) ---")
    print(predict_with_confidence(final_model, ["figma","wireframing","user-research","usability-testing"],
                                   "bachelors","design",3))

    print("\n--- Smoke test: merged-label role (mobile) ---")
    print(predict_with_confidence(final_model, ["swift","ios-development","xcode"],
                                   "bachelors","computer-science",3))


if __name__ == "__main__":
    main()