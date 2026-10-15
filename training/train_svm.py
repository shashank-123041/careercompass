"""
train_svm_classifier.py — CANONICAL SVM training script, supersedes
CatBoost as the production role classifier per the validation chain in
compare_model_architectures.py -> svm_hyperparameter_search.py ->
per_class_comparison_svm_vs_catboost.py -> per_class_calibration_check.py
-> calibration_method_comparison.py -> confusion_pattern_parity_check.py
-> svm_diagnose_confusion_pair.py -> tfidf_tokenization_check.py ->
atomic_vocab_size_check.py.

FINAL LOCKED CONFIG (see atomic_vocab_size_check.py's docstring for the
tokenizer decision specifically):
  - TfidfVectorizer(max_features=800), DEFAULT tokenizer — deliberately
    NOT switched to atomic/whole-tag tokenization. That was tested
    directly: it fixed one specific false-positive confusion
    (product-manager/engineering-manager, caused by tag fragmentation
    creating spurious token overlap) but regressed aggregate macro F1
    by 0.0147 under 5x3 repeated CV, confirmed NOT a vocabulary-size
    artifate (identical results at max_features=1200/2000/uncapped —
    true atomic vocabulary is ~1040 tags, already fully available at
    1200). The word-fragmentation's cross-tag feature-sharing is net
    beneficial, especially for this dataset's thinnest classes (as few
    as 32-51 rows) — DO NOT change this without re-running
    atomic_vocab_size_check.py against the current dataset.
  - LinearSVC(C=1.0) — tuned via svm_hyperparameter_search.py's 45-config
    grid (C x max_features x weight_scheme), same "don't over-tune past
    diminishing returns" discipline CatBoost's own tuning pass
    established.
  - sqrt-balanced sample_weight (compute_sqrt_balanced_weights, reused
    directly from train_role_classifier.py) — NOT class_weight="balanced",
    same reasoning as CatBoost's own choice: the grid confirmed
    sample_weight_only performs at least as well as class_weight
    variants for this dataset.
  - CalibratedClassifierCV(method="isotonic", cv=3) — beat sigmoid
    calibration cleanly (ECE 0.0675 vs 0.1265) with an IDENTICAL 22/22
    per-role threshold pass rate, confirmed via
    calibration_method_comparison.py. Every role's confidence threshold
    (CONFIDENCE_THRESHOLD=0.35 default, TIER3_CONFIDENCE_THRESHOLD=0.55)
    is empirically honored — see per_class_calibration_check.py's and
    calibration_method_comparison.py's output for the underlying evidence.

VALIDATION SUMMARY (full detail lives in each named script above, not
reproduced here in full):
  - Per-class F1: beat CatBoost on all 22 classes, all 9 Tier-3 classes
    included, with the largest gains concentrated exactly on CatBoost's
    weakest classes (devsecops +0.325, network-engineer +0.319).
  - Known confusion pair qa->data-engineer: 19.2% -> 5.3%.
  - No unresolved NEW confusion patterns — the one candidate
    (product-manager/engineering-manager) was investigated via
    svm_diagnose_confusion_pair.py and traced to the tokenizer behavior
    documented above, not a data-quality gap.

WHAT THIS SCRIPT DOES DIFFERENTLY FROM THE VALIDATION SCRIPTS ABOVE:
those all used 5x3 repeated CV (15 fold-fits) for statistically sound
comparisons — appropriate for a one-time architecture decision, but too
slow for routine retraining. This script runs a SINGLE 3-fold CV as a
quick sanity check (same convention as train_role_classifier.py's own
run_cross_validation) before the final fit — a smoke test that nothing
broke, not a re-validation of the whole decision chain. If you retrain
on meaningfully different data (a large new augmentation batch, a
schema change), re-run the FULL validation chain above, not just this
script's quick check.

OUTPUT: bundles the fitted TfidfVectorizer, OneHotEncoder, SimpleImputer,
and CalibratedClassifierCV(LinearSVC) into ONE joblib artifact — see
module docstring in role_classifier_inference.py (Phase 4's next
deliverable) for why bundling matters: a mismatched vectorizer+model
pair fails silently, not loudly, so there must be no way to load one
without the other.

USAGE
  python -m training.train_svm
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from careercompass.config import Config
from careercompass.ml.common import (
    DATA_CSV, load_and_prepare, compute_sqrt_balanced_weights,
    TEXT_FEATURE, CAT_FEATURES, NUM_FEATURES, TARGET,
    TIER3_ROLES, TIER3_CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD,
    MERGED_LABEL_MAP, UNROUTABLE_ROLE_SLUGS_NO_DATA, threshold_for_role,
)

# ── Locked final config — see module docstring for the evidence behind
#    every one of these values ─────────────────────────────────────────────
SVM_C = 1.0
SVM_MAX_FEATURES = 800
SVM_CALIBRATION_METHOD = "isotonic"
SVM_CALIBRATION_CV = 3

N_FOLDS = 3           # quick sanity check only — see module docstring
SEED_BASE = 42

MODEL_OUT = str(Config.SVM_MODEL_PATH)
META_OUT = str(Config.SVM_META_PATH)


def build_features(train_df: pd.DataFrame, val_df: pd.DataFrame = None):
    """
    Fits all three preprocessors on train_df ONLY, transforms train_df
    (and val_df, if given). Returns (X_train, X_val_or_None, artifacts)
    where artifacts is the dict that gets bundled into MODEL_OUT — this
    EXACT function (or an identical port of it) must be reused at
    inference time; a hand-rewritten equivalent is exactly how a
    train/serve skew bug gets introduced silently.
    """
    tfidf = TfidfVectorizer(max_features=SVM_MAX_FEATURES)  # default tokenizer — see module docstring
    X_text_train = tfidf.fit_transform(train_df[TEXT_FEATURE])

    ohe = OneHotEncoder(handle_unknown="ignore")
    X_cat_train = ohe.fit_transform(train_df[CAT_FEATURES])

    imputer = SimpleImputer(strategy="median")
    X_num_train = imputer.fit_transform(train_df[NUM_FEATURES])

    X_train = hstack([X_text_train, X_cat_train, csr_matrix(X_num_train)]).tocsr()

    artifacts = {"tfidf": tfidf, "ohe": ohe, "imputer": imputer}

    if val_df is None:
        return X_train, None, artifacts

    X_text_val = tfidf.transform(val_df[TEXT_FEATURE])
    X_cat_val = ohe.transform(val_df[CAT_FEATURES])
    X_num_val = imputer.transform(val_df[NUM_FEATURES])
    X_val = hstack([X_text_val, X_cat_val, csr_matrix(X_num_val)]).tocsr()

    return X_train, X_val, artifacts


def make_svm() -> CalibratedClassifierCV:
    base = LinearSVC(C=SVM_C, random_state=SEED_BASE, max_iter=5000)
    return CalibratedClassifierCV(base, method=SVM_CALIBRATION_METHOD, cv=SVM_CALIBRATION_CV)


def run_sanity_check_cv(df: pd.DataFrame, classes: list, class_to_idx: dict):
    """Quick single 3-fold check — NOT a re-validation of the architecture
    decision (see module docstring). Just confirms nothing broke before
    committing to the final full-data fit."""
    print(f"Running {N_FOLDS}-fold sanity-check CV (NOT the full validation "
          f"chain — see module docstring)...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_BASE)
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    oof_preds = pd.Series(index=df.index, dtype=object)
    fold_f1s, fold_accs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[TARGET])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        X_train, X_val, _ = build_features(train_df, val_df)
        y_train = train_df[TARGET].map(class_to_idx).values

        weights = compute_sqrt_balanced_weights(train_df)
        sample_weight = train_df[TARGET].map(weights).values

        model = make_svm()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        pred_idx = model.predict(X_val)
        preds = [idx_to_class[p] for p in pred_idx]
        oof_preds.iloc[val_idx] = preds

        acc = accuracy_score(val_df[TARGET], preds)
        f1 = f1_score(val_df[TARGET], preds, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold+1}: accuracy={acc:.3f}  macro_f1={f1:.3f}  (val size={len(val_df)})")

    print(f"\nSanity-check mean accuracy: {np.mean(fold_accs):.3f} (+/- {np.std(fold_accs):.3f})")
    print(f"Sanity-check mean macro F1: {np.mean(fold_f1s):.3f} (+/- {np.std(fold_f1s):.3f})")
    print("(Compare against svm_hyperparameter_search.py's 0.917 5x3-repeated-CV figure —")
    print("this single-3-fold number will naturally be noisier; a LARGE deviation from 0.917")
    print("warrants investigation before trusting the final fit below, per")
    print("check_class_stability.py's own finding that single-split numbers for this")
    print("dataset's smaller classes can swing meaningfully from fold luck alone.)")
    print("\nOut-of-fold classification report:")
    print(classification_report(df[TARGET], oof_preds, zero_division=0))


def train_final_model(df: pd.DataFrame, class_to_idx: dict) -> tuple:
    print("\nTraining FINAL model on 100% of data...")
    X_train, _, artifacts = build_features(df, val_df=None)
    y_train = df[TARGET].map(class_to_idx).values

    weights = compute_sqrt_balanced_weights(df)
    sample_weight = df[TARGET].map(weights).values

    model = make_svm()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, artifacts


def main():
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare(DATA_CSV)
    classes = sorted(df[TARGET].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Loaded {len(df)} rows, {df[TARGET].nunique()} classes")
    print(df[TARGET].value_counts().sort_values(), "\n")

    run_sanity_check_cv(df, classes, class_to_idx)

    model, artifacts = train_final_model(df, class_to_idx)

    # ── Bundle EVERYTHING needed for inference into one artifact ────────────
    # No separate files for tfidf/ohe/imputer/model — a mismatched set is a
    # silent-failure risk this bundling eliminates by construction.
    bundle = {
        "model": model,
        "tfidf": artifacts["tfidf"],
        "ohe": artifacts["ohe"],
        "imputer": artifacts["imputer"],
        "classes": [str(c) for c in classes],  # int-encoded label -> role string
        "text_feature": TEXT_FEATURE,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
    }
    joblib.dump(bundle, MODEL_OUT)
    print(f"\nSaved bundled artifact to {MODEL_OUT}")

    meta = {
        "model_type": "linear_svm",  # NEW field — role_classifier_inference.py
                                       # branches on this; absence/= "catboost"
                                       # means the old CatBoost loading path
        "classes": [str(c) for c in classes],
        "confidence_threshold_default": CONFIDENCE_THRESHOLD,
        "confidence_threshold_tier3": TIER3_CONFIDENCE_THRESHOLD,
        "tier3_roles": sorted(TIER3_ROLES),
        "merged_label_map": MERGED_LABEL_MAP,
        "unroutable_role_slugs_no_data": UNROUTABLE_ROLE_SLUGS_NO_DATA,
        "text_feature": TEXT_FEATURE,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "n_training_rows": len(df),
        "class_weighting_scheme": "sqrt_balanced",
        "svm_config": {
            "C": SVM_C, "max_features": SVM_MAX_FEATURES,
            "tokenizer": "default",  # deliberately NOT "atomic" — see module docstring
            "calibration_method": SVM_CALIBRATION_METHOD,
            "calibration_cv": SVM_CALIBRATION_CV,
        },
        "validated_via": [
            "compare_model_architectures.py", "svm_hyperparameter_search.py",
            "per_class_comparison_svm_vs_catboost.py", "per_class_calibration_check.py",
            "calibration_method_comparison.py", "confusion_pattern_parity_check.py",
            "svm_diagnose_confusion_pair.py", "tfidf_tokenization_check.py",
            "atomic_vocab_size_check.py",
        ],
        "validated_macro_f1_5x3_repeated_cv": 0.917,
        "validated_calibration_ece": 0.0675,
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {META_OUT}")

    print("\n--- Smoke test: default-threshold role ---")
    smoke_row = pd.DataFrame([{
        TEXT_FEATURE: "python django react postgresql",
        "degree_level": "bachelors", "degree_field": "computer-science",
        "years_experience": 3,
    }])
    X_smoke_text = artifacts["tfidf"].transform(smoke_row[TEXT_FEATURE])
    X_smoke_cat = artifacts["ohe"].transform(smoke_row[CAT_FEATURES])
    X_smoke_num = artifacts["imputer"].transform(smoke_row[NUM_FEATURES])
    X_smoke = hstack([X_smoke_text, X_smoke_cat, csr_matrix(X_smoke_num)]).tocsr()
    proba = model.predict_proba(X_smoke)[0]
    best_idx = int(np.argmax(proba))
    predicted_role = classes[best_idx]
    confidence = float(proba[best_idx])
    print(f"predicted_role={predicted_role}  confidence={confidence:.3f}  "
          f"threshold={threshold_for_role(predicted_role)}  "
          f"is_confident={confidence >= threshold_for_role(predicted_role)}")


if __name__ == "__main__":
    main()