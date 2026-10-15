"""
svm_hyperparameter_search.py — Phase 1 tuning pass for the Linear SVM
candidate identified in compare_model_architectures.py (macro F1 0.905
vs CatBoost's 0.745 — see that script for the original 5-model comparison).

WHAT THIS TESTS, AND WHY THESE THREE KNOBS SPECIFICALLY:
  - C (regularization strength): the single highest-leverage SVM
    hyperparameter, untested in the original comparison (that run used
    LinearSVC's default C=1.0). Given how little CatBoost's own tuning
    pass moved the needle (depth/iter/lr grid: +0.023), this is run as
    ONE bounded grid pass, not an open-ended search — see main()'s
    stopping-rule note at the end.
  - TFIDF_MAX_FEATURES: currently hardcoded to 500 in
    compare_model_architectures.py. skill_dictionary.txt suggests a
    controlled vocabulary of ~568 terms exists elsewhere in this project
    (a DIFFERENT, narrower vocabulary than the granular skill_tags text
    feeding this model) — worth checking whether 500 is truncating real
    signal from the actual (likely larger) skill_tags vocabulary.
  - weight_scheme: sample_weight (sqrt-balanced, current) vs
    class_weight='balanced' vs both together. These COMPOSE
    multiplicatively, not additively, so "both" is not simply "more
    balanced" than either alone — this is exactly the kind of
    interaction that motivated compute_sqrt_balanced_weights() replacing
    auto_class_weights='Balanced' for CatBoost in the first place, and
    is untested for this model.

REUSES, DOES NOT REIMPLEMENT: load_and_prepare / compute_sqrt_balanced_weights
/ TEXT_FEATURE / CAT_FEATURES / NUM_FEATURES / TARGET all come directly
from train_role_classifier.py — same discipline as every other diagnostic
script in this project (see check_class_stability.py, diagnose_confusion_pair.py).
build_sklearn_features() is the same TF-IDF+OHE+imputer preprocessing
used in compare_model_architectures.py, parametrized here to accept a
variable max_features instead of the hardcoded constant.

OUTPUT: svm_hyperparameter_search_results.csv — one row per (C,
max_features, weight_scheme) combination, sorted by macro_f1_mean
descending, so the winning config is immediately visible at the top.

USAGE
  python -m scripts.experiments.svm_hyperparameter_search
  Runs 5 (C values) x 3 (max_features values) x 3 (weight schemes) = 45
  configs, each under 5x3 repeated CV (15 fold-fits) = 675 total fits.
  This WILL take a while — progress is printed per config, not just at
  the end, so a long run stays inspectable while it's still going.
"""
import itertools

import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from careercompass.config import Config
from careercompass.ml.common import (
    DATA_CSV, load_and_prepare, compute_sqrt_balanced_weights,
    TEXT_FEATURE, CAT_FEATURES, NUM_FEATURES, TARGET,
)

N_REPEATS = 5
N_FOLDS = 3
SEED_BASE = 42

# ── Grid — see module docstring for why these three knobs specifically ──────
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_FEATURES_GRID = [500, 800, 1200]
WEIGHT_SCHEMES = ["sample_weight_only", "class_weight_only", "both"]

OUTPUT_CSV = str(Config.RESULTS_DIR / "svm_hyperparameter_search_results.csv")

# The untuned baseline already measured in compare_model_architectures.py —
# used at the end to decide whether this grid actually found anything worth
# adopting, per the stopping rule agreed on before running this.
UNTUNED_BASELINE_MACRO_F1 = 0.905


def build_sklearn_features(train_df: pd.DataFrame, val_df: pd.DataFrame, max_features: int):
    """Same preprocessing as compare_model_architectures.py's version,
    parametrized on max_features instead of the hardcoded TFIDF_MAX_FEATURES
    constant — everything else (fit-on-train-only, OHE handle_unknown,
    median imputation) is unchanged."""
    tfidf = TfidfVectorizer(max_features=max_features)
    X_text_train = tfidf.fit_transform(train_df[TEXT_FEATURE])
    X_text_val = tfidf.transform(val_df[TEXT_FEATURE])

    ohe = OneHotEncoder(handle_unknown="ignore")
    X_cat_train = ohe.fit_transform(train_df[CAT_FEATURES])
    X_cat_val = ohe.transform(val_df[CAT_FEATURES])

    imputer = SimpleImputer(strategy="median")
    X_num_train = imputer.fit_transform(train_df[NUM_FEATURES])
    X_num_val = imputer.transform(val_df[NUM_FEATURES])

    X_train = hstack([X_text_train, X_cat_train, csr_matrix(X_num_train)]).tocsr()
    X_val = hstack([X_text_val, X_cat_val, csr_matrix(X_num_val)]).tocsr()
    return X_train, X_val


def make_svm(C: float, weight_scheme: str) -> CalibratedClassifierCV:
    """
    weight_scheme controls how class imbalance is handled:
      - "sample_weight_only": LinearSVC's own class_weight left at None;
        imbalance handled entirely via sqrt-balanced sample_weight at
        .fit() time (the CURRENT/original config, unchanged).
      - "class_weight_only":  LinearSVC(class_weight="balanced"), no
        sample_weight passed at .fit() time.
      - "both": LinearSVC(class_weight="balanced") AND sqrt-balanced
        sample_weight passed together — these COMPOSE (multiply), so
        this is deliberately its own condition, not assumed equivalent
        to either alone.
    Calibration (sigmoid, cv=3) is held FIXED across the whole grid so
    only C / max_features / weight_scheme vary — same "change one thing
    at a time" discipline as every other comparison in this project.
    """
    class_weight = "balanced" if weight_scheme in ("class_weight_only", "both") else None
    base = LinearSVC(C=C, class_weight=class_weight, random_state=SEED_BASE, max_iter=5000)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def run_config_cv(df: pd.DataFrame, C: float, max_features: int, weight_scheme: str) -> pd.DataFrame:
    classes = sorted(df[TARGET].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED_BASE)

    records = []
    for i, (train_idx, val_idx) in enumerate(rskf.split(df, df[TARGET])):
        repeat_num, fold_num = divmod(i, N_FOLDS)
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        X_train, X_val = build_sklearn_features(train_df, val_df, max_features)
        y_train = train_df[TARGET].map(class_to_idx).values

        model = make_svm(C, weight_scheme)
        if weight_scheme in ("sample_weight_only", "both"):
            weights = compute_sqrt_balanced_weights(train_df)
            sample_weight = train_df[TARGET].map(weights).values
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        pred_idx = model.predict(X_val)
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        preds = [idx_to_class[p] for p in pred_idx]

        macro_f1 = f1_score(val_df[TARGET], preds, labels=classes, average="macro", zero_division=0)
        records.append({"repeat": repeat_num, "fold": fold_num, "macro_f1": macro_f1})

    return pd.DataFrame(records)


def main():
    print(f"Loading {DATA_CSV}...")
    df = load_and_prepare(DATA_CSV)
    print(f"{len(df)} rows, {df[TARGET].nunique()} classes\n")

    grid = list(itertools.product(C_GRID, MAX_FEATURES_GRID, WEIGHT_SCHEMES))
    print(f"Testing {len(grid)} configs x {N_REPEATS}x{N_FOLDS} repeated CV "
          f"({len(grid) * N_REPEATS * N_FOLDS} total fold-fits)...\n")

    summary = []
    for idx, (C, max_features, weight_scheme) in enumerate(grid, start=1):
        label = f"C={C} max_features={max_features} weights={weight_scheme}"
        print(f"[{idx}/{len(grid)}] {label}")
        results = run_config_cv(df, C, max_features, weight_scheme)
        mean_f1 = results["macro_f1"].mean()
        std_f1 = results["macro_f1"].std()
        print(f"    -> macro_f1 = {mean_f1:.3f} +/- {std_f1:.3f}\n")
        summary.append({
            "C": C, "max_features": max_features, "weight_scheme": weight_scheme,
            "macro_f1_mean": round(mean_f1, 4), "macro_f1_std": round(std_f1, 4),
        })

    summary_df = pd.DataFrame(summary).sort_values("macro_f1_mean", ascending=False)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 70)
    print("TOP 10 CONFIGS (sorted by macro_f1_mean)")
    print("=" * 70)
    print(summary_df.head(10).to_string(index=False))
    print(f"\nFull grid written to {OUTPUT_CSV}")

    best = summary_df.iloc[0]
    gain = best["macro_f1_mean"] - UNTUNED_BASELINE_MACRO_F1
    print(f"\nUntuned baseline (C=1.0, max_features=500, sample_weight_only): "
          f"{UNTUNED_BASELINE_MACRO_F1:.3f}")
    print(f"Best config found: {best['macro_f1_mean']:.3f} (gain: {gain:+.3f})")
    if abs(gain) < best["macro_f1_std"]:
        print("NOTE: gain is smaller than the winning config's own std — per the")
        print("stopping rule agreed on for this pass, this does NOT justify a")
        print("wider/further grid search. Take this config (or the untuned")
        print("baseline, if simplicity is preferred) into")
        print("per_class_comparison_svm_vs_catboost.py next.")
    else:
        print("Gain exceeds the winning config's std — worth adopting this config.")
        print("Copy its C / max_features / weight_scheme values into")
        print("per_class_comparison_svm_vs_catboost.py's SVM_* constants next.")


if __name__ == "__main__":
    main()