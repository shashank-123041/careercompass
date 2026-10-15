"""
per_class_comparison_svm_vs_catboost.py — Phase 0.2 deliverable: per-class
F1 for CatBoost vs the tuned Linear SVM, side by side, under identical
5x3 repeated stratified CV.

WHY THIS EXISTS: compare_model_architectures.py and
svm_hyperparameter_search.py both report only AGGREGATE macro F1. That
number alone can't tell you whether SVM's +0.16 over CatBoost is spread
evenly across classes or concentrated in the large, already-easy ones
(backend, data-engineer) while leaving the Tier-3 / thin classes
(devrel, postgresql-dba, ux-design, etc. — see TIER3_ROLES in
train_role_classifier.py) unchanged or worse. Since those thin classes
are exactly the ones gated by TIER3_CONFIDENCE_THRESHOLD in production,
this is the deciding evidence for whether replacing CatBoost is a net
win where it actually matters, not just on average.

METHOD: mirrors check_class_stability.py's repeated-CV -> out-of-fold
per-class-F1 pattern, run once per model, on the SAME fold splits (same
random_state) so the two models are compared on identical validation
sets fold-for-fold — not just under the same CV *scheme*, but the exact
same rows in each fold.

BEFORE RUNNING: if you ran svm_hyperparameter_search.py, copy its
winning row's C / max_features / weight_scheme into the SVM_* constants
below. Left at their defaults, this compares CatBoost against the
UNTUNED SVM config (C=1.0, max_features=500, sample_weight_only) already
measured in compare_model_architectures.py.

USAGE
  python -m scripts.experiments.per_class_comparison_svm_vs_catboost
"""
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from training.train_catboost import build_pool, make_model
from careercompass.config import Config
from careercompass.ml.common import (
    DATA_CSV, load_and_prepare, compute_sqrt_balanced_weights, TEXT_FEATURE, CAT_FEATURES, NUM_FEATURES,
    TARGET, TIER3_ROLES,
)

N_REPEATS = 5
N_FOLDS = 3
SEED_BASE = 42

# ── Fill these in from svm_hyperparameter_search.py's top row ───────────────
# Defaults below = the UNTUNED config already tested in
# compare_model_architectures.py (C=1.0, max_features=500, sample_weight
# only) — update after running the tuning pass.
SVM_C = 1.0
SVM_MAX_FEATURES = 800
SVM_WEIGHT_SCHEME = "sample_weight_only"   # "sample_weight_only" | "class_weight_only" | "both"

OUTPUT_CSV = str(Config.RESULTS_DIR / "per_class_f1_comparison.csv")


def build_sklearn_features(train_df, val_df, max_features):
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


def make_svm():
    class_weight = "balanced" if SVM_WEIGHT_SCHEME in ("class_weight_only", "both") else None
    base = LinearSVC(C=SVM_C, class_weight=class_weight, random_state=SEED_BASE, max_iter=5000)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def run_repeated_cv(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Same record shape as check_class_stability.py's run_repeated_cv —
    per (repeat, fold, class) F1 — but dispatches to either CatBoost or
    the SVM pipeline depending on model_name, using the SAME fold splits
    (identical random_state) for both, so per-class numbers are directly
    comparable fold-for-fold, not just under the same CV scheme."""
    classes = sorted(df[TARGET].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED_BASE)

    records = []
    for i, (train_idx, val_idx) in enumerate(rskf.split(df, df[TARGET])):
        repeat_num, fold_num = divmod(i, N_FOLDS)
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        if model_name == "catboost":
            weights = compute_sqrt_balanced_weights(train_df)
            model = make_model(class_weights=weights)
            model.fit(build_pool(train_df))
            preds = model.predict(build_pool(val_df)).flatten()
        else:  # "svm"
            X_train, X_val = build_sklearn_features(train_df, val_df, SVM_MAX_FEATURES)
            y_train = train_df[TARGET].map(class_to_idx).values
            model = make_svm()
            if SVM_WEIGHT_SCHEME in ("sample_weight_only", "both"):
                weights = compute_sqrt_balanced_weights(train_df)
                sample_weight = train_df[TARGET].map(weights).values
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
            pred_idx = model.predict(X_val)
            idx_to_class = {i: c for c, i in class_to_idx.items()}
            preds = [idx_to_class[p] for p in pred_idx]

        per_class_f1 = f1_score(val_df[TARGET], preds, labels=classes, average=None, zero_division=0)
        print(f"  [{model_name}] repeat {repeat_num+1}/{N_REPEATS} fold {fold_num+1}/{N_FOLDS}: "
              f"macro_f1={per_class_f1.mean():.3f}")

        for cls, f1 in zip(classes, per_class_f1):
            records.append({"repeat": repeat_num, "fold": fold_num, "class": cls, "f1": f1})

    return pd.DataFrame(records)


def main():
    print(f"Loading {DATA_CSV}...")
    df = load_and_prepare(DATA_CSV)
    support = df[TARGET].value_counts()
    print(f"{len(df)} rows, {df[TARGET].nunique()} classes\n")

    print("Running CatBoost (production config)...")
    cb_results = run_repeated_cv(df, "catboost")
    cb_summary = cb_results.groupby("class")["f1"].mean().round(3)

    print(f"\nRunning SVM (C={SVM_C}, max_features={SVM_MAX_FEATURES}, "
          f"weights={SVM_WEIGHT_SCHEME})...")
    svm_results = run_repeated_cv(df, "svm")
    svm_summary = svm_results.groupby("class")["f1"].mean().round(3)

    comparison = pd.DataFrame({
        "catboost_f1": cb_summary,
        "svm_f1": svm_summary,
    })
    comparison["diff"] = (comparison["svm_f1"] - comparison["catboost_f1"]).round(3)
    comparison["support"] = support
    comparison["is_tier3"] = comparison.index.isin(TIER3_ROLES)
    comparison = comparison.sort_values("diff")

    print("\n" + "=" * 78)
    print("PER-CLASS F1: CATBOOST vs SVM (sorted by diff, worst-for-SVM first)")
    print("=" * 78)
    print(comparison.to_string())

    tier3_rows = comparison[comparison["is_tier3"]]
    print("\n" + "-" * 78)
    print("TIER-3 CLASSES ONLY (gated by TIER3_CONFIDENCE_THRESHOLD in production —")
    print("this subset is the actual Phase 0.2 decision evidence):")
    print("-" * 78)
    print(tier3_rows.to_string())

    n_worse = (comparison["diff"] < 0).sum()
    n_tier3_worse = (tier3_rows["diff"] < 0).sum()
    print(f"\nSVM worse than CatBoost on {n_worse}/{len(comparison)} classes overall, "
          f"{n_tier3_worse}/{len(tier3_rows)} of them Tier-3.")
    if n_tier3_worse > 0:
        print("NOTE: per the Phase 0 criterion, a Tier-3 regression here weakens")
        print("the case for switching regardless of the aggregate macro F1 win —")
        print("inspect these specific classes with diagnose_confusion_pair.py")
        print("before finalizing the replace decision.")
    else:
        print("SVM matches or beats CatBoost on every Tier-3 class — this clears")
        print("the Phase 0.2 criterion; proceed to Phase 2 (calibration check).")

    comparison.to_csv(OUTPUT_CSV)
    print(f"\nWritten to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()