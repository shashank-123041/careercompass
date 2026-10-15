"""
check_class_stability_svm.py — SVM analogue of check_class_stability.py,
the last genuinely CatBoost-locked diagnostic tool (build_pool/make_model
are direct CatBoost calls; find_thin_qa_rows.py needed no fork at all —
it never calls a model — and diagnose_confusion_pair.py was already
forked as svm_diagnose_confusion_pair.py).

WHY THIS EXISTS, IN THE SVM'S CASE: check_class_stability.py measures
how much each class's F1 swings from RANDOM FOLD/SEED VARIATION ALONE,
with zero data changes — the answer to "is an apparent regression real,
or just what this class's score looks like across different random
splits." Every validation script so far (svm_hyperparameter_search.py,
per_class_comparison_svm_vs_catboost.py, etc.) already used 5x3 repeated
CV, which is MORE robust than the single-3-fold runs this diagnostic was
originally built to distrust — but none of them reported per-class
STABILITY (std across repeats) in this exact, dedicated format, matched
to the SAME class_stability_report.csv shape CatBoost's own report used,
for a direct side-by-side comparison.

METHOD: identical to check_class_stability.py's — RepeatedStratifiedKFold,
same 3-fold x 5-repeat = 15 fold-fits, per-class F1 collected from every
fold, mean/std reported per class, sorted by std (most volatile first),
with the SAME FLAGGED_CLASSES highlighted first.

USAGE
  python -m scripts.experiments.check_class_stability_svm
  Reads DATA_CSV from careercompass/ml/common.py (Config.FINAL_DATASET_CSV), same as
  the CatBoost version.
"""
import numpy as np
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

# ── Final locked SVM config — see train_svm_classifier.py's docstring
#    for the full evidence trail behind each of these values ────────────────
SVM_C = 1.0
SVM_MAX_FEATURES = 800
SVM_WEIGHT_SCHEME = "sample_weight_only"
SVM_CALIBRATION_METHOD = "isotonic"

# Same classes CatBoost's own stability check flagged — kept identical so
# the two reports are directly comparable, not just similarly-shaped.
FLAGGED_CLASSES = [
    "engineering-manager", "game-developer", "devrel", "software-architect",
    "data-analyst", "product-manager", "backend", "mobile",
]

OUTPUT_CSV = str(Config.RESULTS_DIR / "class_stability_report_svm.csv")


def build_sklearn_features(train_df, val_df, max_features):
    tfidf = TfidfVectorizer(max_features=max_features)  # default tokenizer — validated choice
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


def make_svm() -> CalibratedClassifierCV:
    class_weight = "balanced" if SVM_WEIGHT_SCHEME in ("class_weight_only", "both") else None
    base = LinearSVC(C=SVM_C, class_weight=class_weight, random_state=SEED_BASE, max_iter=5000)
    return CalibratedClassifierCV(base, method=SVM_CALIBRATION_METHOD, cv=3)


def run_repeated_cv(df: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(df[TARGET].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED_BASE)

    records = []
    for i, (train_idx, val_idx) in enumerate(rskf.split(df, df[TARGET])):
        repeat_num, fold_num = divmod(i, N_FOLDS)
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

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
        preds = [idx_to_class[p] for p in pred_idx]

        per_class_f1 = f1_score(val_df[TARGET], preds, labels=classes, average=None, zero_division=0)
        macro_f1 = per_class_f1.mean()

        print(f"  repeat {repeat_num+1}/{N_REPEATS} fold {fold_num+1}/{N_FOLDS}: "
              f"macro_f1={macro_f1:.3f}  (val size={len(val_df)})")

        for cls, f1 in zip(classes, per_class_f1):
            records.append({"repeat": repeat_num, "fold": fold_num, "class": cls, "f1": f1})

    return pd.DataFrame(records)


def summarize(results: pd.DataFrame, support: pd.Series):
    summary = (
        results.groupby("class")["f1"]
        .agg(["mean", "std", "min", "max"])
        .round(3)
    )
    summary["support"] = support
    summary = summary.sort_values("std", ascending=False)

    print("\n" + "=" * 70)
    print(f"SVM PER-CLASS F1 STABILITY across {N_REPEATS}x{N_FOLDS} = {N_REPEATS*N_FOLDS} fold-fits")
    print("=" * 70)
    print("\nSorted by std (most volatile first) — a HIGH std here means this")
    print("class's F1 swings a lot from fold-luck ALONE, with no data changes:\n")
    print(summary.to_string())

    print("\n" + "-" * 70)
    print("FLAGGED CLASSES (same set CatBoost's own stability check flagged —")
    print("kept identical for direct comparison against class_stability_report.csv):")
    print("-" * 70)
    flagged = summary.loc[summary.index.intersection(FLAGGED_CLASSES)]
    print(flagged.to_string())

    macro_by_fold = results.groupby(["repeat", "fold"])["f1"].mean()
    print(f"\nOverall macro F1 across all {len(macro_by_fold)} fold-fits: "
          f"mean={macro_by_fold.mean():.3f}  std={macro_by_fold.std():.3f}")

    return summary


def main():
    print(f"Loading {DATA_CSV}...")
    df = load_and_prepare(DATA_CSV)
    support = df[TARGET].value_counts()
    print(f"{len(df)} rows, {df[TARGET].nunique()} classes\n")

    print(f"Running {N_REPEATS}x repeated {N_FOLDS}-fold stratified CV "
          f"({N_REPEATS*N_FOLDS} total fold-fits)...")
    results = run_repeated_cv(df)

    summary = summarize(results, support)
    summary.to_csv(OUTPUT_CSV)
    print(f"\nFull per-class stability report written to {OUTPUT_CSV}")
    print("Compare directly against class_stability_report.csv (CatBoost's own) —")
    print("a class with a LOWER std here than in that file is more stable under")
    print("SVM than it ever was under CatBoost, independent of the mean F1 gain")
    print("already established.")


if __name__ == "__main__":
    main()