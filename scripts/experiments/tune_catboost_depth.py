"""
tune_catboost_depth.py — tests whether CatBoost's underperformance
against Random Forest / XGBoost / Logistic Regression (see
compare_model_architectures.py's result: CatBoost 0.745 vs Logistic
Regression 0.885) is explained by its hyperparameters having never been
tuned — every experiment this whole project has run reused the exact
same depth=4, iterations=150, learning_rate=0.08 configuration.

WHY THIS MATTERS BEFORE CHANGING THE MODEL CHOICE: the comparison showed
a clear pattern — every alternative model had MORE tree depth / more
expressive capacity than CatBoost's depth=4, and they won in roughly
that order (Random Forest depth=12 > XGBoost depth=6 > CatBoost depth=4;
Logistic Regression's linear-over-sparse-TF-IDF representation is a
separate, also-plausible phenomenon for short, clean skill-tag text).
This is consistent with "CatBoost was undertuned," not "gradient
boosting is inherently worse here." That hypothesis is directly
testable, and should be tested BEFORE concluding the model choice
itself needs to change.

METHOD: a SMALL grid search (cheap, single 3-fold CV per candidate —
NOT full 5x3 repeated CV, to keep this exploratory pass affordable)
across depth and iterations. Whichever candidate wins gets a full 5x3
repeated CV confirmation run afterward — the same two-stage discipline
(cheap search, then rigorous confirmation of the winner only) used
implicitly elsewhere in this project to avoid spending full repeated-CV
budget on every candidate up front.

USAGE
  python -m scripts.experiments.tune_catboost_depth
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

from training.train_catboost import build_pool
from careercompass.ml.common import (
    DATA_CSV, load_and_prepare, compute_sqrt_balanced_weights, TARGET,
)

N_FOLDS = 3
SEED = 42

# Grid: current production config first (as the baseline to beat), then
# progressively deeper/longer configs consistent with the pattern seen
# in compare_model_architectures.py (RF depth=12, XGBoost depth=6 both
# beat CatBoost depth=4).
GRID = [
    {"depth": 4, "iterations": 150, "learning_rate": 0.08},   # current production config
    {"depth": 6, "iterations": 150, "learning_rate": 0.08},
    {"depth": 6, "iterations": 300, "learning_rate": 0.05},
    {"depth": 8, "iterations": 300, "learning_rate": 0.05},
    {"depth": 8, "iterations": 500, "learning_rate": 0.03},
]


def make_model_variant(class_weights: dict, depth: int, iterations: int, learning_rate: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        loss_function="MultiClass", class_weights=class_weights,
        verbose=False, random_seed=SEED,
    )


def evaluate_config(df: pd.DataFrame, config: dict) -> tuple:
    classes = sorted(df[TARGET].unique())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for train_idx, val_idx in skf.split(df, df[TARGET]):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        weights = compute_sqrt_balanced_weights(train_df)
        model = make_model_variant(weights, **config)
        model.fit(build_pool(train_df))
        preds = model.predict(build_pool(val_df)).flatten()
        macro_f1 = f1_score(val_df[TARGET], preds, labels=classes, average="macro", zero_division=0)
        fold_scores.append(macro_f1)

    return sum(fold_scores) / len(fold_scores), fold_scores


def main():
    print(f"Loading {DATA_CSV}...")
    df = load_and_prepare(DATA_CSV)
    print(f"{len(df)} rows, {df[TARGET].nunique()} classes\n")

    print("Testing CatBoost hyperparameter configs (single 3-fold CV each — exploratory pass):\n")
    results = []
    for config in GRID:
        mean_f1, fold_scores = evaluate_config(df, config)
        label = f"depth={config['depth']} iter={config['iterations']} lr={config['learning_rate']}"
        print(f"  {label:45s} macro_f1={mean_f1:.3f}  (folds: {[round(f,3) for f in fold_scores]})")
        results.append({"config": config, "label": label, "mean_f1": mean_f1})

    results.sort(key=lambda r: r["mean_f1"], reverse=True)
    best = results[0]
    baseline = next(r for r in results if r["config"] == GRID[0])

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  Current production config: {baseline['label']} -> macro_f1={baseline['mean_f1']:.3f}")
    print(f"  Best config found:         {best['label']} -> macro_f1={best['mean_f1']:.3f}")
    print(f"  Improvement: {best['mean_f1'] - baseline['mean_f1']:+.3f}")
    print(f"\n  For reference, from compare_model_architectures.py (5x3 repeated CV):")
    print(f"    catboost (depth=4, untuned):  0.745")
    print(f"    xgboost:                      0.834")
    print(f"    random_forest:                0.838")
    print(f"    logistic_regression:          0.885")

    if best["mean_f1"] > baseline["mean_f1"] + 0.03:
        print(f"\n  Meaningful improvement found. NEXT STEP: run the winning config")
        print(f"  ({best['label']}) through the FULL 5x3 repeated CV (same protocol as")
        print(f"  check_class_stability.py) for a trustworthy, noise-robust confirmation")
        print(f"  before deciding whether this closes the gap to the baselines above.")
    else:
        print(f"\n  No meaningful improvement from hyperparameter tuning alone. This weakens")
        print(f"  the 'CatBoost was just undertuned' hypothesis and strengthens the case")
        print(f"  that the baselines' advantage is real, not a tuning artifact.")


if __name__ == "__main__":
    main()