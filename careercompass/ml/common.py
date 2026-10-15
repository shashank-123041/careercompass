"""
ml/common.py — the single source of truth shared by BOTH the training
scripts and the serving layer for the role classifier.

WHY THIS EXISTS: previously the serving modules (svm_inference.py,
role_classifier_inference.py) imported these constants straight out of
train_role_classifier.py — a training SCRIPT that imports CatBoost and
carried hardcoded training paths. That coupling meant the running Flask
app dragged in the whole training script (and CatBoost) just to read a
few constants. Everything that both sides need now lives here, with no
CatBoost dependency:

  - feature column names / target column
  - confidence thresholds, Tier-3 role set, merged-label map
  - load_and_prepare() and compute_sqrt_balanced_weights()
  - threshold_for_role()
  - disambiguate_merged_label() (architecture-agnostic — operates on the
    predicted role string and the user's skill_tags only)

These definitions are moved VERBATIM from train_role_classifier.py and
role_classifier_inference.py — no behavioral change.
"""
import ast
from typing import Optional

import numpy as np
import pandas as pd

from careercompass.config import Config

# ── Data location (see Config.FINAL_DATASET_CSV) ─────────────────────────────
DATA_CSV = Config.FINAL_DATASET_CSV

CONFIDENCE_THRESHOLD = 0.35  # default, for classes with real/semi-real grounding

TIER3_ROLES = {
    "devsecops", "ux-design", "mlops", "devrel", "engineering-manager",
    "technical-writer", "network-engineer", "software-architect", "postgresql-dba",
}
TIER3_CONFIDENCE_THRESHOLD = 0.55

# Roles Config.ROLE_SLUGS defines that this model cannot predict at all —
# not silently dropped; surfaced in meta.json so the serving layer knows to
# route them elsewhere rather than discovering it by trial.
UNROUTABLE_ROLE_SLUGS_NO_DATA = ["devops-engineer", "server-side-game-developer"]

# ai-engineer/ai-data-scientist/machine-learning were merged into one class
# (insufficient data to separate them) — same for android/ios -> mobile.
# The serving layer picks a specific real ROLE_SLUG to route to.
MERGED_LABEL_MAP = {
    "ml-cluster": ["ai-engineer", "ai-data-scientist", "machine-learning"],
    "mobile": ["android", "ios"],
}

TEXT_FEATURE = "skill_tags_text"
CAT_FEATURES = ["degree_level", "degree_field"]
NUM_FEATURES = ["years_experience"]
TARGET = "role_slug"


def load_and_prepare(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["skill_tags_parsed"] = df["skill_tags"].apply(ast.literal_eval)
    df[TEXT_FEATURE] = df["skill_tags_parsed"].apply(lambda tags: " ".join(tags))
    df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce")
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def compute_sqrt_balanced_weights(df: pd.DataFrame) -> dict:
    """
    sqrt-balanced class weighting (softer than full inverse-frequency).
    Replaced auto_class_weights='Balanced' after a direct 4-scheme
    comparison showed it improves macro F1, backend recall AND small-class
    F1 simultaneously on identical data/folds.

    Computed from whatever df is passed in (training data at the call
    site) — NOT hardcoded, so it adapts as class distribution shifts.
    """
    counts = df[TARGET].value_counts()
    n_total = len(df)
    n_classes = len(counts)
    return {c: float(np.sqrt(n_total / (n_classes * cnt))) for c, cnt in counts.items()}


def threshold_for_role(role: str) -> float:
    return TIER3_CONFIDENCE_THRESHOLD if role in TIER3_ROLES else CONFIDENCE_THRESHOLD


# Keyword hints per merged-label candidate, for disambiguate_merged_label()
# below. Reuses the SAME android/ios and general skill-slug vocabulary
# already established in taxonomy/skill_taxonomy.py's ROLE_ALIASES where it
# overlaps, rather than inventing a parallel keyword set.
MERGED_LABEL_DISAMBIGUATION_HINTS = {
    "mobile": {
        "android": ["android", "kotlin", "android-sdk", "jetpack-compose",
                    "android-studio", "java"],
        "ios": ["ios", "swift", "xcode", "swiftui", "objective-c", "cocoapods"],
    },
    "ml-cluster": {
        "machine-learning": ["scikit-learn", "machine-learning", "feature-engineering",
                              "regression", "classification", "xgboost"],
        "ai-engineer": ["tensorflow", "pytorch", "llm", "prompt-engineering",
                         "model-deployment", "mlops", "generative-ai", "langchain"],
        "ai-data-scientist": ["pandas", "numpy", "statistics", "data-visualization",
                               "jupyter", "data-analysis", "r"],
    },
}


def disambiguate_merged_label(merged_label: str, skill_tags: list) -> Optional[str]:
    """
    Returns the best-guess resolved role slug for a merged label, or None
    if there is no hint table for this label at all (caller should fall
    back to MERGED_LABEL_MAP's first candidate in that case).

    Picks the candidate with the MOST keyword overlap against the user's
    own skill_tags. Ties or zero matches both resolve to the first
    candidate in the hint table's own ordering — deterministic, never
    returns None just because the signal was weak.
    """
    hints = MERGED_LABEL_DISAMBIGUATION_HINTS.get(merged_label)
    if not hints:
        return None

    skill_set = set(s.lower() for s in (skill_tags or []))
    scores = {cand: len(skill_set & set(kw)) for cand, kw in hints.items()}
    best_cand = max(scores, key=scores.get)
    return best_cand
