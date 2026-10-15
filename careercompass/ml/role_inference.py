"""
ml/role_inference.py — serving layer for the tuned Linear SVM role classifier
(role_classifier_svm.joblib / role_classifier_svm_meta.json from
train_svm_classifier.py). PRODUCTION model as of the Option B cutover.

Replaces BOTH ml/svm_inference.py and ml/role_classifier_inference.py.
Deliberately self-contained: no imports from the training scripts and no
catboost dependency, so the deployed app needs only scikit-learn + joblib.
Constants and disambiguation logic below are copied unchanged from the
previously validated modules (thresholds, MERGED_LABEL_MAP, hint tables).

CRASH SAFETY: a missing/failed artifact logs and sets _AVAILABLE = False;
every public function then returns {"available": False} / [] and
onboarding falls back to its word-overlap heuristic.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

#from config import Config
from careercompass.config import Config

log = logging.getLogger(__name__)

# ── Feature schema (must match train_svm_classifier.py) ─────────────────────
TEXT_FEATURE = "skill_tags_text"
CAT_FEATURES = ["degree_level", "degree_field"]
NUM_FEATURES = ["years_experience"]

# ── Confidence thresholds (same values the 22/22 calibration check used) ────
CONFIDENCE_THRESHOLD = 0.35
TIER3_CONFIDENCE_THRESHOLD = 0.55
TIER3_ROLES = frozenset({
    "devsecops", "ux-design", "mlops", "devrel", "engineering-manager",
    "technical-writer", "network-engineer", "software-architect", "postgresql-dba",
})

# Data-scarcity merges: the model predicts these, the app can't route to them.
MERGED_LABEL_MAP = {
    "ml-cluster": ["ai-engineer", "ai-data-scientist", "machine-learning"],
    "mobile": ["android", "ios"],
}

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


def threshold_for_role(role: str) -> float:
    return TIER3_CONFIDENCE_THRESHOLD if role in TIER3_ROLES else CONFIDENCE_THRESHOLD


def get_confidence_threshold(role_slug: str) -> float:
    return threshold_for_role(role_slug)


def disambiguate_merged_label(merged_label: str, skill_tags: list) -> Optional[str]:
    """Best-overlap candidate for a merged label; None if no hint table.
    NOTE (preserved behavior): ties/zero-overlap resolve to the FIRST entry
    of the hint table above, not MERGED_LABEL_MAP's first entry."""
    hints = MERGED_LABEL_DISAMBIGUATION_HINTS.get(merged_label)
    if not hints:
        return None
    skill_set = set(s.lower() for s in (skill_tags or []))
    scores = {cand: len(skill_set & set(kw)) for cand, kw in hints.items()}
    return max(scores, key=scores.get)


def _resolve(label: str, skill_tags: list) -> str:
    if label not in MERGED_LABEL_MAP:
        return label
    return disambiguate_merged_label(label, skill_tags) or MERGED_LABEL_MAP[label][0]


# ── Artifact loading — once, at import ──────────────────────────────────────
_BUNDLE = None
_META: dict = {}
_AVAILABLE = False


def _load():
    global _BUNDLE, _META, _AVAILABLE
    model_path = Path(Config.ROLE_CLASSIFIER_MODEL_PATH)
    meta_path = Path(Config.ROLE_CLASSIFIER_META_PATH)

    if not model_path.exists() or not meta_path.exists():
        log.warning("Role classifier not found (model=%s exists=%s, meta=%s exists=%s) — "
                    "onboarding will use the word-overlap heuristic fallback.",
                    model_path, model_path.exists(), meta_path, meta_path.exists())
        return
    try:
        import joblib
        bundle = joblib.load(model_path)
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        log.error("Role classifier failed to load (%s) — using heuristic fallback.",
                  e, exc_info=True)
        return

    if meta.get("model_type") != "linear_svm":
        log.error("meta.json model_type is %r, expected 'linear_svm' — refusing to "
                  "load a mismatched artifact/meta pair.", meta.get("model_type"))
        return

    _BUNDLE, _META, _AVAILABLE = bundle, meta, True
    log.info("Role classifier loaded: %d classes, %d training rows",
             len(meta.get("classes", [])), meta.get("n_training_rows", 0))


_load()


def _scores(ml_features: dict):
    """Returns (classes, proba, skill_tags) or None if unavailable / no signal.
    Reproduces train_svm_classifier.py's feature transform EXACTLY (same three
    bundled preprocessors, same column order) — divergence = train/serve skew."""
    if not _AVAILABLE:
        return None
    skill_tags = ml_features.get("skill_tags", []) or []
    if not skill_tags:
        return None

    yrs = ml_features.get("years_experience")
    row = pd.DataFrame([{
        TEXT_FEATURE: " ".join(skill_tags),
        "degree_level": ml_features.get("degree_level") or "unknown",
        "degree_field": ml_features.get("degree_field") or "unknown",
        "years_experience": yrs if isinstance(yrs, (int, float)) else np.nan,
    }])
    X = hstack([
        _BUNDLE["tfidf"].transform(row[TEXT_FEATURE]),
        _BUNDLE["ohe"].transform(row[CAT_FEATURES]),
        csr_matrix(_BUNDLE["imputer"].transform(row[NUM_FEATURES])),
    ]).tocsr()
    proba = _BUNDLE["model"].predict_proba(X)[0]
    return _BUNDLE["classes"], proba, skill_tags


# ── Public serving interface ────────────────────────────────────────────────
def predict_role_for_user(ml_features: dict) -> dict:
    """Top-1 prediction. {"available": False} if the model can't score this."""
    s = _scores(ml_features)
    if s is None:
        return {"available": False}
    classes, proba, skill_tags = s

    best = int(np.argmax(proba))
    predicted = classes[best]
    confidence = float(proba[best])
    threshold = threshold_for_role(predicted)
    needs_disambig = predicted in MERGED_LABEL_MAP

    return {
        "available": True,
        "predicted_role": predicted,
        "confidence": confidence,
        "threshold_used": threshold,
        "is_tier3": predicted in TIER3_ROLES,
        "is_confident": confidence >= threshold,
        "fallback_to_heuristic": confidence < threshold,
        "needs_merged_label_disambiguation": needs_disambig,
        "disambiguation_candidates": MERGED_LABEL_MAP.get(predicted, []),
        "resolved_role": _resolve(predicted, skill_tags),
    }


def predict_role_rankings(ml_features: dict) -> list:
    """Full ranked [{role_slug, score}] across all classes, merged labels
    resolved to ONE routable slug (collision keeps the higher score).
    [] means 'no ranking available', not 'everything scored zero'."""
    s = _scores(ml_features)
    if s is None:
        return []
    classes, proba, skill_tags = s

    resolved_scores: dict = {}
    for cls, score in zip(classes, proba):
        slug = _resolve(cls, skill_tags)
        resolved_scores[slug] = max(float(score), resolved_scores.get(slug, 0.0))

    ranked = sorted(resolved_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"role_slug": slug, "score": score} for slug, score in ranked]