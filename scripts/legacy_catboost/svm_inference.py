"""
ml/svm_inference.py — serving-layer loader for the tuned Linear SVM role
classifier (role_classifier_svm.joblib / role_classifier_svm_meta.json,
produced by train_svm_classifier.py). Phase 4, shadow-mode stage of the
CatBoost -> SVM migration.

WHY A SEPARATE MODULE, NOT A ROLE_CLASSIFIER_INFERENCE.PY REWRITE:
role_classifier_inference.py is the ACTIVE production CatBoost loader —
onboarding/routes.py already depends on it. Per the shadow-mode rollout
plan (run the new model alongside real traffic, logged for comparison,
before any cutover), this needs to run WITHOUT changing what onboarding
actually serves yet. A separate module makes that boundary explicit and
makes the eventual cutover a small, reviewable diff (swap which module
onboarding imports) instead of a rewrite of a file already in production.

REUSES: disambiguate_merged_label() is imported directly from
role_classifier_inference.py, unchanged — confirmed architecture-agnostic
during the Phase 0 migration-cost review (it operates on the resolved
role string and skill_tags overlap, not on how the prediction was
produced). threshold_for_role(), TIER3_ROLES (via meta.json),
MERGED_LABEL_MAP come from train_role_classifier.py — the SAME source of
truth CatBoost's serving path already uses, so both models are always
gated by identical thresholds, whichever one eventually serves.

CRASH SAFETY: mirrors role_classifier_inference.py's own contract — a
missing/failed-to-load artifact logs a warning and sets
_SVM_AVAILABLE = False rather than raising; predict_svm() returns
{"available": False} in that case rather than crashing anything.

CURRENT INTENDED USE (shadow mode — svm_result is NEVER served):
  from careercompass.ml import svm_inference
  svm_inference.log_shadow_prediction(user_id, ml_features, catboost_result)
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from careercompass.config import Config
from careercompass.ml.common import (
    TEXT_FEATURE, CAT_FEATURES, NUM_FEATURES, MERGED_LABEL_MAP, threshold_for_role,
    disambiguate_merged_label,
)

log = logging.getLogger(__name__)

# Paths to the bundled artifact from training/train_svm.py (see
# Config.SVM_MODEL_PATH / SVM_META_PATH — override via .env if needed).
SVM_MODEL_PATH = Path(Config.SVM_MODEL_PATH)
SVM_META_PATH = Path(Config.SVM_META_PATH)

_BUNDLE = None
_META: dict = {}
_SVM_AVAILABLE = False


def _load():
    global _BUNDLE, _META, _SVM_AVAILABLE

    if not SVM_MODEL_PATH.exists() or not SVM_META_PATH.exists():
        log.warning(
            "SVM role classifier not found (model=%s exists=%s, meta=%s exists=%s) — "
            "predict_svm() will return {'available': False} until "
            "train_svm_classifier.py has been run.",
            SVM_MODEL_PATH, SVM_MODEL_PATH.exists(), SVM_META_PATH, SVM_META_PATH.exists(),
        )
        return

    try:
        import joblib
        bundle = joblib.load(SVM_MODEL_PATH)
        meta = json.loads(SVM_META_PATH.read_text())
    except Exception as e:
        log.error("SVM role classifier failed to load (%s) — predict_svm() will "
                  "return {'available': False} until this is fixed.", e, exc_info=True)
        return

    if meta.get("model_type") != "linear_svm":
        log.error("SVM meta.json model_type is %r, expected 'linear_svm' — refusing "
                  "to load a mismatched artifact/meta pair.", meta.get("model_type"))
        return

    _BUNDLE = bundle
    _META = meta
    _SVM_AVAILABLE = True
    log.info("SVM role classifier loaded (shadow mode): %d classes, %d training rows",
             len(meta.get("classes", [])), meta.get("n_training_rows", 0))


_load()


def _transform(ml_features: dict):
    """Reproduces train_svm_classifier.py's build_features() transform
    step EXACTLY — same three fitted preprocessors (bundled together,
    never loaded separately), same column order, same hstack. Any
    divergence from this is a silent train/serve skew bug."""
    row = pd.DataFrame([{
        TEXT_FEATURE: " ".join(ml_features.get("skill_tags", []) or []),
        "degree_level": ml_features.get("degree_level") or "unknown",
        "degree_field": ml_features.get("degree_field") or "unknown",
        "years_experience": ml_features.get("years_experience")
            if isinstance(ml_features.get("years_experience"), (int, float)) else np.nan,
    }])
    X_text = _BUNDLE["tfidf"].transform(row[TEXT_FEATURE])
    X_cat = _BUNDLE["ohe"].transform(row[CAT_FEATURES])
    X_num = _BUNDLE["imputer"].transform(row[NUM_FEATURES])
    return hstack([X_text, X_cat, csr_matrix(X_num)]).tocsr()


def predict_svm(ml_features: dict) -> dict:
    """
    SVM analogue of role_classifier_inference.py's predict_role_for_user()
    — SAME return shape, so shadow comparison (and, eventually, a
    cutover) is drop-in rather than a reinterpretation of the contract.
    """
    if not _SVM_AVAILABLE:
        return {"available": False}

    skill_tags = ml_features.get("skill_tags", []) or []
    if not skill_tags:
        return {"available": False}  # same "no signal" convention as the CatBoost path

    X = _transform(ml_features)
    proba = _BUNDLE["model"].predict_proba(X)[0]
    classes = _BUNDLE["classes"]
    best_idx = int(np.argmax(proba))
    predicted_role = classes[best_idx]
    confidence = float(proba[best_idx])
    threshold = threshold_for_role(predicted_role)

    result = {
        "predicted_role": predicted_role,
        "confidence": confidence,
        "threshold_used": threshold,
        "is_tier3": predicted_role in _META.get("tier3_roles", []),
        "is_confident": confidence >= threshold,
        "fallback_to_heuristic": confidence < threshold,
        "needs_merged_label_disambiguation": predicted_role in MERGED_LABEL_MAP,
        "disambiguation_candidates": MERGED_LABEL_MAP.get(predicted_role, []),
    }

    resolved_role = predicted_role
    if result["needs_merged_label_disambiguation"]:
        resolved = disambiguate_merged_label(predicted_role, skill_tags)
        if resolved is None:
            resolved = result["disambiguation_candidates"][0]
        resolved_role = resolved

    return {
        "available": True,
        **result,
        "resolved_role": resolved_role,
    }

def predict_svm_rankings(ml_features: dict) -> list:
    """
    SVM analogue of role_classifier_inference.py's predict_role_rankings()
    — the FULL ranked list across every class, not just the top-1 pick.
    THIS is what onboarding.html's "Recommended For You" ranked list and
    per-role match-percentage badges actually render from. predict_svm()
    (top-1 only) never touches that UI at all — until this function is
    wired in somewhere, the ranked list stays 100% CatBoost regardless of
    what predict_svm() shadow-logs.

    Merged labels (e.g. "ml-cluster") are resolved to ONE specific
    routable role PER CANDIDATE here, same as the CatBoost version —
    never left as a non-routable label in the returned list, and a
    resolution collision keeps the HIGHER score rather than overwriting
    it (mirrors predict_role_rankings()'s own collision handling).
    """
    if not _SVM_AVAILABLE:
        return []

    skill_tags = ml_features.get("skill_tags", []) or []
    if not skill_tags:
        return []

    X = _transform(ml_features)
    proba = _BUNDLE["model"].predict_proba(X)[0]
    classes = _BUNDLE["classes"]

    resolved_scores: dict = {}
    for cls, score in zip(classes, proba):
        if cls in MERGED_LABEL_MAP:
            resolved = disambiguate_merged_label(cls, skill_tags)
            if resolved is None:
                resolved = MERGED_LABEL_MAP[cls][0]
        else:
            resolved = cls
        resolved_scores[resolved] = max(float(score), resolved_scores.get(resolved, 0.0))

    ranked = sorted(resolved_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"role_slug": slug, "score": score} for slug, score in ranked]

def log_shadow_prediction(user_id, ml_features: dict, catboost_result: dict,
                           catboost_rankings: list = None) -> None:
    """
    Fire-and-forget: runs the SVM model (both top-1 and full rankings)
    alongside an ALREADY-COMPUTED CatBoost top-1 result and, where
    available, its full rankings — never recomputed here, so everything
    is compared on the exact same call. Logs a TOP-3 comparison, not just
    top-1 agreement: with shadow data still thin, a disagreement where
    the other model's pick was still in the top-3 is meaningfully
    different evidence than a totally orthogonal miss, and this captures
    that distinction instead of collapsing it into a binary agree/disagree.

    catboost_rankings is optional (backward compatible with callers that
    only have the top-1 result) — top-3 fields are simply omitted from
    the log when it's not passed.

    Never raises — a logging failure here must not affect onboarding,
    which doesn't see this function's return value.
    """
    try:
        from careercompass.db import interaction_operations as interaction_ops
    except ImportError:
        return

    svm_result = predict_svm(ml_features)
    svm_rankings = predict_svm_rankings(ml_features)

    def top_n(rankings, n=3):
        return [r["role_slug"] for r in (rankings or [])[:n]]

    cb_top3 = top_n(catboost_rankings)
    svm_top3 = top_n(svm_rankings)

    top3_snapshot = None
    if catboost_rankings:
        top3_snapshot = {
            "catboost_top3": cb_top3,
            "svm_top3": svm_top3,
            "top3_overlap_count": len(set(cb_top3) & set(svm_top3)),
            "catboost_top1_in_svm_top3": catboost_result.get("resolved_role") in svm_top3,
            "svm_top1_in_catboost_top3": svm_result.get("resolved_role") in cb_top3,
        }

    try:
        interaction_ops.log_event(
            user_id=user_id, item_type="role_classifier_shadow", item_id="svm_v1",
            event_type="shadow_prediction", source="svm_shadow_v1",
            context_snapshot={
                "catboost": {
                    "available": catboost_result.get("available"),
                    "predicted_role": catboost_result.get("predicted_role"),
                    "resolved_role": catboost_result.get("resolved_role"),
                    "confidence": catboost_result.get("confidence"),
                    "is_confident": catboost_result.get("is_confident"),
                },
                "svm": {
                    "available": svm_result.get("available"),
                    "predicted_role": svm_result.get("predicted_role"),
                    "resolved_role": svm_result.get("resolved_role"),
                    "confidence": svm_result.get("confidence"),
                    "is_confident": svm_result.get("is_confident"),
                },
                "agree": (
                    catboost_result.get("resolved_role") == svm_result.get("resolved_role")
                    if catboost_result.get("available") and svm_result.get("available")
                    else None
                ),
                **({"top3": top3_snapshot} if top3_snapshot else {}),
            },
        )
    except Exception as e:
        log.warning("Failed to log SVM shadow prediction for user %s: %s", user_id, e)

# ── Drop-in interface — matches role_classifier_inference.py's public
#    function names exactly, so any caller can switch models by changing
#    only its import line. See PROJECT_STATUS.md for the cutover record. ──
predict_role_for_user = predict_svm
predict_role_rankings = predict_svm_rankings


def get_confidence_threshold(role_slug: str) -> float:
    """Matches role_classifier_inference.py's signature exactly."""
    return threshold_for_role(role_slug)