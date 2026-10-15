"""
ml/catboost_inference.py — serving-layer wrapper for role_classifier.cbm.
(ROLLBACK ONLY — the production classifier is ml/svm_inference.py.)

WHY THIS EXISTS: train_role_classifier.py already defines predict_with_confidence()
— the exact serving interface, feature-engineering, and threshold logic the
model was validated against — but that file is a training SCRIPT (main()
retrains from scratch and rewrites the model file). Importing it directly
into the live Flask app would mean every import pulls in DATA_CSV/MODEL_OUT
path constants meant for a training run, not serving. This module imports
ONLY the pure, side-effect-free pieces (predict_with_confidence, TIER3_ROLES,
MERGED_LABEL_MAP) and loads the ALREADY-TRAINED model from disk once, at
import time — the same "boot once at startup" pattern auth/__init__.py
already uses for the Firebase Admin SDK.

CRASH SAFETY: if role_classifier.cbm or its meta.json aren't present at
Config.ROLE_CLASSIFIER_MODEL_PATH / _META_PATH (e.g. a fresh dev environment
that hasn't run train_role_classifier.py yet), this logs a warning and sets
_MODEL_AVAILABLE = False — it does NOT raise on import. Every public
function below checks that flag first and returns {"available": False}
rather than crashing. onboarding/routes.py falls back to the original
word-overlap heuristic whenever this is the case — see that file's
suggested_roles() for exactly where that fallback triggers.

MERGED-LABEL DISAMBIGUATION: train_role_classifier.py's MERGED_LABEL_MAP
documents that "ml-cluster" and "mobile" aren't real routable ROLE_SLUGS —
they're data-scarcity merges the model can predict but the app can't route
to directly. disambiguate_merged_label() resolves these via skill-keyword
matching against the user's OWN skill_tags (not a coin flip, not always
picking the first candidate) — but ALWAYS resolves to something concrete,
even with zero matching keywords (falls back to the first candidate
deterministically), so onboarding is never blocked waiting on ambiguity
resolution. The full candidate list is still returned in case a future UI
wants to let the user override the auto-pick.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from careercompass.config import Config

log = logging.getLogger(__name__)

# Shared constants/helpers live in ml/common.py (no CatBoost, no training-script
# side effects) — the same module the SVM serving path uses.
from careercompass.ml.common import (  # noqa: E402
    MERGED_LABEL_MAP, TEXT_FEATURE, CAT_FEATURES, NUM_FEATURES, TIER3_ROLES,
    threshold_for_role, disambiguate_merged_label,
)


def predict_with_confidence(model: CatBoostClassifier, skill_tags: list, degree_level: str,
                             degree_field: str, years_experience) -> dict:
    """
    THE CATBOOST SERVING INTERFACE (moved here unchanged from
    train_role_classifier.py so serving no longer imports a training script).
    Uses a STRICTER threshold automatically for Tier 3 predictions (see
    TIER3_ROLES) — the caller doesn't need to know which roles are Tier 3.
    """
    row = pd.DataFrame([{
        TEXT_FEATURE: " ".join(skill_tags),
        "degree_level": degree_level or "unknown",
        "degree_field": degree_field or "unknown",
        "years_experience": years_experience if isinstance(years_experience, (int, float)) else np.nan,
    }])
    pool = Pool(row[[TEXT_FEATURE] + CAT_FEATURES + NUM_FEATURES],
                text_features=[TEXT_FEATURE], cat_features=CAT_FEATURES)
    probs = model.predict_proba(pool)[0]
    classes = model.classes_
    best_idx = probs.argmax()
    predicted_role = str(classes[best_idx])
    confidence = float(probs[best_idx])
    threshold = threshold_for_role(predicted_role)

    return {
        "predicted_role": predicted_role,
        "confidence": confidence,
        "threshold_used": threshold,
        "is_tier3": predicted_role in TIER3_ROLES,
        "is_confident": confidence >= threshold,
        "fallback_to_heuristic": confidence < threshold,
        "needs_merged_label_disambiguation": predicted_role in MERGED_LABEL_MAP,
        "disambiguation_candidates": MERGED_LABEL_MAP.get(predicted_role, []),
    }


# ── Model loading — once, at import time ────────────────────────────────────

_MODEL = None
_META: dict = {}
_MODEL_AVAILABLE = False


def _load():
    global _MODEL, _META, _MODEL_AVAILABLE
    model_path = Path(Config.ROLE_CLASSIFIER_MODEL_PATH)
    meta_path = Path(Config.ROLE_CLASSIFIER_META_PATH)

    if not model_path.exists() or not meta_path.exists():
        log.warning(
            "Role classifier not found (model=%s exists=%s, meta=%s exists=%s) — "
            "onboarding will use the word-overlap heuristic fallback until "
            "train_role_classifier.py has been run and its output is present "
            "at these paths (see Config.ROLE_CLASSIFIER_MODEL_PATH / _META_PATH).",
            model_path, model_path.exists(), meta_path, meta_path.exists(),
        )
        return

    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        log.error("Role classifier failed to load (%s) — falling back to the "
                  "word-overlap heuristic for all onboarding role suggestions "
                  "until this is fixed.", e, exc_info=True)
        return

    _MODEL = model
    _META = meta
    _MODEL_AVAILABLE = True
    log.info("Role classifier loaded: %d classes, %d training rows",
             len(meta.get("classes", [])), meta.get("n_training_rows", 0))


_load()


# ── Public serving interface ────────────────────────────────────────────────

def predict_role_for_user(ml_features: dict) -> dict:
    """
    THE function onboarding/routes.py calls. Takes the ml_features block
    ALREADY present in a user's stored resume_skills (see resume_parser.py's
    PROMPT — skill_tags/degree_level/degree_field/years_experience, the
    exact 4-field contract the model was trained on).

    Returns a dict that's always safe to consume regardless of model
    availability:
      {"available": False}  — model not loaded, caller must use the
                               word-overlap heuristic entirely.
      {"available": True, "predicted_role": ..., "confidence": ...,
       "is_tier3": ..., "is_confident": ..., "fallback_to_heuristic": ...,
       "needs_merged_label_disambiguation": ...,
       "disambiguation_candidates": [...],
       "resolved_role": ...}  — resolved_role is the FINAL routable slug
                                (post-disambiguation if needed) — this is
                                what the caller should actually use to set
                                match_hint, not predicted_role directly,
                                since predicted_role can be a non-routable
                                merged label like "ml-cluster".
    """
    if not _MODEL_AVAILABLE:
        return {"available": False}

    skill_tags = ml_features.get("skill_tags", []) or []
    degree_level = ml_features.get("degree_level", "unknown")
    degree_field = ml_features.get("degree_field", "unknown")
    years_experience = ml_features.get("years_experience", "unknown")
    if not isinstance(years_experience, (int, float)):
        years_experience = None  # predict_with_confidence expects a number or NaN-safe value

    if not skill_tags:
        # No skills at all — technically callable, but a prediction from
        # zero signal isn't meaningful. Treat the same as "unavailable"
        # for THIS call so the caller falls back to the heuristic (which
        # itself will also find nothing to match on, but at least won't
        # report false confidence).
        return {"available": False}

    result = predict_with_confidence(_MODEL, skill_tags, degree_level, degree_field, years_experience)

    resolved_role = result["predicted_role"]
    if result["needs_merged_label_disambiguation"]:
        resolved = disambiguate_merged_label(result["predicted_role"], skill_tags)
        if resolved is None:
            # No hint table for this merged label at all — deterministic
            # fallback to MERGED_LABEL_MAP's first listed candidate rather
            # than leaving onboarding with no concrete suggestion.
            resolved = result["disambiguation_candidates"][0]
        resolved_role = resolved

    return {
        "available": True,
        "predicted_role": result["predicted_role"],
        "confidence": result["confidence"],
        "is_tier3": result["is_tier3"],
        "is_confident": result["is_confident"],
        "fallback_to_heuristic": result["fallback_to_heuristic"],
        "needs_merged_label_disambiguation": result["needs_merged_label_disambiguation"],
        "disambiguation_candidates": result["disambiguation_candidates"],
        "resolved_role": resolved_role,
    }


def predict_role_rankings(ml_features: dict) -> list:
    """
    Returns a FULL ranked list of {role_slug, score} across every class the
    classifier was trained on, sorted descending — NOT just the single
    top-1 prediction predict_role_for_user() returns. This is what powers
    the "ranked roles with match percentage" onboarding UI: every role the
    model has an opinion on, not only its single best guess.

    Merged labels ("mobile", "ml-cluster") are resolved to ONE specific
    disambiguated candidate via the SAME disambiguate_merged_label() logic
    used for the top-1 case, so the ranked list only ever contains real,
    routable role slugs that actually have a roadmap DB row — never a
    non-existent "mobile" or "ml-cluster" card.

    Returns [] under the same availability contract as predict_role_for_user
    (model not loaded, or no skill_tags to score against) — callers should
    treat an empty list as "no ranking available," not as "every role
    scored zero."
    """
    if not _MODEL_AVAILABLE:
        return []

    skill_tags = ml_features.get("skill_tags", []) or []
    if not skill_tags:
        return []

    degree_level = ml_features.get("degree_level", "unknown")
    degree_field = ml_features.get("degree_field", "unknown")
    years_experience = ml_features.get("years_experience", "unknown")
    if not isinstance(years_experience, (int, float)):
        years_experience = float("nan")

    row = pd.DataFrame([{
        TEXT_FEATURE: " ".join(skill_tags),
        "degree_level": degree_level or "unknown",
        "degree_field": degree_field or "unknown",
        "years_experience": years_experience,
    }])
    pool = Pool(row[[TEXT_FEATURE] + CAT_FEATURES + NUM_FEATURES],
                text_features=[TEXT_FEATURE], cat_features=CAT_FEATURES)
    probs = _MODEL.predict_proba(pool)[0]
    classes = _MODEL.classes_

    resolved_scores: dict = {}
    for cls, score in zip(classes, probs):
        if cls in MERGED_LABEL_MAP:
            resolved = disambiguate_merged_label(cls, skill_tags)
            if resolved is None:
                resolved = MERGED_LABEL_MAP[cls][0]
        else:
            resolved = cls
        # If two merged groups somehow resolved to the same slug (not
        # possible with the current MERGED_LABEL_MAP, but not assumed away
        # permanently), keep the HIGHER score rather than silently
        # overwriting — a role should never lose score because of
        # resolution collisions.
        resolved_scores[resolved] = max(float(score), resolved_scores.get(resolved, 0.0))

    ranked = sorted(resolved_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"role_slug": slug, "score": score} for slug, score in ranked]


def get_confidence_threshold(role_slug: str) -> float:
    """
    Returns the SAME per-role confidence threshold train_role_classifier.py's
    threshold_for_role() uses (0.35 default, 0.55 for Tier 3 roles) — reused
    here (via the loaded meta.json, not re-imported from the training script,
    since this needs to work even if _MODEL_AVAILABLE fell back to a stale
    meta) so the frontend's "Strong Match" / "Possible Match" tiering is
    grounded in the SAME calibrated thresholds the model's own confidence
    gating uses, not an arbitrary separately-chosen UI cutoff.
    """
    if not _MODEL_AVAILABLE:
        return 0.35
    if role_slug in _META.get("tier3_roles", []):
        return _META.get("confidence_threshold_tier3", 0.55)
    return _META.get("confidence_threshold_default", 0.35)