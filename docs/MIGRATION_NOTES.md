# Restructure notes (flat upload → packaged repo)

## Behavior
No functional changes intended. Code was moved and imports rewritten; the
changes below are the only edits beyond that. Verified: all packages import,
Flask app boots, pages/APIs return 200, `python -m training.train_svm`
trains (macro-F1 ≈ 0.91 on the 3-fold sanity check) and `svm_inference` serves.

## Code changes beyond moving files
1. `ml/common.py` (new): shared constants/helpers (thresholds, Tier-3 roles,
   merged-label map, feature names, `load_and_prepare`, sqrt-balanced weights,
   `disambiguate_merged_label`) moved verbatim out of the training script and
   `role_classifier_inference.py`. Serving no longer imports a training script
   and the app no longer needs CatBoost installed.
2. `predict_with_confidence` moved into `ml/catboost_inference.py`
   (was `role_classifier_inference.py` ← `train_role_classifier.py`).
3. `onboarding/routes.py`: no longer imports the CatBoost module at startup
   (it was unused). Rollback = change one import line (comment in file).
4. All hardcoded `C:\Users\Shashank\...` paths, cwd-relative output paths and
   `data/…` strings replaced by `Config.*` paths.
5. `resume_parser.py`: also catches `AttributeError` when `auto_extract` is an
   empty/placeholder module, so the fallback disables cleanly instead of crashing.
6. `templates/profile.html`: `PASTE_ME` Firebase placeholders replaced with the
   same public web config as `login.html` (password change would have failed).
7. `app.py` no longer creates the app at import; `run.py` does.
8. Opportunity bootstrap grounding dataset: `final_dataset_v3.csv` →
   `Config.FINAL_DATASET_CSV` (v7, the dataset the classifier trains on, as the
   original comment instructed).
9. `requirements.txt` split: added `pypdf`; `catboost` + `google-genai` moved to
   `requirements-ml.txt`.

## Renames
| Old | New |
|---|---|
| `test.py` | `scripts/shadow/generate_shadow_data_from_holdout.py` |
| `tire3_extended.py` | `scripts/data_building/tier3_expand.py` |
| `check_class_stability.py` (SVM version) | `scripts/experiments/check_class_stability_svm.py` |
| `train_role_classifier.py` | `training/train_catboost.py` |
| `train_svm_classifier.py` | `training/train_svm.py` |
| `role_classifier_inference.py` | `ml/catboost_inference.py` |
| `sync/heuristic_scorer.py` | `ranking/heuristic_scorer.py` |
| `test.html` | `prototypes/cosmos/index.html` |
| `matched_opportunities.html` | `templates/partials/` |

## Intentionally NOT included
- `.env`, `firebase-service-account.json` — secrets (rotate both: they were shared in plain text).
- `resume_parse_cache.json` (user resume data), `.gemini_budget_state.json`,
  `catboost_training.json`, `learn_error.tsv`, `time_left.tsv`,
  `shadow_log_disagreements.csv`, `opportunity_pseudo_labels*.csv` — generated/runtime files.
- `auto_extract.py`, `run_full_pipeline.py`, `category_alignment.py`,
  `role_profiles.json`, `skill_dictionary.txt` — arrived EMPTY. Drop the real
  `auto_extract.py` + `skill_dictionary.txt` into `careercompass/resume/`.
- Model binaries (`.joblib`, `.cbm`) — not in the upload; retrain (`models/README.md`).

## Scripts referenced in docs but missing from the upload
`compare_model_architectures.py`, `tune_class_weights.py`,
`per_class_calibration_check.py`, `calibration_method_comparison.py`,
`confusion_pattern_parity_check.py`, `svm_diagnose_confusion_pair.py`,
`tfidf_tokenization_check.py`, `atomic_vocab_size_check.py`,
`shadow_log_analysis.py`, `diagnose_shadow_logging.py`,
`cleanup_diagnostic_shadow_rows.py`. Add them under `scripts/experiments/` or
`scripts/shadow/` (import from `careercompass.ml.common`, write to `Config.RESULTS_DIR`).

## Existing local data
- Move your trained `role_classifier/*.joblib|*.cbm` into `models/`.
- Your existing `data/roadmaps.db` works as-is if it sits in the repo's `data/`.
- Scraper cache moved to `data/cache/scrapers/`.
