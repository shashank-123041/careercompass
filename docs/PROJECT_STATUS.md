# Project Status — as of Step 1 (Auth + Roadmap + Server-side Progress merge)

> **Repo restructure note:** file paths in this document use the pre-restructure flat layout. See `MIGRATION_NOTES.md` for the old→new mapping.


This is an honest inventory. Nothing here is guessed or fabricated — if a
file's real logic was never shared with Claude in conversation, it is
explicitly marked as a placeholder rather than invented.

## ✅ Complete and working (carried over unchanged)

| File | Status |
|---|---|
| `sync/engine.py` | Unchanged from your original project |
| `templates/base.html` | Unchanged |
| `templates/index.html` | Unchanged (works standalone; not yet personalized for logged-in users) |
| `templates/roles.html` | Unchanged |
| `templates/skills.html` | Unchanged |
| `templates/roadmap_detail.html` | Unchanged — **note:** this file does not yet have progress checkboxes in the UI. The proposed stepper/checkbox/drawer redesign discussed earlier was never implemented in the version you shared, so it's carried over as-is. |
| `templates/admin.html` | Unchanged |
| `templates/404.html` / `500.html` | Unchanged |
| `diagnose.py` | Unchanged |
| `resume_parser.py` | Exactly as you provided it |

## 🔧 Modified (extended, not rewritten)

| File | What changed |
|---|---|
| `config.py` | Added `FIREBASE_SERVICE_ACCOUNT_PATH`, `GEMINI_API_KEY` |
| `models.py` | Added `users`, `user_roadmaps`, `user_topic_progress` tables + `User` dataclass |
| `db/operations.py` | Added user CRUD, roadmap-selection CRUD, per-user progress toggle/query functions |
| `api/routes.py` | `get_roadmap()` now injects per-user `completed` flags and `progress_pct` when logged in; added `/progress/<id>`, `/user/roadmaps` (GET/POST/DELETE) |
| `app.py` | Registers `auth_bp` and `onboarding_bp`; added `/login` and `/dashboard` routes |
| `requirements.txt` | Added `firebase-admin`. **Also added `beautifulsoup4` and `lxml`** — these were imported in your original `api/routes.py` (the scraping fallback) but were missing from the original requirements.txt, so that endpoint would have crashed on a clean install. |

## 🆕 New

| File | Purpose |
|---|---|
| `auth/__init__.py`, `auth/routes.py` | Firebase ID token verification, session login/logout, `login_required` decorator |
| `onboarding/__init__.py`, `onboarding/routes.py` | Resume upload → Gemini parse → skill review → role suggestion → role selection |
| `templates/login.html` | Google sign-in button (Firebase JS SDK) — needs your real `firebaseConfig` pasted in |
| `templates/onboarding.html` | 3-step onboarding UI wired to the endpoints above |
| `.env.example` | Updated with Firebase/Gemini vars |
| `.gitignore` | Protects `secrets/`, `.env`, DB files |
## 🤖 Role Classifier: CatBoost → Linear SVM Migration (Step 2.5)

**Status: SVM is production as of 2026-10-02.** CatBoost is retained,
loaded, and fully functional for rollback — it is simply no longer called
from onboarding's primary path.

### Why

An empirical 5-architecture comparison (CatBoost, Random Forest, Logistic
Regression, Linear SVM, XGBoost) under 5×3 repeated stratified CV found
CatBoost the weakest performer by a wide, statistically decisive margin.
Full validation chain: architecture comparison → hyperparameter tuning →
calibration validation (sigmoid vs. isotonic) → confusion-pattern parity →
TF-IDF tokenization ablation → per-class stability check.

| Metric | CatBoost (previous) | Linear SVM (current) |
|---|---|---|
| Macro F1 (5×3 repeated CV) | 0.745 ± 0.025 | 0.917 ± 0.009 |
| Per-class wins | — | 22/22 classes, incl. all 9 Tier-3 |
| `qa`→`data-engineer` confusion | 19.2% | 5.3% |
| Calibration ECE | n/a | 0.0675 (isotonic) |
| Per-role threshold pass rate | n/a | 22/22 |

Paired significance test (identical 15 fold-fits): t(14)=35.01,
p=4.93e-15; Wilcoxon W=0, p=6.53e-04; Cohen's d=9.04.

### New files

| File | Purpose |
|---|---|
| `train_svm_classifier.py` | Canonical SVM training script — produces the bundled artifact below |
| `role_classifier/role_classifier_svm.joblib` | Bundled TF-IDF vectorizer + OneHotEncoder + SimpleImputer + CalibratedClassifierCV(LinearSVC), ACTIVE |
| `role_classifier/role_classifier_svm_meta.json` | SVM metadata (`model_type: "linear_svm"`) |
| `ml/svm_inference.py` | Serving module — `predict_svm`/`predict_role_for_user`, `predict_svm_rankings`/`predict_role_rankings`, shadow-logging utilities |
| `check_class_stability_svm.py`, `svm_diagnose_confusion_pair.py` | SVM-specific diagnostic tooling forks |
| `shadow_log_analysis.py`, `generate_shadow_data_from_holdout.py`, `diagnose_shadow_logging.py`, `cleanup_diagnostic_shadow_rows.py` | Shadow-mode validation tooling |

### Modified files

| File | Change |
|---|---|
| `onboarding/routes.py` | `suggested_roles()` now imports `ml.svm_inference` as `role_clf` (production); `ml.role_classifier_inference` kept as `role_clf_catboost` for rollback only |
| `requirements.txt` | Added `scikit-learn==1.9.0`, `joblib==1.5.3`, `pandas==3.0.5`, `numpy==2.5.1`, `scipy==1.18.0`, `catboost==1.2.10` (previously missing entirely — a fresh install would have failed) |

### Known limitation, stated plainly

Real-traffic shadow-mode validation is thin — this project has no live user
base to generate organic onboarding traffic. The cutover decision rests
primarily on the offline repeated-CV evidence above, supplemented by a
batch comparison against genuinely held-out (never-trained-on) rows from
`candidate_job_role_dataset.csv` via `generate_shadow_data_from_holdout.py`,
rather than organic production traffic. If real users are ever onboarded,
re-enabling shadow logging (comparing SVM against the retained CatBoost
model) is recommended as a continued sanity check — see
`ml/svm_inference.py`'s `log_shadow_prediction()`.

### Rollback

If a regression is found: revert `onboarding/routes.py`'s import line to
`import ml.role_classifier_inference as role_clf` — no other change is
required, since CatBoost's model files and serving code were never
removed.

## ⚠️ Placeholder — you must replace before running

| File | Why |
|---|---|
| `github_client/parser.py` | Referenced by `sync/engine.py` and `diagnose.py`, but its actual implementation was **never shared with Claude** in this conversation — only the project file list mentioned its existence. The stub raises `NotImplementedError` on any call so the app fails loudly and specifically rather than silently. Copy your real file over this one. |
| `github_client/api.py` | Same situation — only used by REMOTE sync mode. If you only ever use LOCAL mode (`ROADMAP_REPO_PATH` set), you can leave this stub in place indefinitely and it will never be called. |

## 📋 Known gaps in what's built so far (Step 1 scope, not bugs)

- `roadmap_detail.html` doesn't call `/api/v2/progress/<id>` yet — the backend endpoint exists, but the frontend checkbox UI to trigger it hasn't been built. This is the natural next task once Step 1's plumbing is confirmed working.
- `index.html` doesn't yet branch on whether a user is logged in (e.g., showing "Continue your roadmap" vs. the generic landing page). `app.py`'s `/dashboard` route passes `user` and `user_roadmaps` into the template, but `index.html` itself doesn't use those variables yet.
- `onboarding/routes.py`'s `suggested-roles` matching is a naive word-overlap heuristic, explicitly flagged in its own docstring — it's meant to be replaced once a shared skill-taxonomy matcher exists (needed later for opportunity matching too).
- No account deletion / data export flow yet (relevant once you have real user data).

## 🗓 Planned for later phases (not part of this zip at all)

| Component | Phase |
|---|---|
| `opportunity-finder/` merge as a blueprint (hackathons/internships/jobs/events scrapers) | Phase 2 |
| Shared skill taxonomy (canonical skill names + aliases, used by both onboarding role-matching and opportunity matching) | Phase 2 |
| Rule-based opportunity ↔ skill matching ("Known"/"Missing" skills, % match) | Phase 2 |
| Profile / dashboard management page (edit roles, view completed skills, update resume) | Phase 2–3 |
| CatBoost re-ranking model (requires real usage/interaction data first) | Phase 3, after usage data exists |
| COSMOS assistant (text-based tool-calling agent first, voice later) | Phase 3 |
| Migration off SQLite to Postgres, if/when concurrent multi-user load requires it | Whenever contention becomes a real problem |
