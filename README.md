# 🧭 CareerCompass

AI-powered career guidance for IT students: developer-roadmap explorer,
Firebase-authenticated accounts, resume-based onboarding with an SVM role
classifier, and a scraped-opportunity recommendation engine.

## Quick start

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then fill in your values
```

1. **Firebase**: create a project, enable Google + Email/Password sign-in, put the
   service-account JSON at `secrets/firebase-service-account.json`
   (the web `firebaseConfig` lives in `templates/login.html` and `profile.html`).
2. **Roadmap data**: clone https://github.com/nilbuild/developer-roadmap and set
   `ROADMAP_REPO_PATH` in `.env` (or leave empty to use the GitHub API fallback).
3. **Role classifier**: the trained model is not committed — build it once:
   ```bash
   pip install -r requirements-ml.txt      # only needed for training/experiments
   python -m training.train_svm            # -> models/role_classifier_svm.joblib
   ```
   Without it the app still runs; onboarding falls back to the word-overlap heuristic.
4. Run: `python run.py` → http://localhost:5050

## Layout

```
careercompass/            application package
  app.py  config.py  models.py
  db/ api/ auth/ onboarding/ user_profile/   web layer + persistence
  taxonomy/ github_client/ sync/ scrapers/    data ingestion + skill vocabulary
  ranking/                heuristic opportunity scorer
  resume/                 Gemini resume parser (+ optional dictionary fallback)
  ml/                     common.py (shared), svm_inference.py (production),
                          catboost_inference.py (rollback), opportunity bootstrap
  templates/  static/
training/                 train_svm.py (production), train_catboost.py (rollback)
scripts/                  data_building/  experiments/  shadow/  diagnose.py
data/final/               committed training datasets (v7 = production)
data/{raw,interim}/       gitignored; see data/DATA_LINEAGE.md
models/                   metadata JSON committed; binaries gitignored
results/                  experiment CSVs backing the paper's numbers
docs/                     PROJECT_STATUS.md, MIGRATION_NOTES.md
prototypes/cosmos/        COSMOS landing-page prototype (Phase 3)
```

All filesystem locations are defined once in `careercompass/config.py`.
Scripts are run as modules from the repo root, e.g.
`python -m scripts.experiments.svm_hyperparameter_search`.

See `docs/PROJECT_STATUS.md` for project history and `docs/MIGRATION_NOTES.md`
for what changed in this restructure.
