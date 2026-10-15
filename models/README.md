# models/

Committed: `role_classifier_svm_meta.json`, `role_classifier_meta.json`.

NOT committed (see .gitignore): `role_classifier_svm.joblib` (PRODUCTION) and
`role_classifier.cbm` (rollback). A joblib file is tied to the exact
scikit-learn version, so regenerate it in your own environment:

    python -m training.train_svm        # -> models/role_classifier_svm.joblib (+ meta)
    python -m training.train_catboost   # -> models/role_classifier.cbm (+ meta), optional

If you already have trained artifacts from the old `role_classifier/` folder,
copy them into this folder.
