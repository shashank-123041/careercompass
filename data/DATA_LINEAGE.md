# Dataset lineage

| File | Where | Notes |
|---|---|---|
| `raw/UpdatedResumeDataSet.csv` | gitignored | Public resume dataset (Tier 1). Check licence before redistributing. |
| `raw/candidate_job_role_dataset.csv` | gitignored | Public structured-skills dataset (Tier 2). |
| `interim/final_dataset_v3.csv` | gitignored | Tier 1 (Gemini-extracted) + Tier 2 + `tier3_generated.csv` (2,429 rows). |
| `interim/tier3_expanded_v2.csv` | gitignored | Web-grounded synthetic Tier-3 expansion (`scripts/data_building/tier3_expand.py`). |
| `interim/fullstack_expanded_v1.csv` | gitignored | Full-stack augmentation (`augment_fullstack.py`). |
| `interim/thin_roles_job_postings_v1*.csv` | gitignored | Scraped thin-role postings (`scrape_thin_role_data.py`); scraped content not redistributed. |
| `interim/final_dataset_v4.csv`, `v5.csv` | gitignored | Intermediate merges (v4 = + Tier-3 v2, degree_field normalised via `normalize_degree_field.py`; v5 = + full-stack batch). |
| `final/final_dataset_v6.csv` | committed | v5 + scraped thin-role rows (2,883). |
| `final/final_dataset_v7.csv` | committed | **Production.** v6 minus 22 scraped full-stack rows (`inspect_and_clean.py`) → 2,861 rows, 22 classes. |

The v4/v5 descriptions are inferred from script docstrings and row counts —
please confirm and correct before publishing.
