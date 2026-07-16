# Machine Learning Module — Verification Audit

**Date:** 2026-07-16
**Environment:** Python 3.9.6 · scikit-learn 1.6.1 · XGBoost 2.1.4 · SHAP 0.49.1
**Scope:** Resume of interrupted verification stage. No source code was modified — all checks passed as-is.

---

## 1. End-to-End Pipeline Run

Command: `python src/run_machine_learning.py --config configs/config.yaml`

- **Exit code:** 0 ✅
- **Dataset:** feature store `v001` — 22 features, target `Bankrupt?`
  - train (4773, 23) — 154 positives
  - val (1023, 23) — 33 positives
  - test (1023, 23) — 33 positives
- **Models trained:** 9 / 9 enabled (0 failed). `lightgbm` and `catboost` gracefully skipped (optional back-ends not installed, per config `fail_fast: false`).
- **Outputs:** 32 reports, 67 figures, 9 registry entries + best-model artifact.

## 2. Unit Tests

Command: `python -m pytest tests/test_machine_learning.py`

- **Result:** **31 passed, 0 failed** (8.06 s) ✅
- Coverage spans: model fit/predict for all algorithms, error paths (unsupported model, empty dataset, missing features, predict-before-fit), metrics suite, threshold optimization (youden / max_f1 / custom / disabled / invalid), CV splitters, random & grid search, disabled-tuning no-op, cross-validation, data-loader verification (empty store, missing target, schema mismatch, NaN, hash tamper detection), registry (register/load, best-model, version increment, missing artifact), permutation & native importance, full pipeline run, failing-model isolation, all-models-fail error.

## 3. Component Verification

| Component | Check | Result |
|---|---|---|
| Data loading | `v001` splits loaded, schema/hash verified; tamper detection unit-tested | ✅ |
| Model training | 9 algorithms trained, hyperparameters + train time logged per model | ✅ |
| Hyperparameter tuning | Disabled in default config (by design). Live smoke test with `tuning.enabled=true` (random search, decision_tree, temp output dir) produced `best_params={min_samples_leaf: 2, max_depth: 4}`, `best_score=0.8711`; grid/random/no-op paths also unit-tested | ✅ |
| Cross-validation | 5-fold stratified CV per model (`roc_auc`); e.g. best model cv_mean=0.9239 ± 0.0204 | ✅ |
| Model evaluation | Full metric suite (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, balanced acc., MCC, Cohen's κ, log-loss, Brier) on val + test; Youden threshold optimization applied | ✅ |
| Model comparison | `leaderboard.csv/.md` ranked by test ROC-AUC with tie-breaks; 4 comparison figures | ✅ |
| Explainability | SHAP report (500 samples/model), native + permutation feature importance CSVs and figures for all models | ✅ |
| Model registry | `models/registry.json` — 9 entries, versioned v001–v009; best model registered separately | ✅ |
| Model persistence | `best_model.joblib` reloaded in fresh process; `predict_proba` on held-out test reproduced ROC-AUC **0.9395** exactly matching the pipeline report | ✅ |
| Report generation | 32 reports written: training, evaluation, feature-importance, SHAP, best-model reports, 9 model cards, leaderboard, metrics summary, `ml_report.json` | ✅ |

## 4. Artifact Inventory

**models/** (10 artifacts + metadata)
- `logistic_regression_v001` … `mlp_v009` (.joblib, 9 models)
- `best_model.joblib`, `best_model.json` (extra_trees v004, threshold 0.1904 [youden])
- `registry.json` (9 entries)

**reports/ml/** (32 files)
- `leaderboard.{csv,md}`, `metrics_summary.csv`, `ml_report.json`
- `training_report.md`, `evaluation_report.md`, `best_model_report.md`
- `feature_importance_report.md` + 14 per-model importance CSVs (native + permutation)
- `shap_report.md`
- 9 model cards (`model_card_<algo>.md`)

**reports/ml/figures/** (67 PNGs)
- Per model: ROC, PR, confusion, calibration, lift, gain, importance
- Comparisons: `comparison_{roc_auc,f1,recall,pr_auc}.png`

## 5. Issues Found

**None.** No verification failure occurred; no source code was modified. (The only environment action taken was installing `pytest` into the project venv, which was missing.)

## 6. Verdict

**The Machine Learning module PASSES end-to-end verification.**
