# Machine Learning Module — Audit Report

**Date:** 2026-07-16 · **Type:** Read-only audit (no source files modified)
**Environment:** Python 3.9.6 · scikit-learn 1.6.1 · XGBoost 2.1.4 · SHAP 0.49.1 · LightGBM/CatBoost not installed
**Module:** `src/pipeline/ml/` (11 files, ~1,713 lines) + `src/run_machine_learning.py` + `tests/test_machine_learning.py` (31 tests)

---

## 1. Project Integration — ✅ PASS

| Aspect | Evidence | Status |
|---|---|---|
| Feature Engineering integration | `MLDataLoader` consumes `pipeline.feature_engineering.store.FeatureStore`; design mirrors `FeatureEngineeringPipeline` | ✅ |
| Feature Store loading | Loads versioned splits (`data/features/v001`); verifies splits present, target binary, schema consistency, feature alignment, no NaN/inf, content-hash integrity (`data_loader.py:70-134`) | ✅ |
| Configuration | Fully config-driven via `configs/config.yaml` `ml:` section (algorithms, tuning, CV, metrics, thresholds, explainability, model_params, output dirs) | ✅ |
| Logging | Shared `ingestion.logging_config.get_logger` throughout; namespaced loggers (`ml.pipeline`, `ml.data`, `ml.registry`, …) | ✅ |
| Model Registry | `models/registry.json` — 9 complete entries, all artefact paths exist | ✅ |
| Reports | 32 files under `reports/ml/`, consistent JSON+MD+CSV shape with other modules | ✅ |
| Metadata | Every registry entry records algorithm, hyperparameters, tuning, CV scores, threshold+method, per-split metrics, UTC timestamp, feature schema, dataset version | ✅ |
| Versioning | Models auto-versioned v001–v009; dataset version (`v001`) recorded per entry for lineage | ✅ |

## 2. Models — ✅ PASS (9 trained, 2 optional back-ends correctly gated)

All 11 algorithms implemented as small `BaseModel` subclasses in `models.py`, registered in `MODEL_REGISTRY`, built via a single `build_model` factory:

| Algorithm | Implemented | Trained | Notes |
|---|---|---|---|
| Logistic Regression | ✅ | ✅ v001 | `class_weight=balanced`, lbfgs |
| Decision Tree | ✅ | ✅ v002 | depth/leaf caps, balanced |
| Random Forest | ✅ | ✅ v003 | 400 trees, balanced |
| Extra Trees | ✅ | ✅ v004 | **best model** |
| SVM | ✅ | ✅ v006 | RBF, `probability=True` |
| KNN | ✅ | ✅ v007 | distance-weighted, k=15 |
| Naive Bayes | ✅ | ✅ v008 | GaussianNB |
| MLP | ✅ | ✅ v009 | (64,32), early stopping |
| XGBoost | ✅ | ✅ v005 | runtime `scale_pos_weight` from class ratio |
| LightGBM | ✅ | — | not installed; graceful skip via `available()` |
| CatBoost | ✅ | — | not installed; graceful skip via `available()` |

Imbalance handling is thoughtful and consistent (class weights everywhere applicable, XGBoost pos-weight computed at runtime, `pipeline.py:184-193`).

## 3. Hyperparameter Tuning — ✅ PASS

- **Grid Search**: `GridSearchCV`, `refit=False`, best params merged into wrapper (`tuning.py:90-92`) — unit-tested.
- **Random Search**: `RandomizedSearchCV` with `n_iter` budget, seeded — unit-tested + verified live this session (produced valid `best_params`/`best_score`).
- **Cross Validation**: `StratifiedKFold` (shuffled, seeded) and `TimeSeriesSplit` splitters; `cross_val_score` per model. Production run: 5-fold stratified ROC-AUC per model (best: 0.9239 ± 0.0204).
- Default per-model search spaces for all 11 algorithms, overridable via config. Disabled-tuning is a clean no-op. Tuning disabled in the default config (deliberate; documented in config comments).

## 4. Evaluation — ✅ PASS (11/11 metrics)

All required metrics implemented in `ModelEvaluator` (`evaluation.py:46-60`) via scikit-learn, computed at an optimized threshold: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, balanced accuracy, MCC, Cohen's κ, log loss, Brier score. Plus: confusion matrix per split, `zero_division=0` guards, NaN + note on single-class AUC failure, threshold optimization on **validation** split only (Youden / max-F1 with bounded 200-point grid / custom) — no test leakage. `metrics_summary.csv`: 18 rows (9 models × val+test), zero nulls.

## 5. Explainability — ✅ PASS

- **Native feature importance**: `feature_importances_` or |coef| (`base.py:156-171`) — 6 models produced CSVs.
- **Permutation importance**: model-agnostic, on validation split, 5 repeats, seeded — all 9 models.
- **SHAP**: TreeExplainer fast path with model-agnostic fallback; per-class/3-D value handling; 500-sample cap; graceful skip if uninstalled — `shap_report.md` covers all 9 models.
All three are best-effort with logged degradation — a failing explainer never kills a run.

## 6. Reports — ✅ PASS (32 files, all valid)

`ml_report.json` (complete run record), `training_report.md`, `evaluation_report.md`, `leaderboard.{csv,md}`, `metrics_summary.csv`, `best_model_report.md`, `feature_importance_report.md` + 14 per-model CSVs, `shap_report.md`, 9 model cards (with intended-use & limitations sections). Verified: no empty files, no null cells in CSVs, JSON parses with 9 models / 0 failures / best=extra_trees.

## 7. Visualizations — ✅ PASS (67 figures, full grid)

Verified complete 7-kind × 9-model grid with **zero gaps**: ROC, PR, confusion matrix, calibration, lift, gain, feature importance — plus 4 model-comparison bars (roc_auc, f1, recall, pr_auc). Headless Agg backend, best-effort plotting with logged skips, no zero-byte files.

## 8. Model Registry — ✅ PASS

- **Persistence**: joblib artefacts for all 9 models + `best_model.joblib`; reload verified in a fresh process this session — reproduced test ROC-AUC **0.9395** exactly.
- **Versioning**: monotonic v001–v009.
- **Metadata**: all entries contain the full required key set; all artefact paths exist.
- **Best model**: `extra_trees` v004 recorded in `registry.json` and `best_model.json` (threshold 0.190, Youden); `load_best()`/`load_model()` APIs unit-tested.

## 9. Unit Tests — ✅ PASS

`pytest tests/test_machine_learning.py` → **31 passed, 0 failed** (verified this session). Coverage includes happy paths *and* error paths: unsupported model, empty data, missing features at predict, predict-before-fit, hash tampering, schema mismatch, NaN detection, missing registry entries, failing-model isolation, all-models-fail.

## 10. End-to-End Pipeline — ✅ PASS

`python src/run_machine_learning.py --config configs/config.yaml` (verified this session): exit 0; 9/9 models trained; leaderboard produced; 67 figures, 32 reports, 10 model artefacts written. Best: extra_trees — test ROC-AUC 0.9395, recall 0.7576, PR-AUC 0.4380.

## 11. Code Quality

| Dimension | Assessment |
|---|---|
| Architecture | Excellent separation of concerns: loader / models / tuning / evaluation / explain / registry / viz / report / orchestrator, each a focused ≤241-line file. Deliberate structural parallel with the feature-engineering module. |
| OOP design | Clean template-method `BaseModel` (subclasses only define `name`, `default_params`, `_build`); dataclasses for results; single factory; registry pattern. |
| Logging | Uniform, namespaced, informative (per-model timings, metric summaries, skips). |
| Exception handling | Strong: dedicated `MLError`; per-model isolation honouring `fail_fast`; best-effort explainability/plotting; graceful optional-dependency degradation. |
| Type hints | Present on essentially all public signatures (modern `from __future__ import annotations` style). |
| Documentation | Module docstrings explain design intent; method docstrings throughout; config file is self-documenting with inline comments. |
| Configuration | Everything behaviour-affecting is configurable with sensible defaults; CLI supports version/algorithm overrides. |
| Maintainability | High — small files, uniform patterns, no duplication, adding an algorithm is a ~15-line subclass. |

Minor nits (no fix required): `report.py` methods take untyped `result`; `MLReport.written` accumulates across multiple `write()` calls on one instance; SVM training is slow (1,336 s — no config knob to subsample for SVC).

## 12. Production Readiness Scores

| Category | Score | Rationale |
|---|---|---|
| Architecture | **9/10** | Modular, layered, mirrors sibling modules; no serving layer yet |
| Scalability | **7/10** | `n_jobs=-1` parallelism; but single-node, in-memory, SVM/KNN scale poorly with data size |
| Maintainability | **9/10** | Small uniform files, registry+factory pattern, config-driven |
| Robustness | **9/10** | Extensive pre-flight data verification, per-model failure isolation, graceful degradation everywhere |
| Reproducibility | **9/10** | Seeded end-to-end, dataset version + content hashes + full hyperparameters recorded per model |
| Testing | **8/10** | 31 tests incl. error paths; no coverage measurement, no property-based/integration-CI evidence |
| Documentation | **8/10** | Strong docstrings + model cards; no module-level README/API docs |
| Deployment readiness | **6/10** | Persisted, reloadable, metadata-complete models — but no inference API, monitoring/drift hooks, or containerization |

**Production readiness: 8.1 / 10** (weighted average) · **Completion: 100%** of the specified ML-module scope (11/11 algorithms implemented; 9 trained — the 2 untrained are correctly gated optional back-ends).

## Strengths

1. Rigorous data contract before training: schema, target, NaN/inf, and hash-integrity verification against the feature store.
2. Fault isolation — one failing model never kills the run; optional back-ends degrade to logged skips.
3. Methodologically sound: threshold optimized on validation (not test), stratified CV, class-imbalance handling per algorithm, tie-broken leaderboard.
4. Complete, audited artefact trail: versioned registry with full lineage, 32 reports, 67 figures with a gap-free per-model grid.
5. Verified round-trip persistence (reload reproduces test ROC-AUC exactly).
6. Consistent house style with the rest of the project (logging, config, report shape).

## Weaknesses

1. Hyperparameter tuning is disabled in the default config — production models use hand-picked defaults (the tuning code itself is verified working).
2. Low precision at the chosen operating point (best model: 0.25 precision at 0.76 recall) — inherent to Youden on a ~3% positive class; the max-F1 or a cost-based threshold may suit deployment better.
3. No probability calibration step (calibration curves are plotted, but no `CalibratedClassifierCV` option).
4. SVM (1,336 s) and KNN (193 s) dominate runtime with bottom-half performance.
5. Stratified (random) CV on what may be temporally ordered financial data; `TimeSeriesSplit` exists but is not the default — potential look-ahead optimism.
6. Python 3.9 venv nearing EOL; joblib artefacts are not portable across sklearn versions (version pinning matters for deployment).

## Remaining Issues

- **Stale artifact**: `models/best_model.pkl` (Jul 11) comes from the older `src/run_pipeline.py` monolith, not the ML module; keeping both `best_model.pkl` and `best_model.joblib` risks confusion. (Not modified — read-only audit.)
- Nothing else: no failing tests, no missing artefacts, no invalid reports found.

## Recommendations

1. Enable tuning (`ml.tuning.enabled: true`) for a final model-selection run before any production use.
2. Choose the deployment threshold from business costs (FN vs FP) rather than Youden; consider adding probability calibration.
3. Switch CV default to `time_series` (or verify rows are not temporally ordered) to rule out leakage optimism.
4. Remove or archive the legacy `best_model.pkl`; document `best_model.joblib` as canonical.
5. Consider dropping SVM/KNN from the default algorithm list (cost ≫ benefit here) or gating them behind a "full sweep" flag.
6. Pin dependency versions in `requirements.txt` for artefact portability; plan a Python ≥3.11 migration.

## Verdict

**The Machine Learning module is COMPLETE (100%) and production-ready at 8.1/10 for its intended scope.**

✅ **READY TO PROCEED TO THE DEEP LEARNING PHASE.** The feature store contract, evaluation suite, registry, and reporting infrastructure this module establishes can be reused directly by the DL module; none of the noted weaknesses block the next phase.
