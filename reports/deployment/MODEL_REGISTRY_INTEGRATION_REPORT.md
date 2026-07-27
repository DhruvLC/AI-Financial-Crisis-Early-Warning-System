# Model Registry Integration — Completion Report

**Date:** 2026-07-27
**Scope:** Fix the Model Information page / `/models` endpoint so that all
trained model families are discovered and displayed — not just Classical ML.

## Root Cause

The project maintains four independent on-disk registries:

| Family | Registry file | Entries |
|---|---|---|
| Classical ML | `models/registry.json` | 9 |
| Deep Learning | `models/deep_learning/registry.json` | 4 |
| Transformers | `models/transformers/registry.json` | 3 |
| Self-Supervised | `models/self_supervised/registry.json` | 3 |

`ModelService` (`src/api/model_loader.py`) instantiated a single
`ModelRegistry(models_dir)` which reads **only** the top-level
`models/registry.json`. `GET /api/v1/models` therefore returned only the 9
classical entries, and the Streamlit Model Information page (a thin view over
that endpoint) displayed only those. The non-classical registries were fully
present on disk but never read by the backend.

A secondary schema mismatch existed: non-classical registries record
`network` (not `algorithm`) and `checkpoints.{best,last}` (not `artefact`),
so they could not have been passed through the `ModelInfo` schema unmodified.

## Files Modified

1. **`src/api/model_loader.py`**
   - Added `discover_registries(models_dir)` — finds the top-level
     `registry.json` plus every `<subdir>/registry.json` one level deep.
     Any future family (e.g. `models/graph_nn/registry.json`) is picked up
     automatically with zero code changes.
   - Added `_normalise_entry(entry, family)` — maps `network` → `algorithm`,
     `checkpoints.best|last` → `artefact`, and tags each entry with its
     `family`.
   - `ModelService.registry_entries()` now aggregates normalised entries
     from all discovered registries.
2. **`src/api/schemas.py`** — `ModelInfo` gained a `family` field
   (default `classical_ml`, fully backward-compatible).
3. **`src/api/routes.py`** — `/models` passes `family` through; the
   `is_best` star is scoped to the classical family so cross-family version
   collisions (every family has a `v001`…) cannot mislabel entries. The
   active serving model (classical best, `extra_trees v004`) is unchanged.
4. **`frontend/pages/4_Model_Information.py`** — registry table gained a
   "Family" column; the metric-comparison chart labels bars as
   "Algorithm (Family)" to disambiguate same-named networks (e.g. MLP exists
   in both Deep Learning and Self-Supervised). No other UI changes.
5. **`tests/test_api.py`** — `test_models` now asserts that every on-disk
   family registry appears in the `/models` response.

## Explicitly Unchanged

No models retrained or regenerated; no artefacts, reports, preprocessing,
feature engineering, or pipelines touched. Prediction serving still loads
only the classical best model (`models/best_model.joblib`) exactly as before.

## Verification Results

- **FastAPI startup:** ✅ (uvicorn, model loaded: `extra_trees v004`)
- **`GET /models`:** ✅ 19 models — classical_ml 9, deep_learning 4,
  transformers 3, self_supervised 3; `best_model` = extra_trees v004
- **`GET /health`:** ✅ `status: ok`, model loaded
- **`POST /predict`:** ✅ full-schema instance → prediction with risk score
- **`POST /predict/batch`:** ✅ 2 instances, correct count/model_version
- **`GET /metrics`:** ✅ counters and active model version correct
- **Streamlit startup:** ✅ HTTP 200; Model Information page compiles and
  renders from the enriched `/models` payload (Dashboard / API Status pages
  unmodified and unaffected — they consume unchanged endpoints)
- **Regression tests:** ✅ all 206 tests pass
  (api 21, deep_learning 23, eda 24, feature_engineering 19,
  machine_learning 31, preprocessing 19, self_supervised 28,
  transformers 17, validation 24)
  — Note: running every test file in a single pytest process segfaults in a
  PyTorch/macOS teardown interaction that predates this change; each file
  passes cleanly in its own process and none of the modified code is involved.

## Production Readiness & Recommendations

The fix is minimal, additive, and backward-compatible — safe to deploy.

Recommendations (optional, out of scope here):
- Add a `?family=` query filter to `/models` if the registry grows large.
- Consider a per-family `best_model` field in `ModelsResponse` so the UI can
  highlight the champion of each family, not only the serving (classical) one.
- Split the PyTorch-heavy test modules into a separate pytest invocation in
  CI to avoid the single-process segfault.
