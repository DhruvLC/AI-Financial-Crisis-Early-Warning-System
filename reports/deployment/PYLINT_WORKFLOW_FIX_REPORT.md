# Pylint Workflow Fix Report

Date: 2026-07-29
Project: AI Financial Crisis Early Warning System
Scope: `.github/workflows/pylint.yml` + new `.pylintrc` only. CI, Docker,
and release workflows, all ML code, `models/`, `data/`, and `reports/`
content untouched.

## Root Cause

The original `pylint.yml`:

1. **Installed only pylint, never the project dependencies.** Every
   `import streamlit / fastapi / pandas / numpy / plotly / torch / requests`
   raised `E0401 import-error`, driving the score down and failing the job.
2. **Set no `PYTHONPATH`.** `src` and `frontend` are runtime roots
   (`--app-dir src`, `streamlit run frontend/app.py`), so intra-project
   imports (`from components...`, `from api...`) were also unresolvable.
3. **Linted everything** via `pylint $(git ls-files '*.py')` — including
   `notebooks/` and non-production scripts.
4. **Ran on Python 3.8/3.9/3.10**, but the pinned dependencies
   (`pandas>=2.0`, `torch>=2.1`, `fastapi>=0.110`) do not resolve cleanly on
   3.8, guaranteeing matrix failures.
5. **Zero-tolerance failure mode** — any single convention message failed
   the build.

## Files Modified

| File | Change |
|---|---|
| `.github/workflows/pylint.yml` | Rewritten (details below) |
| `.pylintrc` | **New** — enterprise defaults |
| `reports/deployment/PYLINT_WORKFLOW_FIX_REPORT.md` | This report |

No other files changed.

## Workflow Improvements

- Matrix reduced to **Python 3.10** (matches supported dependency floor).
- `actions/setup-python@v5` (was v3, deprecated) with **pip caching** keyed
  on all three requirements files.
- Explicit `PYTHONPATH=$GITHUB_WORKSPACE/src:$GITHUB_WORKSPACE/frontend` so
  both runtime roots are importable, mirrored inside `.pylintrc` via
  `init-hook` so local runs behave identically.
- Failure policy: `--fail-under=9.0 --fail-on=E` — any **error**-class
  finding (real import/name/member problems) fails the build immediately;
  style/refactor messages fail only if quality drops below 9.0/10. Pylint is
  NOT disabled globally.

## Dependency Installation

The install step iterates defensively — if a file is removed later the step
still works with whatever remains:

```bash
for req in requirements.txt requirements-api.txt requirements-frontend.txt; do
  if [ -f "$req" ]; then pip install -r "$req"; fi
done
pip install pylint
```

All three files exist today and are installed (training, API, and frontend
dependency sets), so streamlit, fastapi, pandas, numpy, plotly, torch,
requests, xgboost, shap, kaggle, etc. are importable before pylint runs.

## Lint Scope

- Linted: **`src/` and `frontend/`** — the only production-code roots
  (`tests/` is exercised by the CI workflow, not style-gated here).
- Excluded via `.pylintrc` `ignore=`: `notebooks`, `reports`, `data`,
  `models`, `logs`, `assets` — generated/derived artefacts, exactly the
  categories the requirements allow excluding.

## `.pylintrc` Design (sensible enterprise defaults)

- `py-version=3.10`, parallel jobs, 100-char lines.
- **Disabled only project-inappropriate rules**, each documented inline:
  docstring requirements for UI helper modules, DS/ML naming (`df`, `X`,
  `y`), ML-typical complexity limits (`too-many-locals/arguments`),
  intentional per-model boilerplate (`duplicate-code`).
- **Kept all error checks, unused-import, redefined-builtin,
  broad-exception-caught, raise-missing-from**, etc. — these still surface
  (verified: 60+ actionable warnings reported at 9.60/10).
- `ignored-modules=lightgbm,catboost,umap` — intentionally-optional
  back-ends guarded by `try/except` / lazy imports in
  `src/pipeline/ml/models.py` and `self_supervised/visualization.py`, not
  installed anywhere in CI. `generated-members` covers matplotlib colormap
  attributes (`plt.cm.tab10`) that pylint cannot introspect.

## Verification

Performed locally with pylint 3.3.9:

1. `python3 -c "yaml.safe_load(...)"` — workflow YAML parses ✓
2. Requirements-file detection loop tested — all three found ✓
3. Full run `PYTHONPATH=src:frontend pylint --rcfile=.pylintrc src frontend`:
   - Score **9.60/10** ✓ (gate: 9.0)
   - **0 non-import errors** ✓ — the only E-codes remaining locally are 32
     `E0401 import-error`s for torch/shap/kaggle/fastapi/streamlit/etc.,
     all of which ARE listed in the requirements files the workflow now
     installs, so they cannot occur on the GitHub runner.
4. Useful checks confirmed still active (unused-import, redefined-builtin,
   raise-missing-from all reported) ✓
5. `ci.yml`, `python-publish.yml`, `Dockerfile`, `docker-compose.yml`
   untouched ✓

## Expected GitHub Actions Result

On the next push, the **Pylint** workflow will:

1. Check out, set up Python **3.10** with cached pip.
2. Install all three requirements files + pylint (~3–5 min cold, seconds
   warm).
3. Run `pylint --rcfile=.pylintrc --fail-under=9.0 --fail-on=E src frontend`
   with `PYTHONPATH` covering both roots.
4. All previously-failing `E0401` import errors are resolved by the
   installed dependencies; measured score 9.60/10 clears the 9.0 gate.

**Expected status: ✅ PASS**, while still failing fast on any future real
error (undefined names, broken imports, invalid members) or a quality
regression below 9.0.
