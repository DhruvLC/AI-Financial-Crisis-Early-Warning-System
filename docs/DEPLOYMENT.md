# Deployment — Financial Crisis EWS Backend

The Backend Deployment module serves the best registered model (currently
`extra_trees` v004, ROC-AUC ≈ 0.94) from the existing Model Registry behind a
production FastAPI service. No retraining happens at serve time.

## Architecture

```
client ──► FastAPI (src/api)
             │  RequestContextMiddleware  → request-id, access log, latency metrics
             │  CORS middleware           → configurable origins
             │  Global exception handlers → standardized JSON errors
             ├─ /api/v1/predict[.batch] ──► InferencePipeline
             │        │ schema validation (names, order, types, NaN/Inf)
             │        └ ModelService ──► ModelRegistry (models/registry.json,
             │                            best_model.joblib, best_model.json)
             │                       └─► FeatureStore metadata (data/features/)
             └─ /health /metrics /models /version ──► Monitor + registry
```

Startup (lifespan): load best model + descriptor (threshold, feature schema,
version) + feature-store metadata. Failures leave the service in `degraded`
state — `/health` still answers, prediction endpoints return 503.

## Local

```bash
.venv/bin/python src/run_api.py
# or
.venv/bin/python -m uvicorn api.app:app --app-dir src --port 8000 --reload
```

## Docker

```bash
docker build -t financial-crisis-ews-api .
docker run -p 8000:8000 financial-crisis-ews-api
# or
docker compose up --build
```

The image:
- `python:3.11-slim`, non-root user, `HEALTHCHECK` on `/health`
- installs only serving deps (`requirements-api.txt` — no torch/shap/kaggle)
- copies `src/`, `configs/`, `models/`, `data/features/` (training data and
  heavy DL artefacts are excluded via `.dockerignore`)
- runs uvicorn with 2 workers

## Production notes

- Scale workers via the CMD (`--workers N`) or replicate containers behind a
  load balancer; the service is stateless (metrics are per-process).
- Set `EWS_CORS_ORIGINS` to your real origins (default `*` is dev-only).
- Ship logs by setting `EWS_LOG_FILE` or scraping stdout.
- Kubernetes: use `/health` for both liveness and readiness probes
  (readiness should additionally check `model_loaded: true`).

## CI/CD

`.github/workflows/ci.yml`:
1. **test job** — install deps, flake8 (error-class checks), pytest on
   `tests/test_api.py`, app-factory import/build verification.
2. **docker job** — builds the production image (verification only; no push,
   no cloud deploy).

## Rollout of a new model

Retrain via the existing ML phase (`src/run_machine_learning.py`); it updates
`models/registry.json` + `best_model.*`. Rebuild/restart the API — it picks up
the new best model at startup. No API code changes required.
