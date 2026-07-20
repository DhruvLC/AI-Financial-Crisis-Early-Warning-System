# Backend Deployment (Phase 10A) — Audit Report

**Audit date:** 2026-07-20
**Audit type:** Read-only — no code modified, no fixes applied; this report is the only file created.
**Scope:** `src/api/`, `src/run_api.py`, `tests/test_api.py`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements-api.txt`, `.github/workflows/ci.yml`, `docs/`, README, and integration with all previous phases.

---

## 1. Executive summary

The Backend Deployment module was independently re-verified end-to-end during this audit: unit tests re-run (21/21 pass, plus 185/185 prior-phase tests — 206 total, 0 failures), a live uvicorn server exercised across every endpoint including error paths, prediction parity checked against the raw registry artefact (probabilities match to 6 decimal places), and the Docker container rebuilt image re-run to a `healthy` healthcheck with a successful in-container prediction.

The module does what the phase required: it serves the best registered model (`extra_trees` v004, threshold 0.190388, 22 features, feature-store v001) behind a versioned, documented, monitored FastAPI service without retraining anything and without modifying any previous phase (confirmed via `git status`: only new files + a README section).

**Verdict: PASS WITH RECOMMENDATIONS** — production readiness **9/10**. All functional requirements verified; the recommendations below (auth, per-process metrics under multi-worker uvicorn, image size) are hardening items, not defects.

---

## 2. Architecture review

```
client ──► FastAPI app factory (src/api/app.py)
             ├─ RequestContextMiddleware → request-id, access log, latency → Monitor
             ├─ CORSMiddleware (EWS_CORS_ORIGINS)
             ├─ Global exception handlers → one JSON error envelope
             ├─ /api/v1 routers (health, version, models, metrics, predict, validate)
             └─ lifespan startup ──► ModelService.load()
                     ├─ ModelRegistry (pipeline.ml.registry)  ← REUSED, unmodified
                     ├─ best_model.json descriptor (threshold, schema, versions)
                     └─ FeatureStore metadata (pipeline.feature_engineering.store) ← REUSED
```

Assessment: clean separation of concerns — one file per responsibility (config, schemas, loading, inference, middleware, exceptions, monitoring, routes, health). App-factory pattern (`create_app()`) makes the app testable and configurable. Dependency injection is used for the model service, inference pipeline, and monitor. Startup failure degrades gracefully (service stays up, `/health` reports `degraded`, prediction endpoints return 503) rather than crash-looping. Verified degraded behavior via test `TestErrorHandlingWithoutModel` (passes).

One architectural observation (pre-existing, not introduced by this phase): importing `pipeline.ml.registry` triggers `pipeline/ml/__init__.py`, which eagerly imports the visualization module (matplotlib/seaborn). This forces plotting libraries into the serving image. Noted in Recommendations.

## 3. Verification checklist

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Repository structure | ✅ | 12 files in `src/api/` + entry point + tests + deploy assets, all present |
| 2 | FastAPI app architecture | ✅ | Factory, lifespan, DI, middleware, versioned routers |
| 3 | Model Registry integration | ✅ | `load_best()` loads v004; registry `best_model` block matches descriptor exactly |
| 4 | Data Validation integration | ✅ | Same structured error-report style; request-level schema validation in `InferencePipeline` |
| 5 | Preprocessing integration | ✅ | Serves the engineered-feature contract (post-preprocessing schema); artefacts untouched |
| 6 | Feature Engineering integration | ✅ | FeatureStore metadata loaded; model's 22 features ⊆ store v001's 22 features (exact match) |
| 7 | Machine Learning integration | ✅ | Direct `joblib` inference vs API inference: probability 0.006822 = 0.006822 (parity) |
| 8 | Deep Learning integration | ✅ | Artefacts + registry intact (4 entries); tests pass; not served (best model is traditional ML — by design) |
| 9 | Transformer integration | ✅ | Artefacts + registry intact (3 entries); tests pass; not served (same rationale) |
| 10 | Self-Supervised integration | ✅ | Artefacts + registry intact (3 entries); tests pass; not served (same rationale) |
| 11 | Inference pipeline | ✅ | Validate → ordered matrix → predict; feature-order independence verified with shuffled input |
| 12 | Request validation | ✅ | Missing/unknown/non-numeric/NaN/null all rejected 422 with per-field detail (22 errors for empty schema) |
| 13 | Error handling | ✅ | 8 typed errors; 422/503/500/404 all return the standard envelope with request_id |
| 14 | Logging | ✅ | Reuses `ingestion.logging_config`; access lines carry rid, method, path, status, latency |
| 15 | Monitoring | ✅ | `/metrics` counters accurate in live test (13 requests, 6 predictions, avg latencies) |
| 16 | Docker configuration | ✅ | Image builds; container runs as `appuser`; HEALTHCHECK → `healthy`; in-container predict OK |
| 17 | GitHub Actions CI/CD | ✅ | `ci.yml` valid YAML; deps → flake8 → pytest → build check → docker build |
| 18 | Documentation | ✅ | README section + `docs/API.md`, `DEPLOYMENT.md`, `INSTALLATION.md` consistent with observed behavior |
| 19 | Unit tests | ✅ | 21/21 pass; prior phases 185/185 pass (backward compatible) |
| 20 | End-to-end functionality | ✅ | Full live sweep of all 8 endpoints + error paths (section 5) |

## 4. Model audit

Verified against artefacts on disk (not documentation):

- **Best model loads from registry**: `models/registry.json → best_model` = `{extra_trees, v004, models/best_model.joblib}`; `ModelService` loads exactly this artefact.
- **No retraining**: no `fit`/`train` calls anywhere in `src/api/`; artefact mtimes (2026-07-16, from the ML phase) unchanged after all audit runs.
- **Feature metadata**: descriptor lists 22 features; feature store v001 metadata lists the same 22; API enforces exact names and rebuilds column order regardless of input dict order (verified: shuffled input → identical probability).
- **Threshold**: API returns 0.190388 = descriptor's Youden threshold (asserted in live test).
- **Model version**: v004 surfaced consistently in `/health`, `/version`, `/models`, and every prediction.
- **Registry compatibility**: `/models` lists all 9 registered models with correct best-flag.

## 5. API assessment (live endpoint sweep)

Live server on :8125; every row observed directly during this audit:

| Endpoint | Status | Observed behavior |
|----------|--------|-------------------|
| `GET /` | 200 | Service index with docs/health links |
| `GET /health` + `/api/v1/health` | 200 | `ok`, model v004, feature-store v001, uptime |
| `GET /api/v1/version` | 200 | api v1, app 1.0.0, model v004, dataset v001 |
| `GET /api/v1/models` | 200 | 9 models, best flagged correctly |
| `GET /api/v1/metrics` | 200 | Counters matched actual request/prediction counts |
| `POST /api/v1/predict` (valid) | 200 | Full payload incl. risk_score 0.7/Low, confidence 0.9642, timestamp |
| `POST /api/v1/predict` (unknown feature) | 422 | `invalid_request`, 22 per-field errors in detail |
| `POST /api/v1/predict` (malformed body) | 422 | `invalid_request` (Pydantic layer) |
| `POST /api/v1/predict` (null value) | 422 | `invalid_request` |
| `POST /api/v1/predict/batch` (5) | 200 | count 5, inference_ms reported |
| `POST /api/v1/predict/batch` (empty) | 422 | `invalid_request` (min_length=1) |
| `POST /api/v1/validate` | 200 | Structured errors (22 missing) + warnings (1 unknown) |
| `GET /docs`, `/redoc`, `/openapi.json` | 200 | OpenAPI lists all 10 paths, correct title/version |
| `GET /nope` | 404 | Standard envelope, type `http_error` |

Headers `X-Request-ID` and `X-Response-Time-Ms` present on responses; access log recorded all 18 requests with request IDs.

Schema quality: Pydantic v2 models with descriptions and an example payload; response models declared on every route, so the OpenAPI contract matches actual responses.

## 6. Deployment assessment

- **Dockerfile**: `python:3.11-slim`, layer-cached deps, non-root `appuser` (verified via `docker inspect`: `user=appuser`), `HEALTHCHECK` on `/health`, 2 uvicorn workers. Copies only `src/`, `configs/`, `models/`, `data/features/`.
- **Container run**: reached `healthy` (checked twice, 15 s apart); in-container prediction returned the same probability (0.006822) as host inference — artefact parity inside the image.
- **docker-compose.yml**: valid; env vars, restart policy, healthcheck mirror the Dockerfile.
- **.dockerignore**: excludes `.venv`, tests, reports, raw/interim/processed data, and the heavy DL/transformer/SSL model dirs — correct, since the API serves only the sklearn best model.
- **Environment configuration**: 11 `EWS_*` variables, all with defaults, YAML fallback for shared values; documented in INSTALLATION.md.
- **Image size**: 2.32 GB — functional but heavy (sklearn + xgboost + matplotlib/seaborn; see Recommendations).

## 7. CI/CD assessment

`.github/workflows/ci.yml` (valid YAML): checkout → Python 3.11 + pip cache → install `requirements-api.txt` + httpx/flake8/pytest → flake8 error-class lint → `pytest tests/test_api.py` → app-factory import/build verification → dependent job building the Docker image. No cloud deploy or registry push (matches phase scope). One caveat: the CI pytest job depends on `models/` and `data/features/` artefacts being committed to the repo — true today (they exist in the tree), but worth keeping in mind if artefacts are ever moved to external storage. Pre-existing `pylint.yml` / `python-publish.yml` workflows were left untouched.

Local equivalents of every CI step were executed in this audit and pass (flake8 clean, 0 issues even at `--max-line-length=100`; tests green; image builds).

## 8. Documentation assessment

| Document | Assessment |
|----------|------------|
| `README.md` | Deployment section added; stage table updated; quickstart, endpoints, Docker, env vars, CI, tech stack — consistent with observed behavior |
| `docs/API.md` | All 8 endpoints, request/response examples matching actual payloads (spot-checked field-for-field), error envelope, curl examples |
| `docs/DEPLOYMENT.md` | Architecture diagram, local/Docker run, production notes (workers, CORS, k8s probes), model-rollout procedure |
| `docs/INSTALLATION.md` | Prereqs, full vs API-only install, verification steps, complete env-var table |

No inconsistencies found between documentation and verified behavior. Minor gap: README's roadmap section still lists "Deployment + monitoring (stage 16)" as future work while the stage table above it correctly marks it ✅ — cosmetic only.

## 9. Test assessment

- **API tests** (`tests/test_api.py`): **21 passed, 0 failed** — model loading (3: success, missing artifact, not-loaded guard), routes/docs (2), health/version/models/metrics/headers (5), predict single/batch + 3 error paths (5), validate good/bad (2), degraded-service 503 (1), inference-pipeline units (2), monitor counters (1). Covers every category the phase required (routes, health, predict, batch, validate, model loading, error handling).
- **Regression**: all prior-phase suites re-run — **185 passed, 0 failed** (validation, preprocessing, EDA, feature engineering, ML, deep learning, transformers, self-supervised). Backward compatibility confirmed.
- **Coverage**: not measurable — `pytest-cov` is not installed in the project venv (consistent with prior phases, which also don't use it). By inspection, all 12 API modules are exercised by the suite; untested paths are narrow (e.g., registry-corruption branches in `model_loader.py`).

## 10. Risks

1. **No authentication/authorization** — anyone who can reach the port can predict and enumerate models. Acceptable inside a private network; a blocker for public exposure. (Documented as deferred in the completion report.)
2. **Per-process metrics under multi-worker uvicorn** — the Docker CMD runs 2 workers, but `Monitor` counters live in process memory, so `/metrics` reflects only the worker that answers the call and counts split across workers. Numbers are correct in the single-worker/local case and directionally useful in Docker, but not aggregate-accurate.
3. **CORS default `*`** — dev-friendly default; must be overridden in production (documented, but a default-unsafe posture).
4. **Serving image weight (2.32 GB)** — driven by the eager `pipeline.ml.__init__` import chain pulling matplotlib/seaborn into a service that never plots.
5. **No rate limiting / request size caps beyond batch limit** — large hostile payloads are bounded only by `EWS_MAX_BATCH_SIZE` (1000) and server defaults.
6. **Deprecation warnings** — Starlette warns `HTTP_422_UNPROCESSABLE_ENTITY` is deprecated (rename to `..._CONTENT` in a future starlette will be needed); harmless today.

## 11. Recommendations (not implemented — audit only)

1. Add an auth layer (API key header or gateway-level) before any non-private deployment.
2. For accurate aggregate metrics with multiple workers, either run 1 worker per container and scale by replicas (documented pattern), or export Prometheus metrics with multiprocess mode.
3. Change `pipeline/ml/__init__.py` to lazily import visualization/plotting (or have the API import `pipeline.ml.registry` submodule directly without package side effects) — would cut matplotlib/seaborn from the image and shrink it substantially. (Requires a small prior-phase change; deliberately not done in the deployment phase.)
4. Pin exact dependency versions in `requirements-api.txt` (currently `>=` ranges) for reproducible images.
5. Add `pytest-cov` to CI for a coverage gate.
6. Set a non-`*` CORS default and require explicit opt-in to wildcard.
7. Cosmetic: update the README roadmap line that still lists deployment as future work.

## 12. Production readiness score

| Dimension | Score /10 | Basis |
|-----------|-----------|-------|
| Reliability | 9 | Graceful degraded mode, healthchecks, restart policy; verified live + in container |
| Maintainability | 10 | Small single-purpose modules, docstrings, consistent with project conventions |
| Security | 7 | Non-root container, input validation, no secrets in code; no auth, wildcard CORS default |
| Scalability | 8 | Stateless service, worker/replica scaling documented; metrics not aggregate under multi-worker |
| Modularity | 10 | Factory + DI + typed errors; zero coupling changes to prior phases |
| Documentation | 9 | Four consistent docs verified against behavior; one cosmetic README staleness |
| Deployment readiness | 9 | Image builds and runs healthy; heavy image, no push pipeline (out of scope) |
| API quality | 9 | Versioned, OpenAPI-accurate, structured errors, request IDs; no rate limiting |
| Code quality | 9 | flake8 clean at 100-col; minor deprecation warnings pending upstream renames |
| Test quality | 9 | 21 targeted tests, all requirement categories covered; no coverage tooling |

**Overall: 9 / 10**

## 13. Final verdict

**PASS WITH RECOMMENDATIONS**

Every audit objective was independently re-verified: correct registry integration with no retraining, exact prediction parity with the registered artefact, full endpoint contract compliance including error paths, working logging and monitoring, a healthy production container, valid CI, consistent documentation, and 206/206 tests passing with all previous phases intact. The recommendations are hardening measures for public/production exposure, not corrections of defects.
