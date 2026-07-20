# API Reference — Financial Crisis EWS Backend

Base URL: `http://<host>:8000` — all business endpoints are versioned under `/api/v1`.
Interactive docs: **Swagger UI** at `/docs`, **ReDoc** at `/redoc`, OpenAPI spec at `/openapi.json`.

Every response carries `X-Request-ID` and `X-Response-Time-Ms` headers.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service index (links to docs/health) |
| GET | `/health` (also `/api/v1/health`) | Liveness/readiness — model load state, uptime |
| GET | `/api/v1/version` | API, app, model & dataset versions |
| GET | `/api/v1/models` | All registered models + best model, with test metrics |
| GET | `/api/v1/metrics` | Request/prediction counters, avg latency, uptime |
| POST | `/api/v1/predict` | Single prediction |
| POST | `/api/v1/predict/batch` | Batch predictions (max `EWS_MAX_BATCH_SIZE`, default 1000) |
| POST | `/api/v1/validate` | Validate payload against model feature schema (no inference) |

## POST /api/v1/predict

Request — the `features` object must contain **exactly** the 22 engineered
features the best model was trained on (see `GET /api/v1/models` or
`models/best_model.json` → `feature_schema.features`):

```json
{
  "id": "company-001",
  "features": {
    "ROA(B) before interest and depreciation after tax": 0.42,
    "Debt ratio %": 0.15,
    "...": 0.0
  }
}
```

Response:

```json
{
  "id": "company-001",
  "prediction": 0,
  "probability": 0.014044,
  "risk_score": 1.4,
  "risk_level": "Low",
  "confidence_score": 0.9262,
  "threshold": 0.190388,
  "model_version": "v004",
  "algorithm": "extra_trees",
  "prediction_timestamp": "2026-07-20T07:46:28.919426+00:00"
}
```

- **prediction** — 1 if `probability >= threshold` (Youden-optimal threshold from training).
- **risk_score** — probability × 100; **risk_level** — Low (<33) / Medium (<66) / High.
- **confidence_score** — normalized distance of the probability from the decision threshold (0–1).

## POST /api/v1/predict/batch

```json
{"instances": [{"id": "a", "features": {...}}, {"id": "b", "features": {...}}]}
```

Returns `{"predictions": [...], "count": 2, "model_version": "v004", "inference_ms": 27.5}`.

## POST /api/v1/validate

Same body as batch predict. Returns structured issues without running inference:

```json
{
  "valid": false,
  "n_instances": 1,
  "errors": [{"field": "instances[0].Debt ratio %", "issue": "missing required feature"}],
  "warnings": [{"field": "instances[0].foo", "issue": "unknown feature (ignored)"}]
}
```

Checks: missing features, unknown features, non-numeric values, NaN/Inf, empty payloads.

## Error envelope

All errors return:

```json
{
  "error": {
    "type": "invalid_request | model_not_loaded | model_loading_failure | registry_error | missing_artifact | configuration_error | prediction_failure | http_error | internal_error",
    "message": "human-readable message",
    "detail": null,
    "request_id": "78a097aeab0a",
    "path": "/api/v1/predict"
  }
}
```

Status codes: 422 invalid request/validation, 503 model not loaded, 500 server-side failures.

## Example (curl)

```bash
# Health
curl -s localhost:8000/api/v1/health | jq

# Predict (features file must contain all 22 feature names)
curl -s -X POST localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' -d @payload.json | jq
```
