# Installation — Financial Crisis EWS

## Prerequisites

- Python 3.9+ (3.11 recommended; the Docker image uses 3.11)
- (optional) Docker / Docker Compose for containerized serving
- Trained model artefacts present in `models/` (produced by the ML phase —
  `models/best_model.joblib` + `models/best_model.json` + `models/registry.json`)
  and the feature store in `data/features/` (produced by the feature-engineering phase)

## Full project (training + serving)

```bash
git clone <repo>
cd financial-crisis-ews
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full pipeline deps
pip install fastapi "uvicorn[standard]" httpx   # API deps
```

## API-only (serving)

```bash
pip install -r requirements-api.txt
```

## Verify

```bash
.venv/bin/python -m pytest tests/test_api.py -v
.venv/bin/python src/run_api.py    # then open http://localhost:8000/docs
```

## Environment variables

All optional; prefix `EWS_`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `EWS_HOST` | `0.0.0.0` | Bind host |
| `EWS_PORT` | `8000` | Bind port |
| `EWS_LOG_LEVEL` | `INFO` | Logging level |
| `EWS_LOG_FILE` | *(console only)* | Optional log file path |
| `EWS_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `EWS_PROJECT_ROOT` | repo root | Root for configs/models/data |
| `EWS_CONFIG_PATH` | `configs/config.yaml` | Central YAML config |
| `EWS_MODELS_DIR` | `models` | Model registry directory |
| `EWS_FEATURE_STORE_ROOT` | `data/features` | Feature store root |
| `EWS_RISK_SCORE_SCALE` | `100` (from config.yaml) | Probability → risk scale |
| `EWS_MAX_BATCH_SIZE` | `1000` | Max instances per batch request |
