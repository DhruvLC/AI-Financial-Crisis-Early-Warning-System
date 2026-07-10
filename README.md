# AI Financial Crisis Early Warning System

An end-to-end ML pipeline that predicts corporate financial distress / bankruptcy
and outputs a **0–100 risk score**. This scaffold implements the "traditional ML"
path of the full diagram; deep-learning, transformer, and self-supervised stages
are stubbed for phase 2.

## Pipeline stages (mapped to the diagram)

| Stage | Module | Status |
|-------|--------|--------|
| 2  Data Collection      | `src/pipeline/data_collection.py` | ✅ Kaggle auto-download |
| 3  Data Preparation     | `src/pipeline/data_prep.py`       | ✅ clean, outliers, split |
| 6  Feature Engineering  | `src/pipeline/features.py`        | ✅ variance filter, scaling |
| 7  Traditional ML       | `src/pipeline/models.py`          | ✅ RandomForest, XGBoost |
| 8–12 Deep / Transformer / SSL | `models.py` (stubs)         | 🔜 phase 2 |
| 13 Explainable AI       | `src/pipeline/explain.py`         | ✅ SHAP |
| Output  Risk Score      | `src/pipeline/risk_score.py`      | ✅ 0–100 + Low/Med/High |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Kaggle credentials (for auto-download)

1. kaggle.com → Account → **Create New API Token** → downloads `kaggle.json`
2. ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

> No Kaggle account? Drop any CSV with a `Bankrupt?` label column into
> `data/raw/` — the downloader detects it and skips the API call. Point
> `data.target_col` in the config at your label column.

## Run

```bash
cd financial-crisis-ews
python src/run_pipeline.py --config configs/config.yaml
```

## Outputs

- `reports/model_comparison.csv` — ROC-AUC / PR-AUC / P / R / F1 per model
- `reports/shap_<model>.png` + `reports/feature_importance_<model>.csv`
- `reports/risk_scores.csv` — per-company 0–100 risk score + level
- `models/best_model.pkl` — winning model + fitted transformers

## Data ingestion module

Production ingestion for 10 sources under `src/ingestion/`. Every source is a
`BaseIngestor` subclass: `fetch → validate → store raw+interim → metadata`,
with per-source failure isolation.

```bash
python src/run_ingestion.py --list                 # show sources
python src/run_ingestion.py --config configs/ingestion.yaml
python src/run_ingestion.py --only kaggle_bankruptcy fred world_bank
```

**Works with zero credentials** (enabled by default): `fred` (CSV fallback),
`yahoo_finance`, `world_bank`, `imf`, `oecd`, `sec_edgar` (set a real
`user_agent` email in the config).

**Need credentials** (disabled by default — flip `enabled: true`):

| Source | Requirement |
|--------|-------------|
| `kaggle_bankruptcy`, `kaggle_news`, `kaggle_stock` | `~/.kaggle/kaggle.json` (chmod 600) |
| `alpha_vantage` | `export ALPHAVANTAGE_API_KEY=...` |
| `fred` (faster API path) | `export FRED_API_KEY=...` (optional) |

**Outputs**
- `data/raw/<source>/…` — untouched fetch (audit + self-supervised reuse)
- `data/interim/<source>.parquet` — validated, cleaned
- `data/raw/_metadata/<source>.meta.json` — per-dataset metadata + checksum
- `data/raw/_metadata/run_manifest.json` — status of every source in the run
- `logs/ingestion.log` — full ingestion log

**Config:** `configs/ingestion.yaml` (tickers, CIKs, FRED series, countries, etc.)

## Roadmap (phase 2)

- Enrich stage 2 with **FRED** (macro), **yfinance** (market), news sentiment
- Implement stages 8–11 (Deep MLP, LSTM/GRU, TabTransformer, self-supervised)
- Deployment + monitoring (stage 16): FastAPI service, drift checks, retraining
