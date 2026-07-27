"""About — project background, architecture, and pipeline phases."""
from __future__ import annotations

import streamlit as st

from components.sidebar import render_sidebar, setup_page
from config import APP_TAGLINE, APP_TITLE

setup_page("About", "ℹ️")
render_sidebar()

st.title(f"ℹ️ About — {APP_TITLE}")
st.caption(APP_TAGLINE)

st.markdown(
    """
### What is this?

An end-to-end **AI Financial Crisis Early Warning System** that predicts
corporate financial-distress risk from engineered financial ratios. The
system was built in phases — data ingestion through deployment — and this
Streamlit app is the user-facing frontend over the FastAPI inference
service.

### Architecture

```
┌────────────────────┐     HTTP (REST)      ┌─────────────────────────┐
│ Streamlit frontend │ ───────────────────▶ │ FastAPI backend         │
│  · Dashboard       │  /api/v1/predict     │  · Inference pipeline   │
│  · Predictions     │  /api/v1/predict/    │  · Model registry       │
│  · Model info      │        batch         │  · Validation           │
│  · API status      │  /api/v1/models …    │  · Monitoring           │
└────────────────────┘                      └───────────┬─────────────┘
                                                        │ loads
                                            ┌───────────▼─────────────┐
                                            │ Trained model artefacts │
                                            │ (classical ML, DL,      │
                                            │  transformer, SSL)      │
                                            └─────────────────────────┘
```

All preprocessing, feature engineering, model selection, and inference run
in the **backend** — the frontend never duplicates business logic.

### Pipeline phases

| Phase | What it delivered |
|---|---|
| Data ingestion | FRED, World Bank, SEC EDGAR, Yahoo Finance, Kaggle, … |
| Data validation | Schema, missing-value, outlier, and time-series checks |
| Preprocessing & EDA | Cleaning, imputation, exploratory analysis |
| Feature engineering | 22 engineered financial-distress features |
| Classical ML | 11 algorithms benchmarked; best model registered |
| Deep learning | MLP / sequence models with training pipeline |
| Transformers | Attention-based tabular/sequence models |
| Self-supervised | Contrastive representation pretraining |
| FastAPI deployment | REST API with health, prediction, and monitoring |
| **Streamlit frontend** | **This app** |

### Model output semantics

- **prediction** — `1` = distress flagged, `0` = healthy
- **probability** — model-predicted probability of distress
- **risk_score** — probability scaled to 0–100
- **risk_level** — Low / Medium / High banding
- **threshold** — the tuned decision threshold applied to probability

### Disclaimer

This system is a research/educational tool. Predictions are **not**
financial advice and must not be the sole basis for credit, investment,
or supervisory decisions.
"""
)
