# Financial Crisis EWS — Streamlit Frontend

Interactive web frontend for the **AI Financial Crisis Early Warning System**.
It is a thin client over the project's FastAPI backend — all preprocessing,
feature engineering, and inference run server-side.

## Structure

```
frontend/
├── app.py                     # Streamlit entry point (Home)
├── config.py                  # Backend URL resolution, theming constants
├── pages/
│   ├── 1_Dashboard.py         # Backend status, active model, service metrics
│   ├── 2_Single_Prediction.py # Score one company interactively
│   ├── 3_Batch_Prediction.py  # CSV upload → batch scoring → download + charts
│   ├── 4_Model_Information.py # Registry, metrics, feature schema
│   ├── 5_API_Status.py        # Health, version, endpoint probes
│   └── 6_About.py             # Project background & architecture
├── components/                # Reusable UI (sidebar, cards, charts, tables, uploader)
├── services/                  # API client + CSV/dataframe utilities
└── assets/                    # Static assets
```

## Installation

```bash
pip install -r requirements-frontend.txt
```

## Running

1. **Start the backend** (from the repo root):

   ```bash
   pip install -r requirements-api.txt
   uvicorn api.app:create_app --factory --app-dir src --port 8000
   ```

2. **Start the frontend** (from the repo root):

   ```bash
   streamlit run frontend/app.py
   ```

   Streamlit opens at <http://localhost:8501>.

## Backend configuration

The backend base URL is resolved in this order:

1. `EWS_API_URL` environment variable
2. `EWS_API_URL` in `.streamlit/secrets.toml`
3. Default `http://localhost:8000`

Switch to a deployed backend without code changes:

```bash
EWS_API_URL=https://my-ews-api.example.com streamlit run frontend/app.py
```

or edit the **Backend URL** field in the sidebar at runtime (per-session).

## Usage

- **Dashboard** — at-a-glance backend health, active model card, service
  metrics, and quick navigation.
- **Single Prediction** — enter the 22 engineered feature values and score
  one company; results show a risk banner, probability gauge, and full
  API response.
- **Batch Prediction** — download the CSV template, fill one row per
  company, upload, score, review interactive Plotly summaries, and
  download results as CSV.
- **Model Information** — model registry table, metric comparison chart,
  and the full feature schema.
- **API Status** — live health/version/metrics probes and an endpoint
  latency table.

## Screenshots

| Page | Screenshot |
|---|---|
| Dashboard | _placeholder — `assets/screenshot_dashboard.png`_ |
| Single Prediction | _placeholder — `assets/screenshot_single.png`_ |
| Batch Prediction | _placeholder — `assets/screenshot_batch.png`_ |
| Model Information | _placeholder — `assets/screenshot_models.png`_ |
| API Status | _placeholder — `assets/screenshot_status.png`_ |
