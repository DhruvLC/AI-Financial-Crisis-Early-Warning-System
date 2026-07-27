"""Frontend configuration — backend URL, page metadata, theming constants.

The backend URL is resolved from (in order):
1. ``EWS_API_URL`` environment variable
2. ``EWS_API_URL`` in Streamlit secrets (if present)
3. Local development default ``http://localhost:8000``
"""
from __future__ import annotations

import os

DEFAULT_API_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

APP_TITLE = "Financial Crisis Early Warning System"
APP_ICON = "🏦"
APP_TAGLINE = "AI-powered corporate financial-distress risk prediction"

REQUEST_TIMEOUT = 30  # seconds
BATCH_PREVIEW_ROWS = 100

# Client-side upload guards (backend enforces MAX_BATCH_SIZE=1000 too)
MAX_BATCH_ROWS = 1000
MAX_CSV_MB = 10

RISK_COLORS = {"Low": "#2e9e5b", "Medium": "#e6a817", "High": "#d64545"}

# Path used only as a local fallback to discover the model's feature schema
# when the API's /validate probe is unavailable. Read-only.
LOCAL_BEST_MODEL_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "best_model.json")


def get_api_url() -> str:
    """Resolve the backend base URL (env var > secrets > local default)."""
    url = os.environ.get("EWS_API_URL")
    if not url:
        try:  # streamlit secrets are optional; absent file raises
            import streamlit as st
            url = st.secrets.get("EWS_API_URL")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - secrets file may not exist
            url = None
    return (url or DEFAULT_API_URL).rstrip("/")
