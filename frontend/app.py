"""Streamlit entry point — Financial Crisis Early Warning System frontend.

Run with:  streamlit run frontend/app.py
Backend URL is taken from the EWS_API_URL env var (default localhost:8000).
"""
from __future__ import annotations

import streamlit as st

from components.sidebar import render_sidebar, setup_page
from config import APP_ICON, APP_TAGLINE, APP_TITLE

setup_page("Home")
render_sidebar()

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(APP_TAGLINE)

st.markdown(
    """
Welcome. This frontend consumes the project's **FastAPI** REST API — all
preprocessing, feature engineering, model selection, and inference run in the
backend.

**Use the sidebar** to navigate:

| Page | What it does |
|---|---|
| 📊 Dashboard | Backend status, active model, service metrics at a glance |
| 🎯 Single Prediction | Score one company interactively |
| 📁 Batch Prediction | Upload a CSV, score many companies, download results |
| 🧠 Model Information | Registry, active model, metrics, feature schema |
| 🩺 API Status | Health, version, endpoints, latency |
| ℹ️ About | Project background and architecture |
"""
)

st.page_link("pages/1_Dashboard.py", label="Go to the Dashboard →",
             icon="📊")
