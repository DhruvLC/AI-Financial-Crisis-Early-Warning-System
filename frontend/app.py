"""Streamlit entry point — Financial Crisis Early Warning System frontend.

Run with:  streamlit run frontend/app.py
Backend URL is taken from the EWS_API_URL env var (default localhost:8000).
"""
from __future__ import annotations

import streamlit as st

from components.cards import nav_card
from components.sidebar import render_sidebar, setup_page
from components.theme import hero, section
from config import APP_ICON, APP_TAGLINE, APP_TITLE

setup_page("Home")
render_sidebar()

hero(APP_TITLE, APP_TAGLINE, APP_ICON)

st.markdown(
    "This frontend consumes the project's **FastAPI** REST API — all "
    "preprocessing, feature engineering, model selection, and inference run "
    "in the backend. The frontend never duplicates business logic."
)

section("Workspaces", "Everything you need, one click away.")
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    nav_card("pages/1_Dashboard.py", "Dashboard",
             "Backend status, active model, and service metrics at a glance.",
             "📊")
with r1c2:
    nav_card("pages/2_Single_Prediction.py", "Single Prediction",
             "Score one company's financial-distress risk interactively.",
             "🎯")
with r1c3:
    nav_card("pages/3_Batch_Prediction.py", "Batch Prediction",
             "Upload a CSV portfolio, score every company, download results.",
             "📁")

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    nav_card("pages/4_Model_Information.py", "Model Information",
             "Model registry, active-model metrics, and feature schema.",
             "🧠")
with r2c2:
    nav_card("pages/5_API_Status.py", "API Status",
             "Live health, version, endpoint probes, and latency.", "🩺")
with r2c3:
    nav_card("pages/6_About.py", "About",
             "Project background, architecture, and pipeline phases.", "ℹ️")
