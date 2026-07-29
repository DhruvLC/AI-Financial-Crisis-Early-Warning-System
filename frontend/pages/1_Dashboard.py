"""Dashboard — project overview, backend status, model + service metrics."""
from __future__ import annotations

import streamlit as st

from components.cards import nav_card, status_badge
from components.charts import model_metrics_bar
from components.sidebar import render_sidebar, setup_page
from components.theme import hero, section
from services.api_client import (APIError, cached_health, cached_models,
                                 get_client, get_feature_schema)

setup_page("Dashboard", "📊")
render_sidebar()

hero("Dashboard",
     "AI Financial Crisis Early Warning System — live system overview",
     "📊")

base_url = st.session_state.api_url
health = cached_health(base_url)
online = bool(health and health.get("status") == "ok")

# ── status row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown("**Backend**  \n" + status_badge(online))
c2.markdown("**Model loaded**  \n"
            + (status_badge(bool(health and health.get("model_loaded")),
                            "Yes", "No") if health else "—"))
c3.metric("Active model", health.get("model_version", "—") if health else "—")
c4.metric("Latency", f"{health.get('_latency_ms', 0):.0f} ms"
          if health else "—")

if not online:
    st.error("Backend is offline. Start it with "
             "`uvicorn api.app:create_app --factory --app-dir src` "
             "or adjust the Backend URL in the sidebar.")
    st.stop()

# ── service metrics ───────────────────────────────────────────────────────────
section("Service metrics", "Live counters from the backend `/metrics` endpoint.")
try:
    metrics, _ = get_client().metrics()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total requests", f"{metrics['request_count']:,}")
    m2.metric("Total predictions", f"{metrics['prediction_count']:,}")
    m3.metric("Errors", f"{metrics['error_count']:,}")
    m4.metric("Avg latency", f"{metrics['avg_latency_ms']:.1f} ms")
    m5.metric("Uptime", f"{metrics['uptime_seconds'] / 3600:.1f} h")
except APIError as exc:
    st.warning(f"Service metrics unavailable: {exc}")

# ── active model card ─────────────────────────────────────────────────────────
section("Active model", "The deployed best model and its headline test metrics.")
models = cached_models(base_url)
best = (models or {}).get("best_model")
if best:
    left, right = st.columns([1, 2])
    with left:
        st.metric("Algorithm", best["algorithm"].replace("_", " ").title())
        st.metric("Version", best["model_version"])
        if best.get("training_timestamp"):
            st.metric("Trained", best["training_timestamp"][:10])
        st.metric("Registered models", (models or {}).get("count", "—"))
        st.metric("Model features", len(get_feature_schema(base_url)))
    with right:
        if best.get("metrics"):
            st.plotly_chart(model_metrics_bar(best["metrics"],
                                              "Best model — test metrics"),
                            use_container_width=True)
else:
    st.info("Model registry information unavailable.")

# ── quick navigation ──────────────────────────────────────────────────────────
section("Quick actions", "Jump straight into the most common workflows.")
n1, n2, n3 = st.columns(3)
with n1:
    nav_card("pages/2_Single_Prediction.py", "Single Prediction",
             "Score one company from manually entered financial ratios.",
             "🎯")
with n2:
    nav_card("pages/3_Batch_Prediction.py", "Batch Prediction",
             "Upload a CSV and score an entire portfolio at once.", "📁")
with n3:
    nav_card("pages/4_Model_Information.py", "Model Information",
             "Inspect the model registry, metrics, and feature schema.", "🧠")
