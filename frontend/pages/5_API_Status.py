"""API Status — health, version, endpoints, live latency probe."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import status_badge
from components.sidebar import render_sidebar, setup_page
from components.theme import hero, section
from services.api_client import APIError, cached_health, get_client

setup_page("API Status", "🩺")
render_sidebar()

hero("API Status",
     "Live health, version, and endpoint diagnostics for the FastAPI "
     "backend.", "🩺")

base_url = st.session_state.api_url
client = get_client()

if st.button("↻ Re-run all probes"):
    cached_health.clear()
    st.rerun()

# ── health ────────────────────────────────────────────────────────────────────
section("Health", "Core liveness check from `/health`.")
health = cached_health(base_url)
online = bool(health and health.get("status") == "ok")

c1, c2, c3, c4 = st.columns(4)
c1.markdown("**Backend**  \n" + status_badge(online))
c2.markdown("**Model loaded**  \n"
            + (status_badge(bool(health and health.get("model_loaded")),
                            "Yes", "No") if health else "—"))
c3.metric("Latency", f"{health.get('_latency_ms', 0):.0f} ms"
          if health else "—")
c4.metric("Uptime", f"{(health.get('uptime_seconds') or 0) / 3600:.1f} h"
          if health else "—")

if not health:
    st.error(f"Backend unreachable at `{base_url}`. Start it with "
             "`uvicorn api.app:create_app --factory --app-dir src` "
             "or adjust the Backend URL in the sidebar.")
    st.stop()

with st.expander("Raw /health response"):
    st.json({k: v for k, v in health.items() if not k.startswith("_")})

# ── version ───────────────────────────────────────────────────────────────────
section("Version", "Build and model versions from `/version`.")
try:
    version, v_ms = client.version()
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("API version", version.get("api_version", "—"))
    v2.metric("App version", version.get("app_version", "—"))
    v3.metric("Model version", version.get("model_version") or "—")
    v4.metric("Probe latency", f"{v_ms:.0f} ms")
    with st.expander("Raw /version response"):
        st.json(version)
except APIError as exc:
    st.warning(f"/version unavailable: {exc}")

# ── service metrics ───────────────────────────────────────────────────────────
section("Service metrics", "Live counters from `/metrics`.")
try:
    metrics, _ = client.metrics()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Requests", f"{metrics['request_count']:,}")
    m2.metric("Predictions", f"{metrics['prediction_count']:,}")
    m3.metric("Errors", f"{metrics['error_count']:,}")
    m4.metric("Avg latency", f"{metrics['avg_latency_ms']:.1f} ms")
    m5.metric("Uptime", f"{metrics['uptime_seconds'] / 3600:.1f} h")
    with st.expander("Raw /metrics response"):
        st.json(metrics)
except APIError as exc:
    st.warning(f"/metrics unavailable: {exc}")

# ── endpoint probe table ──────────────────────────────────────────────────────
section("Endpoint probes", "Round-trip latency for every read endpoint.")
probes = [
    ("GET", "/health", lambda: client.health()),
    ("GET", "/version", lambda: client.version()),
    ("GET", "/models", lambda: client.models()),
    ("GET", "/metrics", lambda: client.metrics()),
]
rows = []
for method, path, call in probes:
    try:
        _, ms = call()
        rows.append({"Method": method, "Endpoint": path,
                     "Status": "🟢 OK", "Latency (ms)": round(ms, 1)})
    except APIError as exc:
        rows.append({"Method": method, "Endpoint": path,
                     "Status": f"🔴 {exc.status or 'ERR'}",
                     "Latency (ms)": None})
rows.append({"Method": "POST", "Endpoint": "/predict",
             "Status": "— (exercised on the Single Prediction page)",
             "Latency (ms)": None})
rows.append({"Method": "POST", "Endpoint": "/predict/batch",
             "Status": "— (exercised on the Batch Prediction page)",
             "Latency (ms)": None})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption(f"Backend base URL: `{base_url}` · API prefix `/api/v1` · "
           f"interactive docs at [{base_url}/docs]({base_url}/docs)")
