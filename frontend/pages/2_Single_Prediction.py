"""Single Prediction — enter feature values, score one company."""
from __future__ import annotations

import streamlit as st

from components.cards import error_box, risk_banner
from components.charts import gauge
from components.sidebar import render_sidebar, setup_page
from services.api_client import APIError, get_client, get_feature_schema

setup_page("Single Prediction", "🎯")
render_sidebar()

st.title("🎯 Single Prediction")
st.caption("Enter the engineered feature values for one company and score "
           "its financial-distress risk via the backend `/predict` endpoint.")

features = get_feature_schema(st.session_state.api_url)
if not features:
    st.error("Could not load the model feature schema — the backend is "
             "unreachable and no local model descriptor was found. Check "
             "the Backend URL in the sidebar and refresh.")
    st.stop()

st.info(f"The active model expects **{len(features)} engineered features**. "
        "Values default to 0.0 — adjust the ones you know.")

with st.form("single_prediction"):
    company_id = st.text_input("Company ID (optional)", value="company-001")
    cols = st.columns(3)
    values: dict[str, float] = {}
    for i, name in enumerate(features):
        with cols[i % 3]:
            values[name] = st.number_input(name, value=0.0, format="%.6f",
                                           key=f"feat_{i}")
    submitted = st.form_submit_button("🔮 Predict risk", type="primary",
                                      use_container_width=True)

if submitted:
    try:
        with st.spinner("Scoring…"):
            result = get_client().predict(values, id=company_id or None)
    except APIError as exc:
        error_box(str(exc), detail=exc.detail)
        st.stop()

    st.divider()
    st.subheader("Prediction result")
    risk_banner(result["risk_level"], result["probability"])

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(result["probability"], result["threshold"]),
                        use_container_width=True)
    with right:
        a, b = st.columns(2)
        a.metric("Prediction",
                 "⚠️ Distress" if result["prediction"] == 1 else "✅ Healthy")
        b.metric("Risk level", result["risk_level"])
        a.metric("Probability", f"{result['probability']:.4f}")
        b.metric("Risk score", f"{result['risk_score']:.1f} / 100")
        a.metric("Confidence", f"{result['confidence_score']:.3f}")
        b.metric("Threshold", f"{result['threshold']:.4f}")
        st.caption(f"Model **{result['model_version']}** "
                   f"({result['algorithm']}) · "
                   f"{result['prediction_timestamp']}")

    with st.expander("Raw API response"):
        st.json(result)
