"""Batch Prediction — upload a CSV, score via /predict/batch, download."""
from __future__ import annotations

import streamlit as st

from components.cards import error_box
from components.charts import (probability_histogram, risk_counts_bar,
                               risk_distribution_donut, risk_score_strip)
from components.sidebar import render_sidebar, setup_page
from components.tables import batch_summary_stats, predictions_table
from components.uploader import csv_uploader
from config import BATCH_PREVIEW_ROWS
from components.theme import hero, section
from services.api_client import APIError, get_client, get_feature_schema
from services.utils import df_to_csv_bytes, df_to_instances, predictions_to_df

setup_page("Batch Prediction", "📁")
render_sidebar()

hero("Batch Prediction",
     "Upload a CSV portfolio and score every company at once via "
     "`/predict/batch`.", "📁")

features = get_feature_schema(st.session_state.api_url)
if not features:
    st.error("Could not load the model feature schema — the backend is "
             "unreachable and no local model descriptor was found. Check "
             "the Backend URL in the sidebar and refresh.")
    st.stop()

df, valid = csv_uploader(features)

if valid and df is not None:
    if st.button(f"🔮 Score {len(df):,} companies", type="primary",
                 use_container_width=True):
        try:
            with st.spinner(f"Scoring {len(df):,} companies…"):
                response = get_client().predict_batch(
                    df_to_instances(df, features))
        except APIError as exc:
            error_box(str(exc), detail=exc.detail)
            st.stop()
        st.session_state.batch_result = response

if "batch_result" in st.session_state:
    response = st.session_state.batch_result
    results = predictions_to_df(response["predictions"])

    st.divider()
    section("Results")
    st.caption(f"Model **{response['model_version']}** · "
               f"{response['count']:,} predictions in "
               f"{response['inference_ms']:.1f} ms")

    batch_summary_stats(results)

    st.download_button("⬇ Download results as CSV",
                       data=df_to_csv_bytes(results),
                       file_name="ews_batch_predictions.csv",
                       mime="text/csv", type="primary")

    predictions_table(results, max_rows=BATCH_PREVIEW_ROWS)

    section("Visual summary")
    threshold = float(results["threshold"].iloc[0]) \
        if "threshold" in results else None
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(risk_distribution_donut(results),
                        use_container_width=True)
        st.plotly_chart(probability_histogram(results, threshold),
                        use_container_width=True)
    with c2:
        st.plotly_chart(risk_counts_bar(results), use_container_width=True)
        st.plotly_chart(risk_score_strip(results), use_container_width=True)
