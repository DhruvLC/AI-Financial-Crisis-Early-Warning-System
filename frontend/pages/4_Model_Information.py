"""Model Information — registry, active model, metrics, feature schema."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import model_metrics_bar
from components.sidebar import render_sidebar, setup_page
from components.theme import hero, section
from services.api_client import cached_models, get_feature_schema

setup_page("Model Information", "🧠")
render_sidebar()

hero("Model Information",
     "Enterprise model registry, active-model metrics, and feature schema.",
     "🧠")

models = cached_models(st.session_state.api_url)
if not models:
    st.error("Could not load model registry from the backend. "
             "Check the Backend URL in the sidebar.")
    st.stop()

best = models.get("best_model")

# ── active model ──────────────────────────────────────────────────────────────
section("Active (best) model", "The currently deployed model serving predictions.")
if best:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", best["algorithm"].replace("_", " ").title())
    c2.metric("Version", best["model_version"])
    c3.metric("Trained", (best.get("training_timestamp") or "—")[:10])
    c4.metric("Registered models", models.get("count", "—"))

    if best.get("metrics"):
        st.plotly_chart(model_metrics_bar(best["metrics"],
                                          "Best model — test metrics"),
                        use_container_width=True)
else:
    st.info("No best model is registered.")

# ── registry table ────────────────────────────────────────────────────────────
section("Model registry", "All registered models across every family.")
entries = models.get("models", [])
if entries:
    rows = []
    for e in entries:
        m = e.get("metrics", {}) or {}
        rows.append({
            "Best": "⭐" if e.get("is_best") else "",
            "Family": (e.get("family") or "classical_ml")
                      .replace("_", " ").title(),
            "Version": e["model_version"],
            "Algorithm": e["algorithm"].replace("_", " ").title(),
            "ROC AUC": m.get("roc_auc"),
            "PR AUC": m.get("pr_auc"),
            "F1": m.get("f1"),
            "Recall": m.get("recall"),
            "Precision": m.get("precision"),
            "Trained": (e.get("training_timestamp") or "")[:10],
        })
    reg_df = pd.DataFrame(rows)
    st.dataframe(
        reg_df, use_container_width=True, hide_index=True,
        column_config={
            k: st.column_config.NumberColumn(k, format="%.4f")
            for k in ("ROC AUC", "PR AUC", "F1", "Recall", "Precision")
        })

    # metric comparison across models
    metric_cols = ["ROC AUC", "PR AUC", "F1", "Recall", "Precision"]
    plot_df = reg_df.dropna(subset=metric_cols, how="all").copy()
    plot_df["Model"] = plot_df["Algorithm"] + " (" + plot_df["Family"] + ")"
    if len(plot_df) > 1:
        with st.expander("Compare a metric across models"):
            metric = st.selectbox("Metric", metric_cols, index=0)
            import plotly.express as px
            fig = px.bar(plot_df.sort_values(metric, ascending=True),
                         x=metric, y="Model", orientation="h",
                         color_discrete_sequence=["#2a78d6"],
                         text=plot_df.sort_values(metric)[metric]
                              .map(lambda v: f"{v:.3f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(template="plotly_white", showlegend=False,
                              xaxis_range=[0, 1.12], yaxis_title=None,
                              margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("The model registry is empty.")

# ── feature schema ────────────────────────────────────────────────────────────
section("Feature schema", "The engineered inputs the active model consumes.")
features = get_feature_schema(st.session_state.api_url)
if features:
    st.caption(f"The active model consumes **{len(features)} engineered "
               "features** (produced by the project's feature-engineering "
               "pipeline).")
    schema_df = pd.DataFrame({"#": range(1, len(features) + 1),
                              "Feature": features})
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
else:
    st.info("Feature schema unavailable.")
