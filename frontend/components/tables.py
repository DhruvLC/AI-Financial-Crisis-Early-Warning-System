"""Table rendering helpers — styled prediction tables + summary stats."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def predictions_table(df: pd.DataFrame, max_rows: int | None = None) -> None:
    """Render batch prediction results with progress-styled columns."""
    view = df.head(max_rows) if max_rows else df
    st.dataframe(
        view, use_container_width=True, hide_index=True,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "Probability", format="%.4f", min_value=0.0, max_value=1.0),
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", format="%.1f", min_value=0.0, max_value=100.0),
            "confidence_score": st.column_config.NumberColumn(
                "Confidence", format="%.3f"),
            "prediction": st.column_config.NumberColumn(
                "Prediction", help="1 = distress flagged"),
        })
    if max_rows and len(df) > max_rows:
        st.caption(f"Showing first {max_rows} of {len(df)} rows — "
                   "download the CSV for the full results.")


def batch_summary_stats(df: pd.DataFrame) -> None:
    """Summary metric row for a batch of predictions."""
    n = len(df)
    flagged = int(df["prediction"].sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Companies scored", f"{n:,}")
    c2.metric("Flagged (distress)", f"{flagged:,}",
              delta=f"{flagged / n * 100:.1f}% of batch",
              delta_color="inverse")
    c3.metric("Mean probability", f"{df['probability'].mean():.4f}")
    c4.metric("Max risk score", f"{df['risk_score'].max():.1f}")
    c5.metric("High-risk count",
              f"{int((df['risk_level'] == 'High').sum()):,}")
