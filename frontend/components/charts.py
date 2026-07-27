"""Plotly chart builders (interactive, theme-safe).

Colors follow the validated reference palette: risk levels use the reserved
status palette (good/warning/critical) — always paired with text labels, never
color alone — and magnitude (histogram) uses the single sequential blue hue.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Reserved status palette (validated) — risk is a state, not a series.
RISK_STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
SEQ_BLUE = "#2a78d6"
GRID = "#e1e0d9"

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=20, t=48, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
)


def _base(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def risk_distribution_donut(df: pd.DataFrame) -> go.Figure:
    """Donut of risk-level counts, labeled directly (never color alone)."""
    counts = (df["risk_level"].value_counts()
              .reindex(["Low", "Medium", "High"]).dropna().reset_index())
    counts.columns = ["risk_level", "count"]
    fig = px.pie(counts, names="risk_level", values="count", hole=0.55,
                 color="risk_level", color_discrete_map=RISK_STATUS)
    fig.update_traces(textinfo="label+percent",
                      hovertemplate="%{label}: %{value} companies "
                                    "(%{percent})<extra></extra>")
    return _base(fig, "Risk level distribution")


def risk_counts_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of risk-category counts with direct value labels."""
    counts = (df["risk_level"].value_counts()
              .reindex(["Low", "Medium", "High"], fill_value=0).reset_index())
    counts.columns = ["risk_level", "count"]
    fig = px.bar(counts, x="risk_level", y="count", color="risk_level",
                 color_discrete_map=RISK_STATUS, text="count")
    fig.update_traces(textposition="outside", marker_line_width=0,
                      hovertemplate="%{x}: %{y} companies<extra></extra>")
    fig.update_layout(showlegend=False, xaxis_title=None,
                      yaxis_title="Companies")
    return _base(fig, "Risk category counts")


def probability_histogram(df: pd.DataFrame,
                          threshold: float | None = None) -> go.Figure:
    """Histogram of predicted distress probabilities (single sequential hue)."""
    fig = px.histogram(df, x="probability", nbins=30,
                       color_discrete_sequence=[SEQ_BLUE])
    fig.update_traces(marker_line_width=1, marker_line_color="#fcfcfb",
                      hovertemplate="p ∈ %{x}: %{y} companies<extra></extra>")
    fig.update_layout(xaxis_title="Predicted probability of distress",
                      yaxis_title="Companies", bargap=0.02)
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="#d03b3b",
                      annotation_text=f"threshold {threshold:.3f}",
                      annotation_position="top right")
    return _base(fig, "Probability distribution")


def risk_score_strip(df: pd.DataFrame) -> go.Figure:
    """Per-company risk scores (0-100), colored by risk level with labels."""
    fig = px.strip(df, x="risk_score", y="risk_level", color="risk_level",
                   color_discrete_map=RISK_STATUS,
                   hover_data=["id", "probability"],
                   category_orders={"risk_level": ["High", "Medium", "Low"]})
    fig.update_layout(showlegend=False, xaxis_title="Risk score (0–100)",
                      yaxis_title=None, xaxis_range=[-2, 102])
    return _base(fig, "Risk scores by category")


def gauge(probability: float, threshold: float) -> go.Figure:
    """Single-prediction probability gauge with the decision threshold."""
    pct = probability * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", font=dict(size=40)),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%"),
            bar=dict(color=SEQ_BLUE, thickness=0.35),
            steps=[dict(range=[0, 33], color="rgba(12,163,12,0.13)"),
                   dict(range=[33, 66], color="rgba(250,178,25,0.13)"),
                   dict(range=[66, 100], color="rgba(208,59,59,0.13)")],
            threshold=dict(line=dict(color="#d03b3b", width=3),
                           thickness=0.9, value=threshold * 100),
        )))
    fig.update_layout(title="Distress probability", height=280, **_LAYOUT)
    return fig


def model_metrics_bar(metrics: dict, title: str = "Test metrics") -> go.Figure:
    """Horizontal bar of a model's headline metrics (single hue)."""
    keys = ["roc_auc", "pr_auc", "balanced_accuracy", "recall", "precision",
            "f1", "accuracy"]
    rows = [(k.replace("_", " ").upper(), metrics[k])
            for k in keys if k in metrics]
    dfm = pd.DataFrame(rows, columns=["metric", "value"])
    fig = px.bar(dfm, x="value", y="metric", orientation="h",
                 color_discrete_sequence=[SEQ_BLUE],
                 text=dfm["value"].map(lambda v: f"{v:.3f}"))
    fig.update_traces(textposition="outside",
                      hovertemplate="%{y}: %{x:.4f}<extra></extra>")
    fig.update_layout(xaxis_range=[0, 1.12], xaxis_title=None,
                      yaxis_title=None, showlegend=False,
                      height=60 + 40 * len(dfm))
    return _base(fig, title)
