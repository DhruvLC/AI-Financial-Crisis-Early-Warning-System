"""Reusable card / badge / alert components (theme-safe, no hard colors on text)."""
from __future__ import annotations

from typing import Optional

import streamlit as st

from config import RISK_COLORS


def metric_card(label: str, value: str, delta: Optional[str] = None,
                help: Optional[str] = None) -> None:
    st.metric(label, value, delta=delta, help=help)


def status_badge(ok: bool, ok_text: str = "Online",
                 fail_text: str = "Offline") -> str:
    """Inline markdown status badge."""
    return f"🟢 **{ok_text}**" if ok else f"🔴 **{fail_text}**"


def risk_badge(risk_level: str) -> str:
    icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪")
    return f"{icon} **{risk_level}**"


def risk_banner(risk_level: str, probability: float) -> None:
    """Prominent colored banner for a single prediction result."""
    color = RISK_COLORS.get(risk_level, "#888888")
    icon = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(risk_level, "ℹ️")
    st.markdown(
        f"""<div style="border-left: 6px solid {color};
             background: {color}18; padding: 0.9rem 1.2rem;
             border-radius: 6px; margin: 0.5rem 0;">
             <span style="font-size:1.3rem;">{icon}
             <b>{risk_level} Risk</b></span><br/>
             Predicted distress probability:
             <b>{probability * 100:.2f}%</b></div>""",
        unsafe_allow_html=True)


def error_box(message: str, detail=None) -> None:
    st.error(message)
    if detail:
        with st.expander("Error details"):
            st.json(detail)


def nav_card(page: str, title: str, description: str, icon: str) -> None:
    """Quick navigation card used on the dashboard."""
    with st.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.caption(description)
        st.page_link(page, label=f"Open {title} →")
