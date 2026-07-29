"""Enterprise design system — global CSS, KPI cards, chips, and section headers.

Colors mirror the validated reference palette used in ``.streamlit/config.toml``
and ``components/charts.py``. All values here are standard CSS (6-digit HEX,
rgb/rgba, or named colors) — never Plotly gauge colors — so nothing here can
trigger a Plotly ValueError.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

# ── palette (CSS) ─────────────────────────────────────────────────────────────
PRIMARY = "#2a78d6"
GOOD = "#0ca30c"
WARN = "#fab219"
CRIT = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#5b5b57"
SURFACE = "#fcfcfb"
LINE = "#e1e0d9"

_CSS = """
<style>
/* ── enterprise design system ─────────────────────────────────────────── */
:root {
  --ews-primary: #2a78d6;
  --ews-good: #0ca30c;
  --ews-warn: #fab219;
  --ews-crit: #d03b3b;
  --ews-ink: #0b0b0b;
  --ews-muted: #5b5b57;
  --ews-surface: #fcfcfb;
  --ews-line: #e1e0d9;
  --ews-radius: 14px;
  --ews-shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.10);
  --ews-shadow-lg: 0 6px 20px rgba(16,24,40,.10);
}
.block-container { padding-top: 2.2rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.01em; font-weight: 700; }

/* KPI / metric cards */
div[data-testid="stMetric"] {
  background: var(--ews-surface);
  border: 1px solid var(--ews-line);
  border-radius: var(--ews-radius);
  padding: 1rem 1.1rem;
  box-shadow: var(--ews-shadow);
  transition: box-shadow .18s ease, transform .18s ease;
}
div[data-testid="stMetric"]:hover {
  box-shadow: var(--ews-shadow-lg);
  transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] p {
  font-size: .78rem; font-weight: 600; letter-spacing: .02em;
  text-transform: uppercase; color: var(--ews-muted);
}

/* bordered containers → cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--ews-radius);
  box-shadow: var(--ews-shadow);
}

/* buttons */
div.stButton > button, div.stDownloadButton > button {
  border-radius: 10px; font-weight: 600;
  transition: transform .12s ease, box-shadow .12s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  transform: translateY(-1px); box-shadow: var(--ews-shadow);
}
div.stButton > button[kind="primary"] {
  box-shadow: 0 2px 8px rgba(42,120,214,.28);
}

/* dataframes */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--ews-line);
  border-radius: var(--ews-radius); overflow: hidden;
}

/* sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid var(--ews-line); }

/* tabs */
button[data-baseweb="tab"] { font-weight: 600; }

/* expanders → card look */
div[data-testid="stExpander"] {
  border: 1px solid var(--ews-line);
  border-radius: var(--ews-radius);
  overflow: hidden;
}
div[data-testid="stExpander"] summary { font-weight: 600; }

/* alerts — soften corners */
div[data-testid="stAlert"] { border-radius: 12px; }

/* accessibility: visible keyboard focus */
div.stButton > button:focus-visible,
div.stDownloadButton > button:focus-visible,
a:focus-visible {
  outline: 2px solid var(--ews-primary);
  outline-offset: 2px;
}

/* accessibility: honor reduced-motion preference */
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}

/* responsive: tighter padding on small screens */
@media (max-width: 768px) {
  .block-container { padding-top: 1.2rem; padding-left: 1rem;
                     padding-right: 1rem; }
  div[data-testid="stMetric"] { padding: .7rem .8rem; }
}
</style>
"""

def inject_css() -> None:
    """Inject the global stylesheet once per session run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, icon: str = "") -> None:
    """Gradient hero banner for the top of a page."""
    st.markdown(
        f"""<div style="
            background: linear-gradient(120deg,#2a78d6 0%,#1f5fae 60%,#17457e 100%);
            border-radius:16px; padding:1.6rem 1.9rem; margin-bottom:1.2rem;
            box-shadow:0 8px 24px rgba(31,95,174,.28); color:#f9f9f7;">
            <div style="font-size:1.7rem; font-weight:800; letter-spacing:-.02em;">
              {icon} {title}</div>
            <div style="opacity:.92; font-size:1rem; margin-top:.25rem;">
              {subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def chip(text: str, kind: str = "neutral") -> str:
    """Return an inline status-chip HTML string.

    ``kind`` ∈ {good, warn, crit, info, neutral}.
    """
    palette = {
        "good": (GOOD, "rgba(12,163,12,0.12)"),
        "warn": (WARN, "rgba(250,178,25,0.15)"),
        "crit": (CRIT, "rgba(208,59,59,0.12)"),
        "info": (PRIMARY, "rgba(42,120,214,0.12)"),
        "neutral": (MUTED, "rgba(91,91,87,0.10)"),
    }
    fg, bg = palette.get(kind, palette["neutral"])
    return (f'<span style="display:inline-block; padding:.18rem .6rem; '
            f'border-radius:999px; font-size:.78rem; font-weight:700; '
            f'color:{fg}; background:{bg}; border:1px solid {fg}33;">'
            f'{text}</span>')


def section(title: str, caption: Optional[str] = None) -> None:
    """Consistent section header with an accent rule."""
    st.markdown(
        f"""<div style="margin:.4rem 0 .2rem;">
            <span style="font-size:1.15rem; font-weight:700; color:{INK};
              border-left:4px solid {PRIMARY}; padding-left:.55rem;">
              {title}</span></div>""",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)

