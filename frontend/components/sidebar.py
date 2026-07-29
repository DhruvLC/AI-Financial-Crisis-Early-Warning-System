"""Shared sidebar — backend status, model version, and page setup helper."""
from __future__ import annotations

import streamlit as st

from config import APP_ICON, APP_TITLE, APP_VERSION, get_api_url
from components.theme import chip, inject_css
from services.api_client import cached_health


def setup_page(title: str, icon: str = APP_ICON) -> None:
    """Standard page config — call first on every page."""
    st.set_page_config(page_title=f"{title} · {APP_TITLE}", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    if "api_url" not in st.session_state:
        st.session_state.api_url = get_api_url()
    inject_css()


def render_sidebar() -> None:
    """Backend connection panel shown on every page's sidebar."""
    with st.sidebar:
        st.markdown(
            f"""<div style="display:flex; align-items:center; gap:.6rem;
                 margin-bottom:.2rem;">
                 <div style="font-size:1.9rem;">{APP_ICON}</div>
                 <div><div style="font-weight:800; font-size:1.02rem;
                   line-height:1.1;">Financial Crisis EWS</div>
                   <div style="font-size:.75rem; color:#5b5b57;">
                   Enterprise Risk Intelligence · v{APP_VERSION}</div></div>
                 </div>""",
            unsafe_allow_html=True)
        st.caption("Streamlit frontend · FastAPI backend")
        st.divider()

        url = st.text_input("Backend URL", value=st.session_state.api_url,
                            help="FastAPI base URL (no trailing slash)")
        if url.rstrip("/") != st.session_state.api_url:
            st.session_state.api_url = url.rstrip("/")
            cached_health.clear()
            st.rerun()

        health = cached_health(st.session_state.api_url)
        if health and health.get("status") == "ok":
            st.markdown("**Backend** &nbsp; "
                        + chip(f"Online · {health.get('_latency_ms', '?')} ms",
                               "good"), unsafe_allow_html=True)
            if health.get("model_loaded"):
                st.markdown("**Model** &nbsp; "
                            + chip("Loaded", "good"), unsafe_allow_html=True)
                st.caption(f"{health.get('model_version')} "
                           f"· {health.get('algorithm')}")
            else:
                st.markdown("**Model** &nbsp; "
                            + chip("Not loaded", "warn"),
                            unsafe_allow_html=True)
        elif health:
            st.markdown("**Backend** &nbsp; "
                        + chip(f"Degraded: {health.get('status')}", "warn"),
                        unsafe_allow_html=True)
        else:
            st.markdown("**Backend** &nbsp; " + chip("Offline", "crit"),
                        unsafe_allow_html=True)

        if st.button("↻ Refresh status", use_container_width=True):
            cached_health.clear()
            st.rerun()

        st.divider()
        st.caption(f"© 2026 Financial Crisis EWS · v{APP_VERSION}  \n"
                   "Research/educational use — not financial advice.")
