"""Shared sidebar — backend status, model version, and page setup helper."""
from __future__ import annotations

import streamlit as st

from config import APP_ICON, APP_TITLE, get_api_url
from services.api_client import cached_health


def setup_page(title: str, icon: str = APP_ICON) -> None:
    """Standard page config — call first on every page."""
    st.set_page_config(page_title=f"{title} · {APP_TITLE}", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    if "api_url" not in st.session_state:
        st.session_state.api_url = get_api_url()


def render_sidebar() -> None:
    """Backend connection panel shown on every page's sidebar."""
    with st.sidebar:
        st.markdown(f"### {APP_ICON} {APP_TITLE}")
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
            st.success(f"Backend online · {health.get('_latency_ms', '?')} ms")
            if health.get("model_loaded"):
                st.caption(f"Model **{health.get('model_version')}** "
                           f"({health.get('algorithm')})")
            else:
                st.warning("Model not loaded")
        elif health:
            st.warning(f"Backend degraded: {health.get('status')}")
        else:
            st.error("Backend offline")

        if st.button("↻ Refresh status", use_container_width=True):
            cached_health.clear()
            st.rerun()
