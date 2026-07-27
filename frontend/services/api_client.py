"""Thin typed client over the FastAPI backend REST API.

All business logic lives in the backend; this module only performs HTTP
calls, error normalization, and light caching via Streamlit.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from config import API_PREFIX, LOCAL_BEST_MODEL_JSON, REQUEST_TIMEOUT, get_api_url


class APIError(Exception):
    """Normalized backend/API failure with a user-displayable message."""

    def __init__(self, message: str, status: Optional[int] = None,
                 detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class APIClient:
    """Client for the Financial Crisis EWS REST API."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or get_api_url()).rstrip("/")
        self.prefix = f"{self.base_url}{API_PREFIX}"

    # ── low-level ─────────────────────────────────────────────────────────────
    def _request(self, method: str, path: str,
                 payload: Optional[dict] = None) -> tuple[Any, float]:
        """Return ``(json_body, latency_ms)`` or raise :class:`APIError`."""
        url = f"{self.prefix}{path}" if path.startswith("/") else path
        t0 = time.perf_counter()
        try:
            resp = requests.request(method, url, json=payload,
                                    timeout=REQUEST_TIMEOUT)
        except requests.exceptions.ConnectionError:
            raise APIError(f"Cannot reach backend at {self.base_url}. "
                           "Is the FastAPI server running?")
        except requests.exceptions.Timeout:
            raise APIError(f"Backend timed out after {REQUEST_TIMEOUT}s.")
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code >= 400:
            detail: Any = None
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            msg = detail.get("detail", detail) if isinstance(detail, dict) \
                else detail
            raise APIError(f"API error {resp.status_code}: {msg}",
                           status=resp.status_code, detail=detail)
        try:
            return resp.json(), latency_ms
        except ValueError:
            raise APIError("Backend returned a non-JSON response.")

    # ── endpoints ─────────────────────────────────────────────────────────────
    def root(self) -> tuple[dict, float]:
        return self._request("GET", f"{self.base_url}/")

    def health(self) -> tuple[dict, float]:
        return self._request("GET", "/health")

    def version(self) -> tuple[dict, float]:
        return self._request("GET", "/version")

    def models(self) -> tuple[dict, float]:
        return self._request("GET", "/models")

    def metrics(self) -> tuple[dict, float]:
        return self._request("GET", "/metrics")

    def predict(self, features: Dict[str, float],
                id: Optional[str] = None) -> dict:
        body, _ = self._request("POST", "/predict",
                                {"features": features, "id": id})
        return body

    def predict_batch(self, instances: List[dict]) -> dict:
        body, _ = self._request("POST", "/predict/batch",
                                {"instances": instances})
        return body

    def validate(self, instances: List[dict]) -> dict:
        body, _ = self._request("POST", "/validate",
                                {"instances": instances})
        return body


# ── cached helpers (shared by pages) ──────────────────────────────────────────
def get_client() -> APIClient:
    """The single shared client — always bound to the session's backend URL.

    Every page and service must use this (or pass the same URL explicitly)
    so the backend shown in the sidebar is the backend actually called.
    """
    base_url = st.session_state.get("api_url") or get_api_url()
    return APIClient(base_url)


@st.cache_data(ttl=15, show_spinner=False)
def cached_health(base_url: str) -> Optional[dict]:
    """Health payload, or None if the backend is unreachable."""
    try:
        body, latency = APIClient(base_url).health()
        body["_latency_ms"] = round(latency, 1)
        return body
    except APIError:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def cached_models(base_url: str) -> Optional[dict]:
    try:
        body, _ = APIClient(base_url).models()
        return body
    except APIError:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_feature_schema(base_url: str) -> List[str]:
    """Model feature names, resolved dynamically from the backend.

    Probes the existing ``POST /validate`` endpoint with a single sentinel
    instance: the backend reports every missing feature by name
    (``instances[0].<feature>`` / "missing required feature"), which yields
    the exact schema of the *active* model — no backend changes, no logic
    duplication. Falls back to the local model descriptor only when the
    backend is unreachable (local development convenience).
    """
    try:
        report = APIClient(base_url).validate(
            [{"id": "schema-probe", "features": {"__schema_probe__": 0.0}}])
        features = sorted(
            e["field"].split(".", 1)[1] for e in report.get("errors", [])
            if e.get("issue") == "missing required feature"
            and "." in e.get("field", ""))
        if features:
            return features
    except APIError:
        pass
    try:  # read-only local fallback (backend offline / pre-start)
        with open(LOCAL_BEST_MODEL_JSON, encoding="utf-8") as f:
            desc = json.load(f)
        return desc.get("feature_schema", {}).get("features", [])
    except (OSError, json.JSONDecodeError):
        return []
