"""FRED — macroeconomic series.

Uses `fredapi` when FRED_API_KEY is set; otherwise falls back to the public
CSV download endpoint (no key required):
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
"""
from __future__ import annotations

import io
import os

import pandas as pd

from ..base import BaseIngestor, IngestionError

CSV_ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class FredIngestor(BaseIngestor):
    name = "fred"

    def _fetch_via_api(self, series: dict, start: str) -> pd.DataFrame | None:
        key = os.environ.get("FRED_API_KEY")
        if not key:
            return None
        try:
            from fredapi import Fred
        except Exception:
            self.log.info("fredapi not installed; using CSV fallback")
            return None
        fred = Fred(api_key=key)
        frames = []
        for label, sid in series.items():
            s = fred.get_series(sid, observation_start=start)
            frames.append(s.rename(label))
        df = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "date"})
        return df

    def _fetch_via_csv(self, series: dict, start: str) -> pd.DataFrame:
        import requests

        timeout = float(self.http_cfg.get("timeout_seconds", 30))
        merged: pd.DataFrame | None = None
        for label, sid in series.items():
            url = f"{CSV_ENDPOINT}?id={sid}"
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
            except Exception as exc:
                self.log.warning("FRED CSV fetch failed for %s (%s)", sid, exc)
                continue
            part = pd.read_csv(io.StringIO(resp.text))
            part.columns = ["date", label]
            part["date"] = pd.to_datetime(part["date"], errors="coerce")
            part[label] = pd.to_numeric(part[label], errors="coerce")
            merged = part if merged is None else merged.merge(part, on="date", how="outer")
        if merged is None:
            raise IngestionError("all FRED series failed to download")
        merged = merged[merged["date"] >= pd.to_datetime(start)]
        return merged.sort_values("date").reset_index(drop=True)

    def fetch(self) -> pd.DataFrame:
        series = self.config.get("series", {})
        start = self.config.get("start", "2000-01-01")
        if not series:
            raise IngestionError("fred requires a 'series' mapping in config")
        df = self._fetch_via_api(series, start)
        if df is None:
            df = self._fetch_via_csv(series, start)
        return df

    def required_columns(self):
        return ["date"]
