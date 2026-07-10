"""Yahoo Finance (yfinance) — prices, volume, returns, volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseIngestor, IngestionError


class YahooFinanceIngestor(BaseIngestor):
    name = "yahoo_finance"

    def fetch(self) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:
            raise IngestionError(f"yfinance not installed: {exc}") from exc

        tickers = self.config.get("tickers", [])
        if not tickers:
            raise IngestionError("yahoo_finance requires 'tickers' in config")

        frames = []
        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    start=self.config.get("start", "2015-01-01"),
                    interval=self.config.get("interval", "1d"),
                    auto_adjust=True, progress=False,
                )
            except Exception as exc:
                self.log.warning("download failed for %s (%s)", ticker, exc)
                continue
            if raw is None or raw.empty:
                self.log.warning("no data for %s", ticker)
                continue

            raw = raw.reset_index()
            # yfinance can return a MultiIndex column frame for single tickers
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw["ticker"] = ticker.upper()
            raw["entity_id"] = ticker.upper()          # canonical entity key
            raw["daily_return"] = raw["Close"].pct_change()
            raw["volatility_21d"] = raw["daily_return"].rolling(21).std() * np.sqrt(252)
            frames.append(raw)

        if not frames:
            raise IngestionError("no ticker data downloaded")
        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={"Date": "date"})
        return df

    def required_columns(self):
        return ["date", "ticker", "Close"]
