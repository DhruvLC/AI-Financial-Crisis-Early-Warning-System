"""Alpha Vantage — market data via REST API (requires ALPHAVANTAGE_API_KEY)."""
from __future__ import annotations

import os

import pandas as pd

from ..base import BaseIngestor, IngestionError

ENDPOINT = "https://www.alphavantage.co/query"


class AlphaVantageIngestor(BaseIngestor):
    name = "alpha_vantage"

    def fetch(self) -> pd.DataFrame:
        key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not key:
            raise IngestionError(
                "ALPHAVANTAGE_API_KEY not set. Get a free key at "
                "https://www.alphavantage.co/support/#api-key"
            )
        symbols = self.config.get("symbols", [])
        function = self.config.get("function", "TIME_SERIES_DAILY")
        if not symbols:
            raise IngestionError("alpha_vantage requires 'symbols' in config")

        frames = []
        for symbol in symbols:
            url = (f"{ENDPOINT}?function={function}&symbol={symbol}"
                   f"&outputsize=compact&apikey={key}")
            data = self.http_get_json(url)

            # Free tier returns a "Note"/"Information" throttling message instead of data.
            if any(k in data for k in ("Note", "Information", "Error Message")):
                msg = data.get("Note") or data.get("Information") or data.get("Error Message")
                self.log.warning("alpha_vantage throttled/error for %s: %s", symbol, msg)
                continue

            ts_key = next((k for k in data if "Time Series" in k), None)
            if not ts_key:
                self.log.warning("unexpected payload for %s: keys=%s", symbol, list(data))
                continue

            part = pd.DataFrame(data[ts_key]).T.reset_index()
            part.columns = ["date"] + [c.split(". ")[-1] for c in part.columns[1:]]
            part["symbol"] = symbol.upper()
            part["entity_id"] = symbol.upper()
            frames.append(part)

        if not frames:
            raise IngestionError("no Alpha Vantage data (throttled or empty)")
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def required_columns(self):
        return ["date", "symbol"]
