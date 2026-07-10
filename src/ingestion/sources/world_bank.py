"""World Bank Open Data — global economic indicators (open REST API, no key).

API: https://api.worldbank.org/v2/country/{codes}/indicator/{ind}?format=json
"""
from __future__ import annotations

import pandas as pd

from ..base import BaseIngestor, IngestionError

BASE = "https://api.worldbank.org/v2"


class WorldBankIngestor(BaseIngestor):
    name = "world_bank"

    def fetch(self) -> pd.DataFrame:
        countries = ";".join(self.config.get("countries", ["US"]))
        indicators = self.config.get("indicators", {})
        start, end = self.config.get("start", 2000), self.config.get("end", 2023)
        if not indicators:
            raise IngestionError("world_bank requires an 'indicators' mapping")

        frames = []
        for label, code in indicators.items():
            url = (f"{BASE}/country/{countries}/indicator/{code}"
                   f"?format=json&date={start}:{end}&per_page=20000")
            payload = self.http_get_json(url)
            # WB responses are [metadata, data]; guard against error dicts
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                self.log.warning("no data for indicator %s (%s)", label, code)
                continue
            recs = [{
                "entity_id": r["countryiso3code"],
                "country": r["country"]["value"],
                "date": r["date"],
                label: r["value"],
            } for r in payload[1]]
            part = pd.DataFrame(recs)
            frames.append(part)

        if not frames:
            raise IngestionError("no World Bank indicators returned")
        # merge indicators on (entity_id, country, date)
        df = frames[0]
        for part in frames[1:]:
            df = df.merge(part, on=["entity_id", "country", "date"], how="outer")
        df["date"] = pd.to_numeric(df["date"], errors="coerce")
        return df.sort_values(["entity_id", "date"]).reset_index(drop=True)

    def required_columns(self):
        return ["entity_id", "date"]
