"""IMF Data — via the DataMapper REST API (open, no key).

The legacy SDMX-JSON host (dataservices.imf.org) was decommissioned, so this
uses the current DataMapper API:
    https://www.imf.org/external/datamapper/api/v1/{indicator}/{area}/{area}/...

Response shape: {"values": {INDICATOR: {AREA: {"YEAR": value, ...}}}}
Areas are ISO-3 country codes (USA, GBR, JPN); indicators are WEO/IFS codes
(e.g. NGDPD = GDP at current prices, USD). List indicators at
    https://www.imf.org/external/datamapper/api/v1/indicators
"""
from __future__ import annotations

import pandas as pd

from ..base import BaseIngestor, IngestionError

BASE = "https://www.imf.org/external/datamapper/api/v1"


class IMFIngestor(BaseIngestor):
    name = "imf"

    def fetch(self) -> pd.DataFrame:
        indicator = self.config.get("indicator")
        areas = self.config.get("areas", [])
        if not indicator or not areas:
            raise IngestionError("imf requires 'indicator' and 'areas' in config")

        url = f"{BASE}/{indicator}/{'/'.join(areas)}"
        payload = self.http_get_json(url)

        values = (payload or {}).get("values", {}).get(indicator)
        if not values:
            raise IngestionError(
                f"no IMF observations returned for '{indicator}' "
                f"(areas={areas}); check the indicator code at {BASE}/indicators"
            )

        start = self.config.get("start")
        rows: list[dict] = []
        for area, by_year in values.items():
            if not isinstance(by_year, dict):
                continue
            for year, value in by_year.items():
                if start is not None and str(year) < str(start):
                    continue
                rows.append({
                    "entity_id": area,
                    "indicator": indicator,
                    "date": year,
                    "value": value,
                })

        if not rows:
            raise IngestionError("no IMF observations returned")
        df = pd.DataFrame(rows)
        df["date"] = pd.to_numeric(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.sort_values(["entity_id", "date"]).reset_index(drop=True)

    def required_columns(self):
        return ["entity_id", "date", "value"]
