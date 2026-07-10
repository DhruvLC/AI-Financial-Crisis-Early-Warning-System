"""IMF Data — via the SDMX-JSON CompactData REST API (open, no key).

API: https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/{flow}/{key}
key form: {frequency}.{area}.{indicator}
"""
from __future__ import annotations

import pandas as pd

from ..base import BaseIngestor, IngestionError

BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData"


class IMFIngestor(BaseIngestor):
    name = "imf"

    def fetch(self) -> pd.DataFrame:
        flow = self.config.get("dataflow", "IFS")
        indicator = self.config.get("indicator")
        freq = self.config.get("frequency", "A")
        areas = self.config.get("areas", [])
        if not indicator or not areas:
            raise IngestionError("imf requires 'indicator' and 'areas' in config")

        rows: list[dict] = []
        for area in areas:
            key = f"{freq}.{area}.{indicator}"
            url = f"{BASE}/{flow}/{key}"
            try:
                payload = self.http_get_json(url)
            except Exception as exc:
                self.log.warning("IMF fetch failed for %s (%s)", area, exc)
                continue

            series = (payload.get("CompactData", {})
                             .get("DataSet", {})
                             .get("Series"))
            if not series:
                self.log.warning("no IMF series for %s", area)
                continue
            observations = series.get("Obs", [])
            if isinstance(observations, dict):      # single observation
                observations = [observations]
            for obs in observations:
                rows.append({
                    "entity_id": area,
                    "indicator": indicator,
                    "date": obs.get("@TIME_PERIOD"),
                    "value": obs.get("@OBS_VALUE"),
                })

        if not rows:
            raise IngestionError("no IMF observations returned")
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    def required_columns(self):
        return ["entity_id", "date", "value"]
