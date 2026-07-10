"""OECD Data — via the SDMX-JSON REST API (open, no key).

API: https://sdmx.oecd.org/public/rest/data/{flow}/{key}?format=jsondata
Falls back gracefully if the dataflow/key shape changes.
"""
from __future__ import annotations

import pandas as pd

from ..base import BaseIngestor, IngestionError

BASE = "https://sdmx.oecd.org/public/rest/data"


class OECDIngestor(BaseIngestor):
    name = "oecd"

    def fetch(self) -> pd.DataFrame:
        flow = self.config.get("dataflow", "MEI_CLI")
        subject = self.config.get("subject", "BSCICP03")
        areas = self.config.get("areas", [])
        if not areas:
            raise IngestionError("oecd requires 'areas' in config")

        # SDMX key: SUBJECT.AREA (dot-joined dimensions). '+' unions areas.
        key = f"{subject}.{'+'.join(areas)}"
        url = f"{BASE}/{flow}/{key}?format=jsondata"
        payload = self.http_get_json(url, headers={"Accept": "application/vnd.sdmx.data+json"})

        try:
            data_sets = payload["data"]["dataSets"][0]["series"]
            structure = payload["data"]["structure"]["dimensions"]["series"]
            time_vals = payload["data"]["structure"]["dimensions"]["observation"][0]["values"]
        except (KeyError, IndexError) as exc:
            raise IngestionError(f"unexpected OECD SDMX payload: {exc}") from exc

        # Build lookup of series-dimension values (to recover area codes).
        dim_values = [d["values"] for d in structure]

        rows: list[dict] = []
        for series_key, series in data_sets.items():
            idx = [int(i) for i in series_key.split(":")]
            labels = {structure[d]["id"]: dim_values[d][i]["id"]
                      for d, i in enumerate(idx)}
            for t_idx, obs in series.get("observations", {}).items():
                rows.append({
                    "entity_id": labels.get("REF_AREA") or labels.get("LOCATION"),
                    "subject": subject,
                    "date": time_vals[int(t_idx)]["id"],
                    "value": obs[0],
                })

        if not rows:
            raise IngestionError("no OECD observations parsed")
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    def required_columns(self):
        return ["entity_id", "date", "value"]
