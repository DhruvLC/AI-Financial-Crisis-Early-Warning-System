"""SEC EDGAR — company fundamentals via the XBRL 'company facts' API.

Docs: https://www.sec.gov/edgar/sec-api-documentation
SEC requires a descriptive User-Agent header (an email). No API key needed.
Endpoint: https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json
"""
from __future__ import annotations

import pandas as pd

from ..base import BaseIngestor, IngestionError

BASE = "https://data.sec.gov/api/xbrl/companyconcept"


class SECEdgarIngestor(BaseIngestor):
    name = "sec_edgar"

    def _headers(self) -> dict:
        ua = self.config.get("user_agent")
        if not ua or "example.com" in ua:
            self.log.warning(
                "SEC user_agent looks like a placeholder; set a real contact "
                "email in configs/ingestion.yaml (SEC may block otherwise)."
            )
        return {"User-Agent": ua or "FinCrisisEWS contact@example.com",
                "Accept-Encoding": "gzip, deflate"}

    def fetch(self) -> pd.DataFrame:
        ciks = [str(c).zfill(10) for c in self.config.get("ciks", [])]
        concepts = self.config.get("concepts", [])
        if not ciks or not concepts:
            raise IngestionError("sec_edgar requires 'ciks' and 'concepts' in config")

        rows: list[dict] = []
        for cik in ciks:
            for concept in concepts:
                url = f"{BASE}/CIK{cik}/us-gaap/{concept}.json"
                try:
                    data = self.http_get_json(url, headers=self._headers())
                except Exception as exc:
                    # a company may simply not report a given concept — skip, don't fail
                    self.log.warning("no data for CIK %s / %s (%s)", cik, concept, exc)
                    continue

                entity = data.get("entityName")
                for unit, facts in data.get("units", {}).items():
                    for f in facts:
                        rows.append({
                            "cik": cik,
                            "entity_id": cik,           # canonical entity key
                            "entity_name": entity,
                            "concept": concept,
                            "unit": unit,
                            "value": f.get("val"),
                            "fy": f.get("fy"),
                            "fp": f.get("fp"),
                            "form": f.get("form"),      # 10-K / 10-Q
                            "date": f.get("end"),       # period end
                            "filed": f.get("filed"),
                        })

        if not rows:
            raise IngestionError("SEC returned no facts for the requested CIKs/concepts")
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def required_columns(self):
        return ["cik", "concept", "value", "date"]
