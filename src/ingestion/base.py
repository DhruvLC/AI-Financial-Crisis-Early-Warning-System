"""Abstract base ingestor — the shared template every source subclasses."""
from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .logging_config import get_logger
from .metadata import MetadataWriter
from .validation import DataValidator


@dataclass
class IngestionResult:
    """Structured outcome of one source's ingestion run."""
    source: str
    status: str                       # "success" | "failed" | "skipped"
    n_rows: int = 0
    raw_path: str | None = None
    interim_path: str | None = None
    metadata_path: str | None = None
    error: str | None = None
    validation: dict = field(default_factory=dict)


class IngestionError(Exception):
    """Raised when a source cannot produce usable data."""


class BaseIngestor(abc.ABC):
    """Template-method base class.

    Subclasses implement :meth:`fetch` (and optionally :meth:`required_columns`
    / :meth:`clean`). Everything else — validation, storage, metadata, logging,
    error handling — is provided here so behaviour is uniform across sources.
    """

    #: short machine name, e.g. "fred"; used for filenames/logging
    name: str = "base"

    def __init__(
        self,
        config: dict,
        storage: dict,
        http_cfg: dict | None = None,
        validation_cfg: dict | None = None,
    ):
        self.config = config or {}
        self.storage = storage
        self.http_cfg = http_cfg or {}
        self.log = get_logger(f"ingest.{self.name}")
        # Per-source `max_missing_pct` overrides the global validation default.
        # Macro sources merge series of differing frequencies (e.g. quarterly GDP
        # vs. daily yields) and are legitimately sparse, so they need a higher cap.
        default_missing = (validation_cfg or {}).get("max_missing_pct", 0.5)
        max_missing = self.config.get("max_missing_pct", default_missing)
        self.validator = DataValidator(max_missing_pct=max_missing)
        self.metadata_writer = MetadataWriter(storage["metadata_layer_dir"])

    # ── to be implemented by subclasses ──────────────────────────────────
    @abc.abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Retrieve data from the source and return a tidy DataFrame."""

    def required_columns(self) -> list[str] | None:
        """Override to enable schema validation."""
        return None

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional light cleaning before interim storage (override as needed)."""
        return df.drop_duplicates().reset_index(drop=True)

    # ── shared machinery ─────────────────────────────────────────────────
    def _raw_dir(self) -> str:
        path = os.path.join(self.storage["raw_dir"], self.name)
        os.makedirs(path, exist_ok=True)
        return path

    def save_raw(self, df: pd.DataFrame, filename: str | None = None) -> str:
        filename = filename or f"{self.name}.csv"
        path = os.path.join(self._raw_dir(), filename)
        df.to_csv(path, index=False)
        self.log.info("raw stored: %s (%d rows)", path, len(df))
        return path

    def save_interim(self, df: pd.DataFrame) -> str:
        os.makedirs(self.storage["interim_dir"], exist_ok=True)
        path = os.path.join(self.storage["interim_dir"], f"{self.name}.parquet")
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:  # pyarrow/fastparquet missing → CSV fallback
            self.log.warning("parquet failed (%s); writing CSV instead", exc)
            path = path.replace(".parquet", ".csv")
            df.to_csv(path, index=False)
        self.log.info("interim stored: %s", path)
        return path

    def http_get_json(self, url: str, **kwargs) -> Any:
        """GET with retry/backoff. Requires `requests`; raises on final failure."""
        import requests

        retries = int(self.http_cfg.get("max_retries", 3))
        backoff = float(self.http_cfg.get("backoff_seconds", 2))
        timeout = float(self.http_cfg.get("timeout_seconds", 30))
        last: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=timeout, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - retry any transient failure
                last = exc
                wait = backoff * (2 ** attempt)
                self.log.warning("GET %s failed (attempt %d/%d): %s; retrying in %.0fs",
                                 url, attempt + 1, retries, exc, wait)
                time.sleep(wait)
        raise IngestionError(f"GET {url} failed after {retries} attempts: {last}")

    # ── the template method ──────────────────────────────────────────────
    def run(self) -> IngestionResult:
        """fetch → validate → clean → store raw+interim → metadata."""
        self.log.info("=== ingest '%s' started ===", self.name)
        try:
            df = self.fetch()
            if df is None or df.empty:
                raise IngestionError("fetch returned no data")

            report = self.validator.validate(
                df, self.name, required_columns=self.required_columns()
            )
            for w in report.warnings:
                self.log.warning("validation: %s", w)
            if not report.is_valid:
                raise IngestionError(f"validation failed: {report.errors}")

            raw_path = self.save_raw(df)
            cleaned = self.clean(df)
            interim_path = self.save_interim(cleaned)
            meta_path = self.metadata_writer.write(
                self.name, cleaned, raw_path, interim_path, report.as_dict()
            )

            self.log.info("=== ingest '%s' success (%d rows) ===", self.name, len(cleaned))
            return IngestionResult(
                source=self.name, status="success", n_rows=len(cleaned),
                raw_path=raw_path, interim_path=interim_path,
                metadata_path=meta_path, validation=report.as_dict(),
            )
        except Exception as exc:  # graceful per-source failure isolation
            self.log.error("=== ingest '%s' FAILED: %s ===", self.name, exc)
            return IngestionResult(source=self.name, status="failed", error=str(exc))
