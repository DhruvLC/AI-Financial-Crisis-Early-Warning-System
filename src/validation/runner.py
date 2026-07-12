"""Orchestrates the full Data Validation pass over the ingested corpus.

For every known/ingested source it loads the interim dataset, runs each check
with per-check and per-dataset exception isolation, computes a quality score,
and writes reports. Mirrors :class:`ingestion.runner.IngestionRunner` in style
and in how it reconstructs storage paths from the shared YAML config.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

import pandas as pd

from ingestion.logging_config import get_logger

from .base import DatasetReport, Severity
from .checks import DEFAULT_CHECKS
from .quality import QualityScorer
from .report import ReportGenerator
from .schemas import SOURCE_SCHEMAS, schema_for


class DataValidationError(RuntimeError):
    """Raised when validation finds fatal errors and fail_fast is enabled."""


class DataValidationRunner:
    """Run all validation checks across every ingested dataset."""

    def __init__(self, storage: dict, dv_cfg: dict | None = None) -> None:
        self.interim_dir = storage["interim_dir"]
        self.metadata_dir = storage.get("metadata_layer_dir")
        self.cfg = dict(dv_cfg or {})
        self.log = get_logger("validation.runner")
        self.fail_fast = bool(self.cfg.get("fail_fast", False))
        self.reports_dir = self.cfg.get("reports_dir", "reports/validation")
        self.scorer = QualityScorer(self.cfg.get("quality_weights"))
        self.reporter = ReportGenerator(self.reports_dir)
        # Instantiate each check once; they are stateless across datasets.
        self.checks = [cls(self.cfg) for cls in DEFAULT_CHECKS]

    # ── discovery ────────────────────────────────────────────────────────────
    def _discover_interim(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for ext in ("parquet", "csv"):
            for path in glob.glob(os.path.join(self.interim_dir, f"*.{ext}")):
                name = os.path.splitext(os.path.basename(path))[0]
                found.setdefault(name, path)   # parquet wins over csv
        return found

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    # ── orchestration ────────────────────────────────────────────────────────
    def run(self, only: list[str] | None = None) -> list[DatasetReport]:
        interim = self._discover_interim()
        targets = sorted(set(SOURCE_SCHEMAS) | set(interim))
        if only:
            targets = [t for t in targets if t in only]
        self.log.info("validation targets: %s", targets)

        reports: list[DatasetReport] = []
        for source in targets:
            reports.append(self._validate_dataset(source, interim.get(source)))

        self.reporter.write_summary(reports)
        n_valid = sum(r.is_valid for r in reports)
        n_present = sum(r.present for r in reports)
        self.log.info("validation complete: %d/%d present dataset(s) valid",
                      n_valid, n_present)

        if self.fail_fast:
            failed = [r.source for r in reports if r.present and not r.is_valid]
            if failed:
                raise DataValidationError(
                    f"validation failed for: {failed}. "
                    "Set data_validation.fail_fast=false to bypass."
                )
        return reports

    def _validate_dataset(self, source: str, path: str | None) -> DatasetReport:
        spec = schema_for(source)

        # ── missing dataset ──────────────────────────────────────────────────
        if not path:
            self.log.warning("[%s] no interim file found — skipping", source)
            return DatasetReport(source=source, present=False)

        # ── corrupted / unreadable file ──────────────────────────────────────
        try:
            df = self._read(path)
        except Exception as exc:  # noqa: BLE001 - isolate a corrupt file
            self.log.error("[%s] could not read %s: %s", source, path, exc)
            return DatasetReport(source=source, present=True, interim_path=path,
                                 load_error=str(exc))

        report = DatasetReport(
            source=source, present=True, interim_path=path,
            n_rows=len(df), n_cols=df.shape[1],
        )
        ctx = {"now": datetime.now(timezone.utc), "source": source}

        for check in self.checks:
            outcome = check.run(df, spec, ctx)   # run() never raises
            report.outcomes.append(outcome)
            for f in outcome.findings:
                if f.level == Severity.ERROR:
                    self.log.error("[%s/%s] %s", source, check.name, f.message)
                elif f.level == Severity.WARN:
                    self.log.warning("[%s/%s] %s", source, check.name, f.message)

        score, grade, components = self.scorer.score(report.outcomes)
        report.quality_score = score
        report.quality_grade = grade
        report.quality_components = components
        self.log.info("[%s] quality score %.1f (%s), %d error(s), %d warning(s)",
                      source, score, grade, report.n_errors, report.n_warnings)

        self.reporter.write_dataset(report)
        return report
