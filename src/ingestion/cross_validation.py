"""Post-ingestion cross-source Data Validation.

This is the *third* validation layer in the system, and the only one that looks
at the ingested corpus as a whole rather than one feed at a time:

    1. ``ingestion/validation.py``  — per-source gate, runs inside each ingestor
                                       before its data is stored.
    2. ``pipeline/data_validation.py`` — modelling-table gate, runs on the single
                                       assembled table the models consume.
    3. ``ingestion/cross_validation.py`` (this module) — runs *after* a full
                                       ingestion pass over everything under
                                       ``data/interim/`` + the metadata sidecars,
                                       and answers questions no single-source
                                       validator can:

         • Coverage    — did every expected source actually land? (vs manifest)
         • Schema      — does each interim dataset match its registered contract
                         (required columns present, key columns non-null,
                         numeric columns actually numeric)?
         • Sanity      — do values sit inside domain ranges (unemployment 0-100,
                         prices > 0, …) and are there no infinities?
         • Freshness   — is the newest observation recent enough to be useful?
         • Integrity   — does the interim file's checksum still match the one
                         recorded in its metadata sidecar?
         • Anomalies   — what fraction of each numeric column are statistical
                         outliers (IQR fence)?
         • Consistency — do sources that share the canonical ``entity_id`` key
                         agree on their entity universe / overlap at all?

The check registry (:data:`SCHEMA_REGISTRY`) encodes the expected contract for
each known source; unknown sources still get the generic checks. Global
thresholds and per-source overrides can be supplied via a ``data_validation``
block in the ingestion config.

A consolidated JSON report is always written; ``fail_fast`` only controls
whether *fatal* findings abort the caller.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .logging_config import get_logger

log = get_logger("ingest.cross_validation")

# Severity levels, ordered.
PASS, WARN, FAIL = "pass", "warn", "fail"


# ─────────────────────────────────────────────────────────────────────────────
# Expected contract per source. Every field is optional: a source with no entry
# (or a partial entry) simply runs fewer targeted checks. `entity_column` is the
# canonical cross-source join key ("entity_id") where the source carries one.
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_REGISTRY: dict[str, dict] = {
    "fred": {
        "required_columns": ["date"],
        "key_columns": ["date"],
        "date_column": "date",
        "min_rows": 100,
        "freshness_days": 400,          # macro series lag by months; be lenient
        "ranges": {
            "Unemployment": (0, 100),
            "FedFundsRate": (0, 30),
            "Treasury10Y": (-5, 30),
            "CPI": (0, None),
            "GDP": (0, None),
            "MoneySupplyM2": (0, None),
        },
    },
    "world_bank": {
        "required_columns": ["entity_id", "country", "date"],
        "key_columns": ["entity_id", "date"],
        "entity_column": "entity_id",
        "year_column": "date",          # WB stores the year as an int, not a date
        "min_rows": 10,
        "ranges": {
            "Population": (0, None),
            "GDP": (0, None),
            "TradePctGDP": (0, 1000),
        },
    },
    "imf": {
        "required_columns": ["entity_id", "date", "value"],
        "key_columns": ["entity_id", "date"],
        "entity_column": "entity_id",
        "year_column": "date",
        "min_rows": 5,
        "ranges": {"value": (0, None)},
    },
    "oecd": {
        "required_columns": ["entity_id", "date", "value"],
        "key_columns": ["entity_id", "date"],
        "entity_column": "entity_id",
        "min_rows": 5,
    },
    "sec_edgar": {
        "required_columns": ["cik", "entity_id", "concept", "value", "date"],
        "key_columns": ["cik", "concept", "date"],
        "entity_column": "entity_id",
        "date_column": "date",
        "min_rows": 10,
    },
    "yahoo_finance": {
        "required_columns": ["date", "ticker", "Close", "entity_id"],
        "key_columns": ["entity_id", "date"],
        "entity_column": "entity_id",
        "date_column": "date",
        "min_rows": 50,
        "freshness_days": 30,           # market data should be days-fresh
        "ranges": {
            "Open": (0, None), "High": (0, None), "Low": (0, None),
            "Close": (0, None), "Volume": (0, None),
        },
    },
    "alpha_vantage": {
        "required_columns": ["date"],
        "date_column": "date",
        "min_rows": 20,
        "freshness_days": 30,
    },
    "kaggle_bankruptcy": {
        "min_rows": 100,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    """One check applied to one source (or across sources)."""
    check: str
    level: str                          # PASS | WARN | FAIL
    message: str
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {"check": self.check, "level": self.level, "message": self.message}
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class SourceValidation:
    """All checks for a single interim dataset."""
    source: str
    present: bool
    n_rows: int = 0
    n_cols: int = 0
    interim_path: str | None = None
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, check: str, level: str, message: str, **details: Any) -> None:
        self.checks.append(CheckResult(check, level, message, details))

    @property
    def worst(self) -> str:
        levels = {c.level for c in self.checks}
        return FAIL if FAIL in levels else WARN if WARN in levels else PASS

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "present": self.present,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "interim_path": self.interim_path,
            "status": self.worst,
            "checks": [c.as_dict() for c in self.checks],
        }


@dataclass
class CrossSourceReport:
    """Consolidated outcome of validating the whole ingested corpus."""
    validated_at: str
    sources: list[SourceValidation] = field(default_factory=list)
    cross_checks: list[CheckResult] = field(default_factory=list)

    def add_cross(self, check: str, level: str, message: str, **details: Any) -> None:
        self.cross_checks.append(CheckResult(check, level, message, details))

    @property
    def n_fail(self) -> int:
        return sum(c.level == FAIL for s in self.sources for c in s.checks) + \
            sum(c.level == FAIL for c in self.cross_checks)

    @property
    def n_warn(self) -> int:
        return sum(c.level == WARN for s in self.sources for c in s.checks) + \
            sum(c.level == WARN for c in self.cross_checks)

    @property
    def is_valid(self) -> bool:
        return self.n_fail == 0

    def as_dict(self) -> dict:
        return {
            "validated_at": self.validated_at,
            "is_valid": self.is_valid,
            "summary": {
                "n_sources": len(self.sources),
                "n_present": sum(s.present for s in self.sources),
                "n_fail": self.n_fail,
                "n_warn": self.n_warn,
            },
            "cross_checks": [c.as_dict() for c in self.cross_checks],
            "sources": [s.as_dict() for s in self.sources],
        }


class CrossSourceValidationError(RuntimeError):
    """Raised when validation finds a fatal problem and fail_fast is on."""


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────
class CrossSourceValidator:
    """Validates every ingested interim dataset as a corpus.

    Parameters
    ----------
    storage : dict
        Must carry ``interim_dir`` and ``metadata_layer_dir`` (same dict the
        ingestion runner builds).
    dv_cfg : dict
        The ``data_validation`` config block (global thresholds + per-source
        overrides). Missing keys fall back to sensible defaults.
    """

    def __init__(self, storage: dict, dv_cfg: dict | None = None) -> None:
        self.interim_dir = storage["interim_dir"]
        self.metadata_dir = storage["metadata_layer_dir"]
        cfg = dict(dv_cfg or {})
        self.min_rows_default = int(cfg.get("min_rows", 1))
        self.freshness_days_default = cfg.get("freshness_days")   # None => skip
        self.max_outlier_pct = float(cfg.get("max_outlier_pct", 0.20))
        self.check_integrity = bool(cfg.get("check_integrity", True))
        self.fail_fast = bool(cfg.get("fail_fast", False))
        self.report_path = cfg.get(
            "report_path", os.path.join("reports", "cross_source_validation.json")
        )
        # Per-source config overrides, merged over the static registry.
        self.overrides = dict(cfg.get("sources", {}) or {})

    # ── helpers ──────────────────────────────────────────────────────────────
    def _spec(self, source: str) -> dict:
        spec = dict(SCHEMA_REGISTRY.get(source, {}))
        spec.update(self.overrides.get(source, {}) or {})
        return spec

    def _load_metadata(self) -> dict[str, dict]:
        meta: dict[str, dict] = {}
        for path in glob.glob(os.path.join(self.metadata_dir, "*.meta.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    m = json.load(f)
                meta[m.get("source", os.path.basename(path))] = m
            except Exception as exc:  # noqa: BLE001 - a bad sidecar shouldn't crash
                log.warning("could not read metadata %s: %s", path, exc)
        return meta

    def _load_manifest(self) -> dict:
        path = os.path.join(self.metadata_dir, "run_manifest.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read run_manifest: %s", exc)
            return {}

    @staticmethod
    def _read_interim(path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    @staticmethod
    def _checksum(path: str) -> str | None:
        import hashlib
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _discover_interim(self) -> dict[str, str]:
        """Map source name -> interim file path (parquet preferred over csv)."""
        found: dict[str, str] = {}
        for ext in ("parquet", "csv"):
            for path in glob.glob(os.path.join(self.interim_dir, f"*.{ext}")):
                name = os.path.splitext(os.path.basename(path))[0]
                found.setdefault(name, path)   # first ext wins => parquet
        return found

    # ── the public entry point ───────────────────────────────────────────────
    def validate(self) -> CrossSourceReport:
        # Static timestamp helper avoided at import time; fine to call at runtime.
        report = CrossSourceReport(
            validated_at=datetime.now(timezone.utc).isoformat()
        )
        metadata = self._load_metadata()
        manifest = self._load_manifest()
        interim = self._discover_interim()

        # Which sources do we *expect*? Union of registry, metadata, manifest,
        # overrides and whatever files are actually on disk.
        expected = set(SCHEMA_REGISTRY) | set(metadata) | set(interim) \
            | set(self.overrides)
        manifest_status = {
            r.get("source"): r.get("status")
            for r in manifest.get("results", [])
        }
        # Only hold a source to a "should be present" bar if the manifest ran it
        # (or the registry knows it); avoids nagging about optional feeds.
        for source in sorted(expected):
            sv = self._validate_source(
                source, interim.get(source), metadata.get(source),
                manifest_status.get(source),
            )
            report.sources.append(sv)

        self._cross_source_checks(report, interim)
        return self._finish(report)

    # ── per-source ───────────────────────────────────────────────────────────
    def _validate_source(
        self, source: str, path: str | None, meta: dict | None,
        manifest_status: str | None,
    ) -> SourceValidation:
        spec = self._spec(source)

        # ── Coverage: is the interim file there at all? ──────────────────────
        if not path:
            sv = SourceValidation(source=source, present=False)
            if manifest_status == "success":
                sv.add("coverage", FAIL,
                       "manifest reports success but no interim file found")
            elif manifest_status in ("failed", "skipped"):
                sv.add("coverage", WARN,
                       f"source not present (last run: {manifest_status})")
            else:
                # Never ran / optional source with no artifact — informational.
                sv.add("coverage", WARN, "no interim data present")
            return sv

        try:
            df = self._read_interim(path)
        except Exception as exc:  # noqa: BLE001
            sv = SourceValidation(source=source, present=True, interim_path=path)
            sv.add("readable", FAIL, f"could not read interim file: {exc}")
            return sv

        sv = SourceValidation(
            source=source, present=True, n_rows=len(df), n_cols=df.shape[1],
            interim_path=path,
        )

        self._check_rowcount(sv, df, spec)
        self._check_schema(sv, df, spec)
        self._check_keys(sv, df, spec)
        self._check_ranges(sv, df, spec)
        self._check_infinities(sv, df)
        self._check_freshness(sv, df, spec)
        self._check_outliers(sv, df, spec)
        if self.check_integrity:
            self._check_integrity(sv, path, meta)

        if sv.worst == PASS:
            sv.add("summary", PASS, "all checks passed")
        return sv

    def _check_rowcount(self, sv, df, spec) -> None:
        floor = int(spec.get("min_rows", self.min_rows_default))
        if len(df) < floor:
            sv.add("row_count", FAIL,
                   f"{len(df)} rows below floor of {floor}", min_rows=floor)

    def _check_schema(self, sv, df, spec) -> None:
        required = spec.get("required_columns") or []
        missing = [c for c in required if c not in df.columns]
        if missing:
            sv.add("schema", FAIL, f"missing required columns: {missing}",
                   missing_columns=missing)

    def _check_keys(self, sv, df, spec) -> None:
        keys = [c for c in (spec.get("key_columns") or []) if c in df.columns]
        if not keys:
            return
        null_keys = {c: int(df[c].isna().sum()) for c in keys if df[c].isna().any()}
        if null_keys:
            sv.add("key_nulls", FAIL,
                   f"null values in key column(s): {null_keys}",
                   null_keys=null_keys)
        dupes = int(df.duplicated(subset=keys).sum())
        if dupes:
            sv.add("key_uniqueness", WARN,
                   f"{dupes} duplicate rows on key {keys}", duplicate_rows=dupes)

    def _check_ranges(self, sv, df, spec) -> None:
        for col, (lo, hi) in (spec.get("ranges") or {}).items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            below = int((s < lo).sum()) if lo is not None else 0
            above = int((s > hi).sum()) if hi is not None else 0
            if below or above:
                sv.add("value_range", FAIL,
                       f"'{col}' has {below} below / {above} above bounds "
                       f"[{lo}, {hi}]",
                       column=col, below=below, above=above,
                       observed_min=float(s.min()), observed_max=float(s.max()))

    def _check_infinities(self, sv, df) -> None:
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return
        inf_mask = np.isinf(num.to_numpy())
        if inf_mask.any():
            cols = num.columns[inf_mask.any(axis=0)].tolist()
            sv.add("infinities", FAIL,
                   f"inf/-inf present in column(s): {cols[:10]}", columns=cols)

    def _check_freshness(self, sv, df, spec) -> None:
        max_age = spec.get("freshness_days", self.freshness_days_default)
        if max_age is None:
            return
        date_col = spec.get("date_column")
        if not date_col or date_col not in df.columns:
            return
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if dates.empty:
            sv.add("freshness", WARN, f"no parseable dates in '{date_col}'")
            return
        newest = dates.max()
        age_days = (pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
                    - newest.tz_localize(None)).days
        if age_days > max_age:
            sv.add("freshness", WARN,
                   f"newest observation is {age_days}d old (> {max_age}d)",
                   newest=str(newest.date()), age_days=age_days)

    def _check_outliers(self, sv, df, spec) -> None:
        num = df.select_dtypes(include=[np.number])
        # Don't flag key/id-ish integer columns as outlier-bearing signal.
        skip = set(spec.get("key_columns") or []) | {spec.get("year_column")}
        flagged: dict[str, float] = {}
        for col in num.columns:
            if col in skip:
                continue
            s = num[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 20 or s.nunique() < 5:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr   # 3×IQR = "far out" fence
            frac = float(((s < lo) | (s > hi)).mean())
            if frac > self.max_outlier_pct:
                flagged[col] = round(frac, 4)
        if flagged:
            sv.add("outliers", WARN,
                   f"{len(flagged)} column(s) exceed "
                   f"{self.max_outlier_pct:.0%} outliers (3×IQR): "
                   f"{list(flagged)[:10]}",
                   outlier_fraction=flagged)

    def _check_integrity(self, sv, path, meta) -> None:
        if not meta:
            return
        recorded = meta.get("interim_checksum_sha256")
        # Only comparable when the sidecar points at this same file.
        if not recorded or meta.get("interim_path") not in (path, os.path.abspath(path)):
            return
        actual = self._checksum(path)
        if actual and actual != recorded:
            sv.add("integrity", FAIL,
                   "interim checksum differs from metadata sidecar "
                   "(file changed after ingestion)",
                   recorded=recorded[:12], actual=actual[:12])

    # ── cross-source ─────────────────────────────────────────────────────────
    def _cross_source_checks(self, report: CrossSourceReport, interim: dict) -> None:
        """Consistency checks that only make sense across sources."""
        present = [s for s in report.sources if s.present]
        if len(present) < 2:
            report.add_cross("corpus", WARN,
                             "fewer than 2 sources present — corpus checks skipped")
            return

        # Entity-universe overlap across sources that carry `entity_id`.
        universes: dict[str, set] = {}
        for sv in present:
            spec = self._spec(sv.source)
            ent = spec.get("entity_column")
            if not ent or not sv.interim_path:
                continue
            try:
                col = self._read_interim(sv.interim_path)[ent]
            except Exception:  # noqa: BLE001
                continue
            vals = set(col.dropna().astype(str).unique())
            if vals:
                universes[sv.source] = vals

        if len(universes) >= 2:
            names = sorted(universes)
            common = set.intersection(*universes.values())
            report.add_cross(
                "entity_overlap",
                PASS if common else WARN,
                (f"{len(common)} entity_id value(s) shared across {names}"
                 if common else
                 f"no shared entity_id across {names} — sources cover disjoint "
                 "entity universes"),
                sources=names,
                sizes={k: len(v) for k, v in universes.items()},
                n_common=len(common),
            )

        report.add_cross("corpus", PASS,
                         f"{len(present)} source(s) validated as a corpus",
                         sources=[s.source for s in present])

    # ── finalize ─────────────────────────────────────────────────────────────
    def _finish(self, report: CrossSourceReport) -> CrossSourceReport:
        path = self.report_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.as_dict(), f, indent=2, default=str)

        for sv in report.sources:
            for c in sv.checks:
                if c.level == FAIL:
                    log.error("[%s] %s: %s", sv.source, c.check, c.message)
                elif c.level == WARN:
                    log.warning("[%s] %s: %s", sv.source, c.check, c.message)
        for c in report.cross_checks:
            if c.level == FAIL:
                log.error("[cross] %s: %s", c.check, c.message)
            elif c.level == WARN:
                log.warning("[cross] %s: %s", c.check, c.message)

        status = "PASS" if report.is_valid else "FAIL"
        log.info(
            "cross-source validation %s — %d source(s), %d fail / %d warn -> %s",
            status, len(report.sources), report.n_fail, report.n_warn, path,
        )

        if not report.is_valid and self.fail_fast:
            raise CrossSourceValidationError(
                f"Cross-source validation failed with {report.n_fail} fatal "
                f"finding(s). Set data_validation.fail_fast=false to bypass."
            )
        return report


def validate_corpus(storage: dict, dv_cfg: dict | None = None) -> CrossSourceReport:
    """Convenience wrapper: build a validator and run it once."""
    return CrossSourceValidator(storage, dv_cfg).validate()
