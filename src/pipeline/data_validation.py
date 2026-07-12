"""Stage 4/5 — Data Validation & Quality.

Runs on the *collected* modelling table (after stage 2, before stage 3) to
catch problems that would silently corrupt training: a missing/degenerate
target, a non-binary label, excessive missingness, constant or duplicated
features, infinities, and severe class imbalance.

This is the pipeline-level gate. It is complementary to the per-source
ingestion validator in ``src/ingestion/validation.py`` (which vets each raw
feed as it lands); here we vet the single assembled table the models see.

Behaviour is configurable via the ``validation`` block in the config:

    validation:
      enabled: true
      max_missing_pct: 0.5      # per-column null fraction -> error above this
      min_rows: 50              # too few rows to train -> error
      min_minority_pct: 0.01    # rarer class below this -> warning
      fail_fast: true           # raise on any error (else just report)
      report_path: "reports/data_validation.json"

A JSON report is always written; ``fail_fast`` only controls whether errors
abort the run.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class DataValidationError(RuntimeError):
    """Raised when validation finds a fatal problem and fail_fast is on."""


@dataclass
class QualityReport:
    """Structured result of validating the modelling table."""
    n_rows: int
    n_cols: int
    target_col: str
    n_positive: int = 0
    n_negative: int = 0
    minority_pct: float = 0.0
    missing_by_col: dict = field(default_factory=dict)
    missing_total_pct: float = 0.0
    constant_cols: list = field(default_factory=list)
    non_numeric_cols: list = field(default_factory=list)
    inf_cols: list = field(default_factory=list)
    duplicate_rows: int = 0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target_col": self.target_col,
            "class_balance": {
                "positive": self.n_positive,
                "negative": self.n_negative,
                "minority_pct": round(self.minority_pct, 4),
            },
            "missing_total_pct": round(self.missing_total_pct, 4),
            "missing_by_col": {k: int(v) for k, v in self.missing_by_col.items() if v},
            "constant_cols": self.constant_cols,
            "non_numeric_cols": self.non_numeric_cols,
            "inf_cols": self.inf_cols,
            "duplicate_rows": self.duplicate_rows,
            "warnings": self.warnings,
            "errors": self.errors,
            "is_valid": self.is_valid,
        }


def _defaults(cfg: dict) -> dict:
    v = dict(cfg.get("validation") or {})
    v.setdefault("enabled", True)
    v.setdefault("max_missing_pct", 0.5)
    v.setdefault("min_rows", 50)
    v.setdefault("min_minority_pct", 0.01)
    v.setdefault("fail_fast", True)
    v.setdefault("report_path", os.path.join(
        cfg.get("output", {}).get("reports_dir", "reports"),
        "data_validation.json",
    ))
    return v


def validate(df: pd.DataFrame, cfg: dict) -> QualityReport:
    """Validate the assembled modelling table. Returns a QualityReport.

    Writes a JSON report and — when ``validation.fail_fast`` is true — raises
    :class:`DataValidationError` if any fatal problem is found.
    """
    vcfg = _defaults(cfg)
    target = cfg["data"]["target_col"]
    report = QualityReport(n_rows=len(df), n_cols=df.shape[1], target_col=target)

    if not vcfg["enabled"]:
        print("[validate] Disabled via config — skipping.")
        return report

    # ── Fatal: empty / too small ───────────────────────────────────────────
    if df.empty:
        report.errors.append("dataset is empty")
        return _finish(report, vcfg)
    if len(df) < vcfg["min_rows"]:
        report.errors.append(
            f"only {len(df)} rows (< min_rows={vcfg['min_rows']})"
        )

    # ── Fatal: target presence + sanity ────────────────────────────────────
    if target not in df.columns:
        report.errors.append(
            f"target_col '{target}' not in columns "
            f"(first 10: {list(df.columns)[:10]})"
        )
        return _finish(report, vcfg)  # nothing else is meaningful without a target

    y = df[target]
    if y.isna().any():
        report.errors.append(f"target '{target}' has {int(y.isna().sum())} missing values")

    non_null = y.dropna()
    classes = sorted(non_null.unique().tolist())
    if len(classes) < 2:
        report.errors.append(f"target has < 2 classes: {classes}")
    elif len(classes) > 2 or not set(classes).issubset({0, 1, 0.0, 1.0, True, False}):
        report.warnings.append(
            f"target is not clean binary 0/1 (values: {classes[:10]})"
        )

    # Class balance (only defined for a usable binary target)
    if len(classes) == 2:
        counts = non_null.astype(float).value_counts()
        report.n_positive = int(counts.get(1.0, 0))
        report.n_negative = int(counts.get(0.0, non_null.shape[0] - report.n_positive))
        minority = min(report.n_positive, report.n_negative)
        report.minority_pct = minority / max(len(non_null), 1)
        if report.minority_pct < vcfg["min_minority_pct"]:
            report.warnings.append(
                f"severe class imbalance: minority class is "
                f"{report.minority_pct:.2%} (< {vcfg['min_minority_pct']:.0%})"
            )

    features = [c for c in df.columns if c != target]

    # ── Missing values (per column, error if any column too sparse) ─────────
    miss = df.isna().sum()
    report.missing_by_col = miss.to_dict()
    total_cells = max(df.shape[0] * df.shape[1], 1)
    report.missing_total_pct = float(miss.sum()) / total_cells
    sparse = {c: miss[c] / len(df) for c in df.columns if miss[c] / len(df) > vcfg["max_missing_pct"]}
    if sparse:
        report.errors.append(
            f"{len(sparse)} column(s) exceed max_missing_pct="
            f"{vcfg['max_missing_pct']:.0%}: "
            f"{ {c: round(p, 3) for c, p in list(sparse.items())[:10]} }"
        )
    elif report.missing_total_pct > 0:
        report.warnings.append(f"{report.missing_total_pct:.2%} of cells missing")

    # ── Non-numeric feature columns (models here expect numeric) ────────────
    non_numeric = [c for c in features
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        report.non_numeric_cols = non_numeric
        report.warnings.append(
            f"{len(non_numeric)} non-numeric feature column(s) will be dropped "
            f"downstream: {non_numeric[:10]}"
        )

    numeric = df[features].select_dtypes(include=[np.number])

    # ── Constant / zero-variance features ──────────────────────────────────
    if not numeric.empty:
        nunique = numeric.nunique(dropna=True)
        report.constant_cols = nunique[nunique <= 1].index.tolist()
        if report.constant_cols:
            report.warnings.append(
                f"{len(report.constant_cols)} constant feature(s) "
                f"(will be removed by variance filter): "
                f"{report.constant_cols[:10]}"
            )

        # ── Infinities (break scaling / tree splits silently) ──────────────
        inf_mask = np.isinf(numeric.to_numpy())
        if inf_mask.any():
            report.inf_cols = numeric.columns[inf_mask.any(axis=0)].tolist()
            report.errors.append(
                f"{len(report.inf_cols)} column(s) contain inf/-inf: "
                f"{report.inf_cols[:10]}"
            )

    # ── Duplicate rows (soft — prep drops them) ─────────────────────────────
    report.duplicate_rows = int(df.duplicated().sum())
    if report.duplicate_rows:
        report.warnings.append(
            f"{report.duplicate_rows} duplicate rows (prep will drop them)"
        )

    return _finish(report, vcfg)


def _finish(report: QualityReport, vcfg: dict) -> QualityReport:
    """Write the JSON report, log a summary, and enforce fail_fast."""
    path = vcfg["report_path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.as_dict(), f, indent=2)

    for w in report.warnings:
        print(f"[validate] WARN: {w}")
    for e in report.errors:
        print(f"[validate] ERROR: {e}")
    status = "PASS" if report.is_valid else "FAIL"
    print(
        f"[validate] {status} — {report.n_rows} rows x {report.n_cols} cols, "
        f"{len(report.warnings)} warning(s), {len(report.errors)} error(s) "
        f"-> {path}"
    )

    if report.errors and vcfg["fail_fast"]:
        raise DataValidationError(
            f"Data validation failed with {len(report.errors)} error(s): "
            f"{report.errors}. Set validation.fail_fast=false to bypass."
        )
    return report
