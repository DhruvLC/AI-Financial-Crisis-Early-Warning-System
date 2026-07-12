"""CLI entry point for the post-ingestion cross-source Data Validation module.

Runs *after* an ingestion pass and validates everything under
``data/interim/`` as a corpus (coverage, schema, sanity ranges, freshness,
integrity, anomalies, cross-source entity consistency). Writes a consolidated
JSON report and prints a per-source summary.

Examples:
    python src/run_data_validation.py --config configs/ingestion.yaml
    python src/run_data_validation.py --config configs/ingestion.yaml --fail-fast
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

from ingestion.cross_validation import CrossSourceValidator
from ingestion.logging_config import configure_logging


def _build_storage(cfg: dict) -> dict:
    """Reconstruct the same storage paths the ingestion runner uses."""
    st = cfg["storage"]
    layer = st.get("metadata_layer", "raw")
    layer_dir = st.get(f"{layer}_dir", st["raw_dir"])
    return {
        "raw_dir": st["raw_dir"],
        "interim_dir": st["interim_dir"],
        "metadata_layer_dir": os.path.join(layer_dir, "_metadata"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Financial EWS post-ingestion cross-source validation"
    )
    ap.add_argument("--config", default="configs/ingestion.yaml")
    ap.add_argument("--fail-fast", action="store_true",
                    help="exit non-zero (and raise) on any fatal finding")
    ap.add_argument("--report", help="override report output path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))

    dv_cfg = dict(cfg.get("data_validation", {}) or {})
    if args.fail_fast:
        dv_cfg["fail_fast"] = True
    if args.report:
        dv_cfg["report_path"] = args.report

    report = CrossSourceValidator(_build_storage(cfg), dv_cfg).validate()

    print("\n─── Cross-source validation summary ───")
    for sv in report.sources:
        mark = {"pass": "✓", "warn": "!", "fail": "✗"}[sv.worst]
        detail = f"{sv.n_rows} rows x {sv.n_cols} cols" if sv.present else "absent"
        print(f"  {mark} {sv.source:<20} {sv.worst:<5} {detail}")
        for c in sv.checks:
            if c.level != "pass":
                print(f"      └─ {c.level.upper()}: {c.message}")
    for c in report.cross_checks:
        if c.level != "pass":
            print(f"  [cross] {c.level.upper()}: {c.message}")

    print(f"\n  {report.n_fail} fatal / {report.n_warn} warning(s) — "
          f"{'PASS' if report.is_valid else 'FAIL'}")
    return 1 if not report.is_valid else 0


if __name__ == "__main__":
    sys.exit(main())
