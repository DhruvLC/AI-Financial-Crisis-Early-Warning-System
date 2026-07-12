"""CLI entry point for the Data Validation module.

Runs the full validation suite over the ingested interim datasets and writes
reports to ``reports/validation/``.

Examples:
    python src/run_validation.py --config configs/ingestion.yaml
    python src/run_validation.py --only fred sec_edgar
    python src/run_validation.py --fail-fast
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

from ingestion.logging_config import configure_logging
from validation.runner import DataValidationRunner


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
    ap = argparse.ArgumentParser(description="Financial EWS data validation")
    ap.add_argument("--config", default="configs/ingestion.yaml")
    ap.add_argument("--only", nargs="*", help="subset of sources to validate")
    ap.add_argument("--fail-fast", action="store_true",
                    help="exit non-zero (and raise) if any dataset has errors")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))

    dv_cfg = dict(cfg.get("data_validation", {}) or {})
    if args.fail_fast:
        dv_cfg["fail_fast"] = True

    runner = DataValidationRunner(_build_storage(cfg), dv_cfg)
    reports = runner.run(only=args.only)

    print("\n─── Data validation summary ───")
    for r in reports:
        if not r.present:
            print(f"  — {r.source:<20} absent")
            continue
        if r.load_error:
            print(f"  ✗ {r.source:<20} unreadable: {r.load_error}")
            continue
        mark = "✓" if r.is_valid else "✗"
        print(f"  {mark} {r.source:<20} score {r.quality_score:5.1f} "
              f"({r.quality_grade})  {r.n_errors} err / {r.n_warnings} warn")

    print(f"\n  reports -> {runner.reports_dir}/")
    failed = [r for r in reports if r.present and not r.is_valid]
    return 1 if (failed and dv_cfg.get("fail_fast")) else 0


if __name__ == "__main__":
    sys.exit(main())
