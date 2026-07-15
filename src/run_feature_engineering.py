"""CLI entry point for the Feature Engineering module.

Loads the processed train/val/test splits from ``data/processed/``, runs the
leak-safe feature-engineering pipeline (generate → drop multicollinear →
select → reduce → score importance), persists the engineered splits to the
versioned feature store under ``data/features/``, and writes JSON + Markdown +
CSV reports to ``reports/feature_engineering/``.

Examples:
    python src/run_feature_engineering.py --config configs/config.yaml
    python src/run_feature_engineering.py --config configs/config.yaml --no-store
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

from ingestion.logging_config import configure_logging, get_logger
from pipeline.feature_engineering import (
    FeatureEngineeringPipeline, FeatureEngineeringReport)


def _load_split(processed_dir: str, name: str) -> pd.DataFrame:
    """Load ``<name>`` from the processed dir, preferring parquet then CSV."""
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = os.path.join(processed_dir, f"{name}{ext}")
        if os.path.exists(path):
            return reader(path)
    raise FileNotFoundError(
        f"no processed split '{name}' (.parquet/.csv) in {processed_dir} — "
        "run src/run_preprocessing.py first")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Financial EWS feature engineering")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-store", action="store_true",
                    help="skip persisting the engineered splits to the "
                         "feature store")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))
    log = get_logger("features.cli")

    if args.no_store:
        cfg.setdefault("feature_engineering", {}).setdefault(
            "store", {})["enabled"] = False

    target = cfg["data"]["target_col"]
    processed_dir = cfg["data"]["processed_dir"]
    train, val, test = (_load_split(processed_dir, n)
                        for n in ("train", "val", "test"))

    pipeline = FeatureEngineeringPipeline(cfg, target_col=target)
    result = pipeline.run(train, val, test)

    reports_dir = os.path.join(
        cfg.get("output", {}).get("reports_dir", "reports"),
        "feature_engineering")
    reporter = FeatureEngineeringReport(reports_dir)
    json_path, md_path = reporter.write(result, target)

    lin = result.lineage
    print("\n─── Feature engineering summary ───")
    print(f"  initial : {lin['initial_shape']['rows']} rows x "
          f"{lin['initial_shape']['cols']} cols (train)")
    print(f"  final   : {lin['final_shape']['rows']} rows x "
          f"{lin['final_shape']['cols']} cols (train)")
    print(f"  splits  : train={len(result.train)} val={len(result.val)} "
          f"test={len(result.test)}")
    print(f"  features: +{lin['n_generated_total']} generated / "
          f"-{lin['n_removed_total']} removed")
    print(f"  steps   : {lin['n_applied']} applied / "
          f"{lin['n_skipped']} skipped")
    for r in lin["trail"]:
        print(f"    {r['order']}. {r['step']:<18} {r['status']:<8} "
              f"cols {r['cols_delta']:+d}")
    if result.store_record:
        print(f"\n  store   -> {result.store_record['version']} "
              f"({result.store_record['files']['train']})")
    print(f"  reports -> {json_path}")
    print(f"             {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
