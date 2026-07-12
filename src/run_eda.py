"""CLI entry point for the Exploratory Data Analysis (EDA) module.

Loads a processed split (default ``train``) from ``data/processed/``, runs the
full EDA analyzer suite, and writes figures + JSON/Markdown/HTML/CSV reports to
``reports/eda/``.

Examples:
    python src/run_eda.py --config configs/config.yaml
    python src/run_eda.py --config configs/config.yaml --dataset val
    python src/run_eda.py --no-figures
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

from ingestion.logging_config import configure_logging, get_logger
from pipeline.eda import EdaRunner


def _load_split(processed_dir: str, name: str) -> pd.DataFrame:
    """Load ``<name>`` from the processed dir, preferring parquet then CSV."""
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = os.path.join(processed_dir, f"{name}{ext}")
        if os.path.exists(path):
            return reader(path)
    raise FileNotFoundError(
        f"no processed split '{name}' (.parquet/.csv) in {processed_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Financial EWS exploratory analysis")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--dataset", default=None,
                    help="processed split to analyze (train|val|test)")
    ap.add_argument("--no-figures", action="store_true",
                    help="disable figure generation")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))
    log = get_logger("eda.cli")

    eda_cfg = cfg.setdefault("eda", {})
    if args.no_figures:
        eda_cfg.setdefault("figures", {})["enabled"] = False

    target = cfg["data"]["target_col"]
    dataset = args.dataset or eda_cfg.get("dataset", "train")
    processed_dir = cfg["data"]["processed_dir"]

    df = _load_split(processed_dir, dataset)
    log.info("loaded processed split '%s': %d rows x %d cols",
             dataset, len(df), df.shape[1])

    runner = EdaRunner(cfg, target_col=target)
    result = runner.run(df, dataset_name=dataset)

    ins = result.insights
    print("\n─── EDA summary ───")
    print(f"  dataset  : {dataset} ({len(df)} rows x {df.shape[1]} cols)")
    print(f"  analyzers: {result.report['meta']['n_analyzers_run']} run / "
          f"{result.report['meta']['n_analyzers_skipped']} skipped")
    print(f"  figures  : {len(result.report['figures'])}")
    print(f"  insights : {ins['n_insights']}  {ins['severity_counts']}")
    print(f"  headline : {ins['headline']}")
    print("\n  reports ->")
    for fmt in ("json", "md", "html"):
        print(f"    {result.outputs[fmt]}")
    print(f"    {len(result.outputs['csv'])} CSV(s) under "
          f"{os.path.join(runner.reports_dir, 'statistics')}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
