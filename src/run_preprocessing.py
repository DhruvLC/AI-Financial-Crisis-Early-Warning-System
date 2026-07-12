"""CLI entry point for the Data Preparation (preprocessing) module.

Loads the modelling table (stage 2), runs the leak-safe preprocessing pipeline
(clean → de-duplicate → split → impute → outliers → encode → scale), writes the
processed splits to ``data/processed/`` and a JSON+Markdown report to
``reports/preprocessing/``.

Examples:
    python src/run_preprocessing.py --config configs/config.yaml
    python src/run_preprocessing.py --config configs/config.yaml --no-write-processed
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

from ingestion.logging_config import configure_logging, get_logger
from pipeline import data_collection
from pipeline.preprocessing import PreprocessingPipeline, PreprocessingReport


def main() -> int:
    ap = argparse.ArgumentParser(description="Financial EWS data preprocessing")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-write-processed", action="store_true",
                    help="skip writing the processed train/val/test parquet files")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))
    log = get_logger("preprocessing.cli")

    target = cfg["data"]["target_col"]

    # Stage 2 — Data Collection (reuse the existing loader).
    df = data_collection.load(cfg)

    # Stage 3 — Data Preparation (the preprocessing module).
    pipeline = PreprocessingPipeline(cfg, target_col=target)
    result = pipeline.run(df)

    reports_dir = os.path.join(
        cfg.get("output", {}).get("reports_dir", "reports"), "preprocessing")
    reporter = PreprocessingReport(reports_dir)
    json_path, md_path = reporter.write(result, target)

    if not args.no_write_processed:
        processed_dir = cfg["data"]["processed_dir"]
        os.makedirs(processed_dir, exist_ok=True)
        for name, frame in (("train", result.train),
                            ("val", result.val), ("test", result.test)):
            path = os.path.join(processed_dir, f"{name}.parquet")
            try:
                frame.to_parquet(path, index=False)
            except Exception as exc:  # noqa: BLE001 - parquet engine optional
                path = os.path.join(processed_dir, f"{name}.csv")
                frame.to_csv(path, index=False)
                log.warning("parquet unavailable (%s); wrote CSV instead", exc)
            log.info("wrote processed split: %s", path)

    lin = result.lineage
    print("\n─── Preprocessing summary ───")
    print(f"  initial : {lin['initial_shape']['rows']} rows x "
          f"{lin['initial_shape']['cols']} cols")
    print(f"  final   : {lin['final_shape']['rows']} rows x "
          f"{lin['final_shape']['cols']} cols (train)")
    print(f"  splits  : train={len(result.train)} val={len(result.val)} "
          f"test={len(result.test)}")
    print(f"  steps   : {lin['n_applied']} applied / {lin['n_skipped']} skipped")
    for r in lin["trail"]:
        print(f"    {r['order']}. {r['step']:<12} {r['status']:<8} "
              f"rows {r['rows_delta']:+d}  cols {r['cols_delta']:+d}")
    print(f"\n  reports -> {json_path}")
    print(f"             {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
