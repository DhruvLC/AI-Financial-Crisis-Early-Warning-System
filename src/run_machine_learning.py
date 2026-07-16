"""CLI entry point for the Machine Learning module.

Loads the engineered train/val/test splits from the versioned feature store
(``data/features/``), trains every enabled algorithm, evaluates each on the
val/test splits, compares them on a ROC-AUC leaderboard, saves per-model
figures and reports under ``reports/ml/``, persists every model to the
registry under ``models/``, and registers the best model separately.

Examples:
    python src/run_machine_learning.py --config configs/config.yaml
    python src/run_machine_learning.py --config configs/config.yaml \
        --version v001 --algorithms logistic_regression random_forest
"""
from __future__ import annotations

import argparse
import sys

import yaml

from ingestion.logging_config import configure_logging, get_logger
from pipeline.ml import MLError, MLPipeline


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Financial EWS machine learning")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--version", default=None,
                    help="feature-store version to train on "
                         "(default: latest)")
    ap.add_argument("--algorithms", nargs="*", default=None,
                    help="override the enabled algorithm list")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))
    log = get_logger("ml.cli")

    if args.algorithms:
        cfg.setdefault("ml", {})["algorithms"] = args.algorithms

    try:
        result = MLPipeline(cfg).run(version=args.version)
    except MLError as exc:
        log.error("machine learning failed: %s", exc)
        return 1

    board = result.leaderboard
    print("\n─── Machine learning summary ───")
    print(f"  dataset : feature store {result.dataset.version} "
          f"({len(result.dataset.features)} features)")
    print(f"  trained : {sum(1 for t in result.trained if not t.failed)} "
          f"models ({sum(1 for t in result.trained if t.failed)} failed)")
    print("\n  leaderboard (test):")
    for _, row in board.iterrows():
        print(f"    {int(row['rank']):>2}. {row['model']:<22} "
              f"roc_auc={row['roc_auc']:.4f} f1={row['f1']:.4f} "
              f"recall={row['recall']:.4f} pr_auc={row['pr_auc']:.4f}")
    print(f"\n  best    : {result.best.name} "
          f"(threshold {result.best.threshold:.3f} "
          f"[{result.best.threshold_method}])")
    print(f"  registry: models/registry.json "
          f"({len(result.registry_entries)} entries)")
    print(f"  figures : {len(result.figures)} saved")
    print(f"  reports : {len(result.reports)} written to reports/ml/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
