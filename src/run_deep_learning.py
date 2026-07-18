"""CLI entry point for the Deep Learning module.

Loads the engineered train/val/test splits from the versioned feature store
(``data/features/``), trains every enabled network, evaluates each on the
val/test splits, compares them on a ROC-AUC leaderboard, saves per-model
figures and reports under ``reports/deep_learning/``, and persists every
network to the registry under ``models/deep_learning/``.

Examples:
    python src/run_deep_learning.py --config configs/config.yaml
    python src/run_deep_learning.py --config configs/config.yaml \
        --version v001 --networks mlp residual
"""
from __future__ import annotations

import argparse
import sys

import yaml

from ingestion.logging_config import configure_logging, get_logger
from pipeline.deep_learning import DLError, DLPipeline


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Financial EWS deep learning")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--version", default=None,
                    help="feature-store version to train on "
                         "(default: latest)")
    ap.add_argument("--networks", nargs="*", default=None,
                    help="override the enabled network list")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the number of training epochs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))
    log = get_logger("dl.cli")

    if args.networks:
        cfg.setdefault("deep_learning", {})["networks"] = args.networks
    if args.epochs:
        (cfg.setdefault("deep_learning", {})
            .setdefault("training", {}))["epochs"] = args.epochs

    try:
        result = DLPipeline(cfg).run(version=args.version)
    except DLError as exc:
        log.error("deep learning failed: %s", exc)
        return 1

    board = result.leaderboard
    print("\n─── Deep learning summary ───")
    print(f"  dataset : feature store {result.data.version} "
          f"({result.data.n_features} features)")
    print(f"  trained : {sum(1 for t in result.trained if not t.failed)} "
          f"networks ({sum(1 for t in result.trained if t.failed)} failed)")
    print("\n  leaderboard (test):")
    for _, row in board.iterrows():
        print(f"    {int(row['rank']):>2}. {row['model']:<12} "
              f"roc_auc={row['roc_auc']:.4f} f1={row['f1']:.4f} "
              f"recall={row['recall']:.4f} pr_auc={row['pr_auc']:.4f}")
    print(f"\n  best    : {result.best.name} "
          f"(best epoch {result.best.history.best_epoch}, threshold "
          f"{result.best.threshold:.3f} [{result.best.threshold_method}])")
    print(f"  registry: models/deep_learning/registry.json "
          f"({len(result.registry_entries)} entries)")
    print(f"  figures : {len(result.figures)} saved")
    print(f"  reports : {len(result.reports)} written to "
          f"reports/deep_learning/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
