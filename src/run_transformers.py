"""CLI entry point for the Transformer Models module.

Loads the engineered train/val/test splits from the versioned feature store
(``data/features/``), trains every enabled tabular transformer
(FT-Transformer, TabTransformer, encoder), evaluates each on the val/test
splits, compares them on a ROC-AUC leaderboard (and against the classical
ML and deep-learning leaderboards), saves per-model figures — including
attention heatmaps — and reports under ``reports/transformers/``, and
persists every model to the registry under ``models/transformers/``.

Examples:
    python src/run_transformers.py --config configs/config.yaml
    python src/run_transformers.py --config configs/config.yaml \
        --version v001 --models ft_transformer tabular_encoder
"""
from __future__ import annotations

import argparse
import sys

import yaml

from ingestion.logging_config import configure_logging, get_logger
from pipeline.transformers import TransformerError, TransformerPipeline


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Financial EWS transformer models")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--version", default=None,
                    help="feature-store version to train on "
                         "(default: latest)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="override the enabled model list")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the number of training epochs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))
    log = get_logger("transformers.cli")

    if args.models:
        cfg.setdefault("transformers", {})["models"] = args.models
    if args.epochs:
        (cfg.setdefault("transformers", {})
            .setdefault("training", {}))["epochs"] = args.epochs

    try:
        result = TransformerPipeline(cfg).run(version=args.version)
    except TransformerError as exc:
        log.error("transformer stage failed: %s", exc)
        return 1

    board = result.leaderboard
    print("\n─── Transformer summary ───")
    print(f"  dataset : feature store {result.data.version} "
          f"({result.data.n_features} features)")
    print(f"  trained : {sum(1 for t in result.trained if not t.failed)} "
          f"models ({sum(1 for t in result.trained if t.failed)} failed)")
    print("\n  leaderboard (test):")
    for _, row in board.iterrows():
        print(f"    {int(row['rank']):>2}. {row['model']:<16} "
              f"roc_auc={row['roc_auc']:.4f} f1={row['f1']:.4f} "
              f"recall={row['recall']:.4f} pr_auc={row['pr_auc']:.4f}")
    print(f"\n  best    : {result.best.name} "
          f"(best epoch {result.best.history.best_epoch}, threshold "
          f"{result.best.threshold:.3f} [{result.best.threshold_method}])")
    print(f"  registry: models/transformers/registry.json "
          f"({len(result.registry_entries)} entries)")
    print(f"  figures : {len(result.figures)} saved")
    print(f"  reports : {len(result.reports)} written to "
          f"reports/transformers/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
