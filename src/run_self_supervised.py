"""CLI entry point for the Self-Supervised Learning module.

Loads the engineered train/val/test splits from the versioned feature
store (``data/features/``), pretrains every enabled encoder (MLP,
Residual, Transformer) with contrastive learning over configurable
tabular augmentations, exports latent representations for downstream
modules, evaluates each encoder with a frozen linear probe (and optional
KNN), saves per-encoder figures — including embedding projections and
similarity matrices — and reports under ``reports/self_supervised/``, and
persists every encoder to the registry under ``models/self_supervised/``.

Examples:
    python src/run_self_supervised.py --config configs/config.yaml
    python src/run_self_supervised.py --config configs/config.yaml \
        --version v001 --encoders mlp transformer --epochs 20
"""
from __future__ import annotations

import argparse
import sys

import yaml

from ingestion.logging_config import configure_logging, get_logger
from pipeline.self_supervised import SSLError, SSLPipeline


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Financial EWS self-supervised representation "
                    "learning")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--version", default=None,
                    help="feature-store version to pretrain on "
                         "(default: latest)")
    ap.add_argument("--encoders", nargs="*", default=None,
                    help="override the enabled encoder list")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the number of pretraining epochs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    configure_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))
    log = get_logger("ssl.cli")

    if args.encoders:
        cfg.setdefault("self_supervised", {})["encoders"] = args.encoders
    if args.epochs:
        (cfg.setdefault("self_supervised", {})
            .setdefault("training", {}))["epochs"] = args.epochs

    try:
        result = SSLPipeline(cfg).run(version=args.version)
    except SSLError as exc:
        log.error("self-supervised stage failed: %s", exc)
        return 1

    board = result.leaderboard
    print("\n─── Self-supervised summary ───")
    print(f"  dataset : feature store {result.data.version} "
          f"({result.data.n_features} features)")
    print(f"  trained : {sum(1 for t in result.trained if not t.failed)} "
          f"encoders ({sum(1 for t in result.trained if t.failed)} "
          f"failed)")
    print("\n  linear-probe leaderboard (test):")
    for _, row in board.iterrows():
        print(f"    {int(row['rank']):>2}. {row['model']:<12} "
              f"roc_auc={row['roc_auc']:.4f} f1={row['f1']:.4f} "
              f"recall={row['recall']:.4f} pr_auc={row['pr_auc']:.4f}")
    print(f"\n  best    : {result.best.name} "
          f"(best epoch {result.best.history.best_epoch}, "
          f"embedding_dim {result.best.architecture['embedding_dim']})")
    print(f"  registry: {result.registry_files.get('best_encoder.pt')}")
    print(f"  latents : {result.representation_metadata}")
    print(f"  figures : {len(result.figures)} saved")
    print(f"  reports : {len(result.reports)} written to "
          f"reports/self_supervised/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
