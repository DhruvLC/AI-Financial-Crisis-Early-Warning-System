"""Self-Supervised Learning orchestrator.

Wires the SSL components into one config-driven flow over the engineered
feature-store splits:

    load + verify features → build augmentation pipeline → build two-view
    contrastive loaders → for each enabled encoder:
        (build encoder + projection head → contrastive pretraining with
         early stopping/checkpoints → extract + export representations
         → frozen-encoder linear-probe + KNN evaluation)
    →  build the probe leaderboard  →  plot figures (loss curves,
       embedding projections, similarity matrices, distributions)
    →  register every encoder  →  select + register the best encoder
    →  write the report suite

Design deliberately parallels :class:`pipeline.transformers.pipeline.
TransformerPipeline`: config-driven construction, per-encoder exception
isolation honouring ``fail_fast``, uniform logging, the shared metric
suite so probe numbers are comparable with the ML/DL/transformer
leaderboards.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger

from .augmentations import build_augmentations
from .base import SSLError, TrainedEncoder, count_parameters, seed_all
from .data_loader import SSLData, SSLDataLoader
from .encoder import ENCODER_REGISTRY, build_encoder
from .evaluation import (KNNProbe, LinearProbe, ModelEvaluator,
                         ThresholdOptimizer)
from .projection_head import build_projection_head
from .registry import SSLModelRegistry
from .report import SSLReport
from .representation import RepresentationExporter, extract_embeddings
from .trainer import SSLTrainer
from .visualization import SSLVisualizer

__all__ = ["SSLPipeline", "SSLResult"]

log = get_logger("ssl.pipeline")

DEFAULT_ENCODERS = list(ENCODER_REGISTRY)
RANKING_METRICS = ("roc_auc", "f1", "recall", "pr_auc")


@dataclass
class SSLResult:
    """Everything a self-supervised run produces."""

    data: SSLData
    augmentations: list[dict] = field(default_factory=list)
    loss_config: dict = field(default_factory=dict)
    trained: list[TrainedEncoder] = field(default_factory=list)
    leaderboard: pd.DataFrame | None = None
    best: TrainedEncoder | None = None
    best_entry: dict | None = None
    registry_entries: list[dict] = field(default_factory=list)
    registry_files: dict[str, str] = field(default_factory=dict)
    representation_metadata: str | None = None
    figures: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)


class SSLPipeline:
    """Run the full pretrain → extract → probe → register flow."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.ssl_cfg = dict(self.cfg.get("self_supervised", {}))
        self.log = get_logger("ssl.pipeline")
        self.fail_fast = bool(self.ssl_cfg.get("fail_fast", False))
        self.random_state = int(self.ssl_cfg.get("random_state", 42))
        self.deterministic = bool(self.ssl_cfg.get("deterministic", True))

        self.target_col = self.cfg.get("data", {}).get("target_col",
                                                       "Bankrupt?")
        store_dir = (self.cfg.get("feature_engineering", {})
                     .get("store", {}).get("dir", "data/features"))
        data_cfg = self.ssl_cfg.get("data", {})
        self.loader = SSLDataLoader(
            store_dir, self.target_col,
            batch_size=int(data_cfg.get("batch_size", 128)),
            shuffle=bool(data_cfg.get("shuffle", True)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=self.random_state)

        self.augment = build_augmentations(
            self.ssl_cfg.get("augmentations"))

        eval_cfg = self.ssl_cfg.get("evaluation", {})
        self.evaluator = ModelEvaluator(eval_cfg.get("metrics"))
        thresholder = ThresholdOptimizer(
            self.ssl_cfg.get("threshold_optimization", {}))
        self.linear_probe = LinearProbe(
            eval_cfg.get("linear_probe", {}), evaluator=self.evaluator,
            thresholder=thresholder, random_state=self.random_state)
        self.knn_probe = KNNProbe(eval_cfg.get("knn", {}),
                                  evaluator=self.evaluator)

        out = self.ssl_cfg.get("output", {})
        self.models_dir = out.get("models_dir", "models/self_supervised")
        self.reports_dir = out.get("reports_dir",
                                   "reports/self_supervised")
        self.figures_dir = out.get(
            "figures_dir", "reports/self_supervised/figures")
        self.representations_dir = out.get(
            "representations_dir",
            f"{self.models_dir}/representations")

        self.trainer = SSLTrainer(self.ssl_cfg,
                                  checkpoint_dir=self.models_dir)
        self.exporter = RepresentationExporter(self.representations_dir)
        self.registry = SSLModelRegistry(self.models_dir)
        viz_cfg = self.ssl_cfg.get("visualization", {})
        self.viz = SSLVisualizer(self.figures_dir,
                                 dpi=int(viz_cfg.get("dpi", 150)),
                                 random_state=self.random_state)
        self.projection_methods = list(
            viz_cfg.get("projections", ["pca", "tsne", "umap"]))
        self.reporter = SSLReport(self.reports_dir)

    # ── orchestration ─────────────────────────────────────────────────────────
    def run(self, version: str | None = None) -> SSLResult:
        """Execute the full self-supervised flow; return an
        :class:`SSLResult`."""
        seed_all(self.random_state, self.deterministic)
        data = self.loader.load(version)
        result = SSLResult(data=data,
                           augmentations=self.augment.as_dict(),
                           loss_config=dict(self.ssl_cfg.get("loss", {})))

        names = self.ssl_cfg.get("encoders") or DEFAULT_ENCODERS
        for name in names:
            trained = self._train_one(name, data)
            if trained is not None:
                result.trained.append(trained)

        successes = [t for t in result.trained if not t.failed]
        if not successes:
            raise SSLError("no encoder pretrained successfully")

        result.leaderboard = self._leaderboard(successes)
        result.figures = self._figures(successes, data)

        rep_entries = []
        for t in successes:
            entry = self.registry.register(t, data.features, data.version,
                                           data.target_col)
            result.registry_entries.append(entry)
            rep_entries.append({
                "encoder": t.name,
                "model_version": entry["model_version"],
                "embedding_dim": t.architecture.get("embedding_dim"),
                "paths": t.representations,
                "embedding_stats": t.embedding_stats})
        result.representation_metadata = self.exporter.write_metadata(
            rep_entries, data.version)

        best_name = result.leaderboard.iloc[0]["model"]
        result.best = next(t for t in successes if t.name == best_name)
        result.best_entry = next(e for e in result.registry_entries
                                 if e["network"] == best_name)
        result.registry_files = self.registry.register_best(
            result.best, result.best_entry, data.features, data.version,
            data.target_col, self.ssl_cfg)

        result.reports = self.reporter.write(result)
        return result

    # ── per-encoder flow ──────────────────────────────────────────────────────
    def _train_one(self, name: str,
                   data: SSLData) -> TrainedEncoder | None:
        started = time.perf_counter()
        try:
            params = (self.ssl_cfg.get("encoder_params") or {}).get(name)
            encoder = build_encoder(name, data.n_features, params)
            head = build_projection_head(
                encoder.embedding_dim,
                self.ssl_cfg.get("projection_head", {}))
            trained = TrainedEncoder(
                name=name, model=encoder, projection_head=head,
                architecture=encoder.architecture(),
                hyperparameters={
                    "data": {"batch_size": self.loader.batch_size},
                    "projection_head": self.ssl_cfg.get("projection_head",
                                                        {}),
                    "loss": self.ssl_cfg.get("loss", {}),
                    "augmentations": self.augment.as_dict(),
                    "optimizer": self.ssl_cfg.get("optimizer", {}),
                    "scheduler": self.ssl_cfg.get("scheduler", {}),
                    "training": self.ssl_cfg.get("training", {})},
                n_parameters=count_parameters(encoder),
                device=str(self.trainer.device))

            loaders = self.loader.contrastive_loaders(data, self.augment)
            trained.history, trained.checkpoints = self.trainer.fit(
                name, encoder, head, loaders["train"], loaders.get("val"),
                resume_from=self.ssl_cfg.get("training", {})
                .get("resume_from"))

            # representations (validated + exported)
            trained.representations, trained.embedding_stats = (
                self.exporter.export(name, encoder, data,
                                     self.trainer.device))
            self.log.info("%s embedding stats (train): %s", name,
                          trained.embedding_stats.get("train"))

            # frozen-encoder probes
            Z = {s: extract_embeddings(encoder, data.numpy(s)[0],
                                       self.trainer.device)
                 for s in data.tensors}
            y = {s: data.numpy(s)[1] for s in data.tensors}
            (trained.evaluations, trained.threshold,
             trained.threshold_method) = self.linear_probe.evaluate(
                name, Z, y)
            trained.knn_evaluations = self.knn_probe.evaluate(name, Z, y)

            trained.train_seconds = time.perf_counter() - started
            self.log.info("finished %s in %.2fs (probe test roc_auc=%.4f)",
                          name, trained.train_seconds,
                          trained.metric("roc_auc"))
            return trained
        except SSLError as exc:
            if "unsupported" in str(exc):
                self.log.warning("skipping %s: %s", name, exc)
                if self.fail_fast:
                    raise
                return None
            if self.fail_fast:
                raise
            self.log.error("pretraining %s failed: %s", name, exc)
            return TrainedEncoder(
                name=name, model=None, error=str(exc),
                train_seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - isolate per-encoder failures
            if self.fail_fast:
                raise
            self.log.error("pretraining %s failed: %s", name, exc)
            return TrainedEncoder(
                name=name, model=None, error=str(exc),
                train_seconds=time.perf_counter() - started)

    # ── comparison & figures ──────────────────────────────────────────────────
    def _leaderboard(self, trained: list[TrainedEncoder]) -> pd.DataFrame:
        """Rank encoders by linear-probe test ROC-AUC."""
        rows = []
        for t in trained:
            row = {"model": t.name,
                   "embedding_dim": t.architecture.get("embedding_dim"),
                   "threshold": t.threshold,
                   "n_parameters": t.n_parameters,
                   "best_epoch": (t.history.best_epoch if t.history
                                  else None),
                   "train_seconds": round(t.train_seconds, 2)}
            for m in ("roc_auc", "f1", "recall", "precision", "pr_auc",
                      "accuracy", "balanced_accuracy", "mcc"):
                row[m] = t.metric(m, "test")
            rows.append(row)
        board = (pd.DataFrame(rows)
                 .sort_values(list(RANKING_METRICS), ascending=False,
                              na_position="last")
                 .reset_index(drop=True))
        board.insert(0, "rank", np.arange(1, len(board) + 1))
        self.log.info("linear-probe leaderboard:\n%s",
                      board[["rank", "model", "roc_auc", "f1", "recall",
                             "pr_auc"]].to_string(index=False))
        return board

    def _figures(self, trained: list[TrainedEncoder],
                 data: SSLData) -> list[str]:
        viz_cfg = self.ssl_cfg.get("visualization", {})
        max_proj = int(viz_cfg.get("projection_max_samples", 2000))
        max_sim = int(viz_cfg.get("similarity_max_samples", 200))
        X_test, y_test = data.numpy("test")
        for t in trained:
            if t.history:
                self.viz.loss_curves(t.name, t.history)
                self.viz.lr_curve(t.name, t.history)
            Z = extract_embeddings(t.model, X_test, self.trainer.device)
            for method in self.projection_methods:
                self.viz.embedding_projection(t.name, Z, y_test, method,
                                              max_samples=max_proj)
            self.viz.similarity_matrix(t.name, Z, y_test,
                                       max_samples=max_sim)
            self.viz.embedding_distribution(t.name, Z)
        if len(trained) > 1:
            board = self._leaderboard(trained)
            self.viz.comparison_bars(board, list(RANKING_METRICS))
        return list(self.viz.saved)
