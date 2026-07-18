"""Transformer orchestrator.

Wires the transformer components into one config-driven flow over the
engineered feature-store splits:

    load + verify features → build tensor loaders → for each enabled model:
        (build → train with early stopping/checkpoints
         → optimize threshold on VAL → evaluate on val/test
         → permutation/SHAP explainability → attention analysis)
    →  build the leaderboard  →  plot figures (incl. attention heatmaps
       and the cross-family comparison)  →  register every model
    →  select + register the best model  →  write the report suite

Design deliberately parallels :class:`pipeline.deep_learning.pipeline.
DLPipeline` (whose permutation/SHAP explainability it reuses directly):
config-driven construction, per-model exception isolation honouring
``fail_fast``, uniform logging, identical metric suite.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ingestion.logging_config import get_logger
from pipeline.deep_learning.pipeline import DLPipeline

from .base import (AttentionSummary, TrainedTransformer, TransformerError,
                   attention_entropy, count_parameters,
                   mean_attention_by_feature, seed_all)
from .data_loader import TransformerData, TransformerDataLoader
from .evaluation import ModelEvaluator, ThresholdOptimizer, predict_proba
from .models import TRANSFORMER_REGISTRY, build_transformer
from .registry import TransformerModelRegistry
from .report import TransformerReport
from .trainer import TransformerTrainer
from .visualization import TransformerVisualizer

__all__ = ["TransformerPipeline", "TransformerResult"]

DEFAULT_MODELS = list(TRANSFORMER_REGISTRY)
RANKING_METRICS = ("roc_auc", "f1", "recall", "pr_auc")


@dataclass
class TransformerResult:
    """Everything a transformer run produces."""

    data: TransformerData
    trained: list[TrainedTransformer] = field(default_factory=list)
    leaderboard: pd.DataFrame | None = None
    best: TrainedTransformer | None = None
    best_entry: dict | None = None
    registry_entries: list[dict] = field(default_factory=list)
    registry_files: dict[str, str] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    family_boards: dict[str, pd.DataFrame] = field(default_factory=dict)


class TransformerPipeline:
    """Run the full train → evaluate → compare → explain → register flow."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.tf_cfg = dict(self.cfg.get("transformers", {}))
        self.log = get_logger("transformers.pipeline")
        self.fail_fast = bool(self.tf_cfg.get("fail_fast", False))
        self.random_state = int(self.tf_cfg.get("random_state", 42))
        self.deterministic = bool(self.tf_cfg.get("deterministic", True))

        self.target_col = self.cfg.get("data", {}).get("target_col",
                                                       "Bankrupt?")
        store_dir = (self.cfg.get("feature_engineering", {})
                     .get("store", {}).get("dir", "data/features"))
        data_cfg = self.tf_cfg.get("data", {})
        self.loader = TransformerDataLoader(
            store_dir, self.target_col,
            batch_size=int(data_cfg.get("batch_size", 64)),
            shuffle=bool(data_cfg.get("shuffle", True)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=self.random_state)

        self.evaluator = ModelEvaluator(
            self.tf_cfg.get("evaluation", {}).get("metrics"))
        self.thresholder = ThresholdOptimizer(
            self.tf_cfg.get("threshold_optimization", {}))
        self.explain_cfg = dict(self.tf_cfg.get("explainability", {}))

        out = self.tf_cfg.get("output", {})
        self.models_dir = out.get("models_dir", "models/transformers")
        self.reports_dir = out.get("reports_dir", "reports/transformers")
        self.figures_dir = out.get("figures_dir",
                                   "reports/transformers/figures")
        self.trainer = TransformerTrainer(self.tf_cfg,
                                          checkpoint_dir=self.models_dir,
                                          evaluator=self.evaluator)
        self.registry = TransformerModelRegistry(self.models_dir)
        self.viz = TransformerVisualizer(
            self.figures_dir,
            dpi=int(self.tf_cfg.get("visualization", {}).get("dpi", 150)))
        self.reporter = TransformerReport(self.reports_dir)

        # borrow the deep-learning permutation/SHAP explainability verbatim
        self._explainer = DLPipeline.__new__(DLPipeline)
        self._explainer.explain_cfg = self.explain_cfg
        self._explainer.random_state = self.random_state
        self._explainer.trainer = self.trainer
        self._explainer.log = self.log

    # ── orchestration ─────────────────────────────────────────────────────────
    def run(self, version: str | None = None) -> TransformerResult:
        """Execute the full transformer flow; return a
        :class:`TransformerResult`."""
        seed_all(self.random_state, self.deterministic)
        data = self.loader.load(version)
        result = TransformerResult(data=data)

        names = self.tf_cfg.get("models") or DEFAULT_MODELS
        for name in names:
            trained = self._train_one(name, data)
            if trained is not None:
                result.trained.append(trained)

        successes = [t for t in result.trained if not t.failed]
        if not successes:
            raise TransformerError("no transformer trained successfully")

        result.leaderboard = self._leaderboard(successes)
        result.family_boards = self._family_boards(result.leaderboard)
        result.figures = self._figures(successes, data, result.leaderboard,
                                       result.family_boards)

        for t in successes:
            entry = self.registry.register(t, data.features, data.version,
                                           data.target_col)
            result.registry_entries.append(entry)

        best_name = result.leaderboard.iloc[0]["model"]
        result.best = next(t for t in successes if t.name == best_name)
        result.best_entry = next(e for e in result.registry_entries
                                 if e["network"] == best_name)
        result.registry_files = self.registry.register_best(
            result.best, result.best_entry, data.features, data.version,
            data.target_col, self.tf_cfg)

        result.reports = self.reporter.write(result)
        return result

    # ── per-model flow ────────────────────────────────────────────────────────
    def _train_one(self, name: str,
                   data: TransformerData) -> TrainedTransformer | None:
        """Train one model; on NaN divergence on an accelerator, retry
        once on CPU (MPS attention kernels can be numerically flaky)."""
        trained = self._train_one_on(name, data, self.trainer)
        if (trained is not None and trained.failed
                and "NaN loss" in (trained.error or "")
                and self.trainer.device.type != "cpu"
                and bool(self.tf_cfg.get("cpu_fallback_on_nan", True))):
            self.log.warning("%s diverged on %s — retrying on CPU", name,
                             self.trainer.device)
            cpu_cfg = dict(self.tf_cfg)
            cpu_cfg["training"] = {**self.tf_cfg.get("training", {}),
                                   "device": "cpu"}
            cpu_trainer = TransformerTrainer(
                cpu_cfg, checkpoint_dir=self.models_dir,
                evaluator=self.evaluator)
            trained = self._train_one_on(name, data, cpu_trainer)
        return trained

    def _train_one_on(self, name: str, data: TransformerData,
                      trainer: TransformerTrainer
                      ) -> TrainedTransformer | None:
        started = time.perf_counter()
        try:
            params = (self.tf_cfg.get("model_params") or {}).get(name)
            model = build_transformer(name, data.n_features, params)
            trained = TrainedTransformer(
                name=name, model=model, architecture=model.architecture(),
                hyperparameters={"data": {"batch_size":
                                          self.loader.batch_size},
                                 "optimizer": self.tf_cfg.get("optimizer",
                                                              {}),
                                 "scheduler": self.tf_cfg.get("scheduler",
                                                              {}),
                                 "loss": self.tf_cfg.get("loss", {}),
                                 "training": self.tf_cfg.get("training",
                                                             {})},
                n_parameters=count_parameters(model),
                device=str(trainer.device))

            trained.history, trained.checkpoints = trainer.fit(
                name, model, data.loaders["train"], data.loaders["val"],
                pos_weight=data.pos_weight,
                resume_from=self.tf_cfg.get("training", {})
                .get("resume_from"))

            # threshold chosen on VAL, applied to every split
            X_val, y_val = data.numpy("val")
            X_test, y_test = data.numpy("test")
            val_proba = predict_proba(model, X_val, trainer.device)
            trained.threshold, trained.threshold_method = (
                self.thresholder.optimize(y_val, val_proba))

            for split, y, proba in (
                    ("val", y_val, val_proba),
                    ("test", y_test, predict_proba(model, X_test,
                                                   trainer.device))):
                trained.evaluations[split] = self.evaluator.evaluate(
                    name, split, y, proba, trained.threshold)

            self._explainer.trainer = trainer
            trained.permutation_importance = self._explainer._permutation(
                model, data)
            trained.shap_summary = self._explainer._shap(model, data)
            trained.attention = self._attention(model, data,
                                                trainer.device)

            trained.train_seconds = time.perf_counter() - started
            self.log.info("finished %s in %.2fs (test roc_auc=%.4f)", name,
                          trained.train_seconds, trained.metric("roc_auc"))
            return trained
        except TransformerError as exc:
            if "unsupported" in str(exc):
                self.log.warning("skipping %s: %s", name, exc)
                if self.fail_fast:
                    raise
                return None
            if self.fail_fast:
                raise
            self.log.error("training %s failed: %s", name, exc)
            return TrainedTransformer(
                name=name, model=None, error=str(exc),
                train_seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - isolate per-model failures
            if self.fail_fast:
                raise
            self.log.error("training %s failed: %s", name, exc)
            return TrainedTransformer(
                name=name, model=None, error=str(exc),
                train_seconds=time.perf_counter() - started)

    # ── attention analysis ────────────────────────────────────────────────────
    @torch.no_grad()
    def _attention(self, model, data: TransformerData,
                   device=None) -> AttentionSummary | None:
        """Capture + aggregate attention on a validation sample."""
        if not self.explain_cfg.get("attention", True):
            return None
        try:
            device = device if device is not None else self.trainer.device
            max_samples = int(self.explain_cfg.get("attention_max_samples",
                                                   256))
            X, _ = data.numpy("val")
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), min(len(X), max_samples),
                             replace=False)
            sample = torch.as_tensor(X[idx]).to(device)

            model.eval()
            model.collect_attention(True)
            model(sample)
            weights = {layer: w.cpu().numpy()
                       for layer, w in model.attention_weights().items()}
            model.collect_attention(False)
            if not weights:
                return None

            # token labels: [CLS]?+features (attention axes are tokens)
            token_labels = model.token_labels()
            labels = [t if t == "[CLS]" else data.features[int(t[1:])]
                      for t in token_labels]
            stacked = np.stack(list(weights.values()))   # (L, B, T, T)
            last_layer = list(weights.values())[-1]
            summary = AttentionSummary(
                feature_names=list(data.features),
                feature_attention=mean_attention_by_feature(
                    stacked, labels, skip_cls=True),
                matrix=last_layer.mean(axis=0),
                matrix_labels=labels,
                entropy={layer: attention_entropy(w)
                         for layer, w in weights.items()},
                n_samples=len(sample))
            return summary
        except Exception as exc:  # noqa: BLE001 - explainability best-effort
            self.log.warning("attention analysis failed: %s", exc)
            return None

    # ── comparison & figures ──────────────────────────────────────────────────
    def _leaderboard(self,
                     trained: list[TrainedTransformer]) -> pd.DataFrame:
        """Rank models by test ROC-AUC, tie-broken by F1/recall/PR-AUC."""
        rows = []
        for t in trained:
            row = {"model": t.name,
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
        self.log.info("leaderboard:\n%s",
                      board[["rank", "model", "roc_auc", "f1",
                             "recall", "pr_auc"]].to_string(index=False))
        return board

    def _family_boards(self,
                       board: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """This run's board + prior ML/DL leaderboards (best-effort)."""
        boards = {"transformers": board}
        cmp_cfg = self.tf_cfg.get("comparison", {})
        for family, path in (
                ("machine_learning",
                 cmp_cfg.get("ml_leaderboard", "reports/ml/leaderboard.csv")),
                ("deep_learning",
                 cmp_cfg.get("dl_leaderboard",
                             "reports/deep_learning/leaderboard.csv"))):
            try:
                if path and os.path.exists(path):
                    boards[family] = pd.read_csv(path)
                else:
                    self.log.warning("no %s leaderboard at %s — skipping "
                                     "comparison", family, path)
            except Exception as exc:  # noqa: BLE001 - comparison best-effort
                self.log.warning("could not read %s leaderboard: %s",
                                 family, exc)
        return boards

    def _figures(self, trained: list[TrainedTransformer],
                 data: TransformerData, leaderboard: pd.DataFrame,
                 family_boards: dict[str, pd.DataFrame]) -> list[str]:
        X_test, y_test = data.numpy("test")
        for t in trained:
            proba = predict_proba(t.model, X_test, self.trainer.device)
            if t.history:
                self.viz.loss_curves(t.name, t.history)
                self.viz.accuracy_curve(t.name, t.history)
                self.viz.lr_curve(t.name, t.history)
            self.viz.roc_curve(t.name, y_test, proba)
            self.viz.pr_curve(t.name, y_test, proba)
            ev = t.evaluations.get("test")
            if ev and ev.confusion:
                self.viz.confusion(t.name, ev.confusion)
            self.viz.calibration(t.name, y_test, proba)
            self.viz.prediction_distribution(t.name, y_test, proba,
                                             t.threshold)
            if t.permutation_importance is not None:
                self.viz.feature_importance(t.name,
                                            t.permutation_importance)
            if t.attention is not None:
                self.viz.attention_heatmap(t.name, t.attention)
                self.viz.attention_by_feature(t.name, t.attention)
        self.viz.comparison_bars(leaderboard, list(RANKING_METRICS))
        self.viz.cross_family_comparison(family_boards)
        return list(self.viz.saved)
