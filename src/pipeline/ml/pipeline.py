"""Machine-learning orchestrator.

Wires the ML components into one config-driven flow over the engineered
feature-store splits:

    load + verify features  →  for each enabled algorithm:
        (tune → cross-validate → train → optimize threshold on VAL
         → evaluate on val/test → explain)
    →  build the leaderboard  →  plot figures  →  register every model
    →  select + register the best model  →  write the report suite

Design deliberately parallels
:class:`pipeline.feature_engineering.pipeline.FeatureEngineeringPipeline`:
config-driven construction, per-model exception isolation honouring
``fail_fast``, uniform logging.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger

from .base import MLError, TrainedModel
from .data_loader import MLDataLoader, MLDataset
from .evaluation import ModelEvaluator, ThresholdOptimizer
from .explain import ModelExplainer
from .models import MODEL_REGISTRY, build_model
from .registry import ModelRegistry
from .report import MLReport
from .tuning import HyperparameterTuner
from .visualization import MLVisualizer

DEFAULT_ALGORITHMS = list(MODEL_REGISTRY)
RANKING_METRICS = ("roc_auc", "f1", "recall", "pr_auc")


@dataclass
class MLResult:
    """Everything an ML run produces."""

    dataset: MLDataset
    trained: list[TrainedModel] = field(default_factory=list)
    leaderboard: pd.DataFrame | None = None
    best: TrainedModel | None = None
    best_entry: dict | None = None
    registry_entries: list[dict] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)


class MLPipeline:
    """Run the full train → evaluate → compare → explain → register flow."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.ml_cfg = dict(self.cfg.get("ml", {}))
        self.log = get_logger("ml.pipeline")
        self.fail_fast = bool(self.ml_cfg.get("fail_fast", False))
        self.random_state = int(self.ml_cfg.get("random_state", 42))

        self.target_col = self.cfg.get("data", {}).get("target_col",
                                                       "Bankrupt?")
        store_dir = (self.cfg.get("feature_engineering", {})
                     .get("store", {}).get("dir", "data/features"))
        self.loader = MLDataLoader(store_dir, self.target_col)
        self.tuner = HyperparameterTuner(
            self.ml_cfg.get("tuning", {}),
            self.ml_cfg.get("cross_validation", {}), self.random_state)
        self.evaluator = ModelEvaluator(self.ml_cfg.get("metrics"))
        self.thresholder = ThresholdOptimizer(
            self.ml_cfg.get("threshold_optimization", {}))
        self.explainer = ModelExplainer(
            self.ml_cfg.get("explainability", {}), self.random_state)

        out = self.ml_cfg.get("output", {})
        self.models_dir = out.get(
            "models_dir",
            self.cfg.get("output", {}).get("models_dir", "models"))
        self.reports_dir = out.get("reports_dir", "reports/ml")
        self.figures_dir = out.get("figures_dir", "reports/ml/figures")
        self.registry = ModelRegistry(self.models_dir)
        self.viz = MLVisualizer(self.figures_dir)
        self.reporter = MLReport(self.reports_dir)

    # ── orchestration ─────────────────────────────────────────────────────────
    def run(self, version: str | None = None) -> MLResult:
        """Execute the full ML flow; return an :class:`MLResult`."""
        dataset = self.loader.load(version)
        result = MLResult(dataset=dataset)

        algorithms = self.ml_cfg.get("algorithms") or DEFAULT_ALGORITHMS
        for name in algorithms:
            trained = self._train_one(name, dataset)
            if trained is not None:
                result.trained.append(trained)

        successes = [t for t in result.trained if not t.failed]
        if not successes:
            raise MLError("no model trained successfully")

        result.leaderboard = self._leaderboard(successes)
        result.figures = self._figures(successes, dataset,
                                       result.leaderboard)

        for t in successes:
            entry = self.registry.register(t, dataset.features,
                                           dataset.version,
                                           dataset.target_col)
            result.registry_entries.append(entry)

        best_name = result.leaderboard.iloc[0]["model"]
        result.best = next(t for t in successes if t.name == best_name)
        result.best_entry = next(e for e in result.registry_entries
                                 if e["algorithm"] == best_name)
        self.registry.register_best(result.best, result.best_entry)

        result.reports = self.reporter.write(result)
        return result

    # ── per-model flow ────────────────────────────────────────────────────────
    def _train_one(self, name: str,
                   dataset: MLDataset) -> TrainedModel | None:
        X_train, y_train = dataset.xy("train")
        X_val, y_val = dataset.xy("val")
        X_test, y_test = dataset.xy("test")
        started = time.perf_counter()
        try:
            model = build_model(name, self._model_params(name, y_train),
                                self.random_state)
            trained = TrainedModel(name=name, model=model)

            trained.tuning = self.tuner.tune(model, X_train, y_train)
            if self.ml_cfg.get("cross_validation", {}).get("enabled", True):
                trained.cv_scores = self.tuner.cross_validate(
                    model, X_train, y_train)

            self.log.info("training %s ...", name)
            model.fit(X_train, y_train)
            trained.hyperparameters = model.get_params()

            # threshold chosen on VAL, applied to every split
            val_proba = model.predict_proba(X_val)
            trained.threshold, trained.threshold_method = (
                self.thresholder.optimize(y_val, val_proba))

            for split, X, y, proba in (
                    ("val", X_val, y_val, val_proba),
                    ("test", X_test, y_test, model.predict_proba(X_test))):
                trained.evaluations[split] = self.evaluator.evaluate(
                    name, split, y, proba, trained.threshold)

            explanations = self.explainer.explain(model, X_val, y_val)
            trained.feature_importance = explanations["native"]
            trained.permutation_importance = explanations["permutation"]
            trained.shap_summary = explanations["shap"]

            trained.train_seconds = time.perf_counter() - started
            self.log.info("finished %s in %.2fs (test roc_auc=%.4f)", name,
                          trained.train_seconds,
                          trained.metric("roc_auc"))
            return trained
        except MLError as exc:
            if "unavailable" in str(exc) or "unsupported" in str(exc):
                self.log.warning("skipping %s: %s", name, exc)
                if self.fail_fast and "unsupported" in str(exc):
                    raise
                return None
            if self.fail_fast:
                raise
            self.log.error("training %s failed: %s", name, exc)
            return TrainedModel(name=name, model=None, error=str(exc),
                                train_seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - isolate per-model failures
            if self.fail_fast:
                raise
            self.log.error("training %s failed: %s", name, exc)
            return TrainedModel(name=name, model=None, error=str(exc),
                                train_seconds=time.perf_counter() - started)

    def _model_params(self, name: str, y_train: pd.Series) -> dict:
        """Config overrides + runtime class-imbalance weight for XGBoost."""
        params = dict((self.ml_cfg.get("model_params") or {}).get(name)
                      or {})
        if name == "xgboost" and "scale_pos_weight" not in params:
            pos = int((y_train == 1).sum())
            if pos:
                params["scale_pos_weight"] = float((y_train == 0).sum()
                                                   / pos)
        return params

    # ── comparison & figures ──────────────────────────────────────────────────
    def _leaderboard(self, trained: list[TrainedModel]) -> pd.DataFrame:
        """Rank models by test ROC-AUC, tie-broken by F1/recall/PR-AUC."""
        rows = []
        for t in trained:
            row = {"model": t.name,
                   "threshold": t.threshold,
                   "cv_mean": t.cv_scores.get("mean"),
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

    def _figures(self, trained: list[TrainedModel], dataset: MLDataset,
                 leaderboard: pd.DataFrame) -> list[str]:
        X_test, y_test = dataset.xy("test")
        y = np.asarray(y_test)
        for t in trained:
            proba = t.model.predict_proba(X_test)
            self.viz.roc_curve(t.name, y, proba)
            self.viz.pr_curve(t.name, y, proba)
            ev = t.evaluations.get("test")
            if ev and ev.confusion:
                self.viz.confusion(t.name, ev.confusion)
            self.viz.calibration(t.name, y, proba)
            self.viz.lift_chart(t.name, y, proba)
            self.viz.gain_chart(t.name, y, proba)
            importance = (t.feature_importance
                          if t.feature_importance is not None
                          else t.permutation_importance)
            if importance is not None:
                self.viz.feature_importance(t.name, importance)
        self.viz.comparison_bars(leaderboard, list(RANKING_METRICS))
        return list(self.viz.saved)
