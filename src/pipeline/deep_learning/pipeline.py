"""Deep-learning orchestrator.

Wires the deep-learning components into one config-driven flow over the
engineered feature-store splits:

    load + verify features → build tensor loaders → for each enabled network:
        (build → train with early stopping/checkpoints
         → optimize threshold on VAL → evaluate on val/test → explain)
    →  build the leaderboard  →  plot figures  →  register every network
    →  select + register the best network  →  write the report suite

Design deliberately parallels :class:`pipeline.ml.pipeline.MLPipeline`:
config-driven construction, per-model exception isolation honouring
``fail_fast``, uniform logging, identical metric suite.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger

from .base import DLError, TrainedNetwork, count_parameters, seed_all
from .data_loader import DLData, DLDataLoader
from .evaluation import ModelEvaluator, ThresholdOptimizer, predict_proba
from .models import NETWORK_REGISTRY, build_network
from .registry import DLModelRegistry
from .report import DLReport
from .trainer import Trainer
from .visualization import DLVisualizer

try:  # optional back-end
    import shap
    _HAS_SHAP, _SHAP_REASON = True, None
except Exception as exc:  # noqa: BLE001 - any import failure counts
    _HAS_SHAP, _SHAP_REASON = False, f"shap unavailable: {exc}"

DEFAULT_NETWORKS = list(NETWORK_REGISTRY)
RANKING_METRICS = ("roc_auc", "f1", "recall", "pr_auc")


@dataclass
class DLResult:
    """Everything a deep-learning run produces."""

    data: DLData
    trained: list[TrainedNetwork] = field(default_factory=list)
    leaderboard: pd.DataFrame | None = None
    best: TrainedNetwork | None = None
    best_entry: dict | None = None
    registry_entries: list[dict] = field(default_factory=list)
    registry_files: dict[str, str] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)


class DLPipeline:
    """Run the full train → evaluate → compare → explain → register flow."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.dl_cfg = dict(self.cfg.get("deep_learning", {}))
        self.log = get_logger("dl.pipeline")
        self.fail_fast = bool(self.dl_cfg.get("fail_fast", False))
        self.random_state = int(self.dl_cfg.get("random_state", 42))
        self.deterministic = bool(self.dl_cfg.get("deterministic", True))

        self.target_col = self.cfg.get("data", {}).get("target_col",
                                                       "Bankrupt?")
        store_dir = (self.cfg.get("feature_engineering", {})
                     .get("store", {}).get("dir", "data/features"))
        data_cfg = self.dl_cfg.get("data", {})
        self.loader = DLDataLoader(
            store_dir, self.target_col,
            batch_size=int(data_cfg.get("batch_size", 64)),
            shuffle=bool(data_cfg.get("shuffle", True)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=self.random_state)

        self.evaluator = ModelEvaluator(
            self.dl_cfg.get("evaluation", {}).get("metrics"))
        self.thresholder = ThresholdOptimizer(
            self.dl_cfg.get("threshold_optimization", {}))
        self.explain_cfg = dict(self.dl_cfg.get("explainability", {}))

        out = self.dl_cfg.get("output", {})
        self.models_dir = out.get("models_dir", "models/deep_learning")
        self.reports_dir = out.get("reports_dir", "reports/deep_learning")
        self.figures_dir = out.get("figures_dir",
                                   "reports/deep_learning/figures")
        self.trainer = Trainer(self.dl_cfg, checkpoint_dir=self.models_dir,
                               evaluator=self.evaluator)
        self.registry = DLModelRegistry(self.models_dir)
        self.viz = DLVisualizer(
            self.figures_dir,
            dpi=int(self.dl_cfg.get("visualization", {}).get("dpi", 150)))
        self.reporter = DLReport(self.reports_dir)

    # ── orchestration ─────────────────────────────────────────────────────────
    def run(self, version: str | None = None) -> DLResult:
        """Execute the full deep-learning flow; return a :class:`DLResult`."""
        seed_all(self.random_state, self.deterministic)
        data = self.loader.load(version)
        result = DLResult(data=data)

        networks = self.dl_cfg.get("networks") or DEFAULT_NETWORKS
        for name in networks:
            trained = self._train_one(name, data)
            if trained is not None:
                result.trained.append(trained)

        successes = [t for t in result.trained if not t.failed]
        if not successes:
            raise DLError("no network trained successfully")

        result.leaderboard = self._leaderboard(successes)
        result.figures = self._figures(successes, data, result.leaderboard)

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
            data.target_col, self.dl_cfg)

        result.reports = self.reporter.write(result)
        return result

    # ── per-network flow ──────────────────────────────────────────────────────
    def _train_one(self, name: str, data: DLData) -> TrainedNetwork | None:
        started = time.perf_counter()
        try:
            params = (self.dl_cfg.get("model_params") or {}).get(name)
            model = build_network(name, data.n_features, params)
            trained = TrainedNetwork(
                name=name, model=model, architecture=model.architecture(),
                hyperparameters={"data": {"batch_size":
                                          self.loader.batch_size},
                                 "optimizer": self.dl_cfg.get("optimizer",
                                                              {}),
                                 "scheduler": self.dl_cfg.get("scheduler",
                                                              {}),
                                 "loss": self.dl_cfg.get("loss", {}),
                                 "training": self.dl_cfg.get("training",
                                                             {})},
                n_parameters=count_parameters(model),
                device=str(self.trainer.device))

            trained.history, trained.checkpoints = self.trainer.fit(
                name, model, data.loaders["train"], data.loaders["val"],
                pos_weight=data.pos_weight,
                resume_from=self.dl_cfg.get("training", {})
                .get("resume_from"))

            # threshold chosen on VAL, applied to every split
            X_val, y_val = data.numpy("val")
            X_test, y_test = data.numpy("test")
            val_proba = predict_proba(model, X_val, self.trainer.device)
            trained.threshold, trained.threshold_method = (
                self.thresholder.optimize(y_val, val_proba))

            for split, y, proba in (
                    ("val", y_val, val_proba),
                    ("test", y_test, predict_proba(model, X_test,
                                                   self.trainer.device))):
                trained.evaluations[split] = self.evaluator.evaluate(
                    name, split, y, proba, trained.threshold)

            trained.permutation_importance = self._permutation(
                model, data)
            trained.shap_summary = self._shap(model, data)

            trained.train_seconds = time.perf_counter() - started
            self.log.info("finished %s in %.2fs (test roc_auc=%.4f)", name,
                          trained.train_seconds, trained.metric("roc_auc"))
            return trained
        except DLError as exc:
            if "unsupported" in str(exc):
                self.log.warning("skipping %s: %s", name, exc)
                if self.fail_fast:
                    raise
                return None
            if self.fail_fast:
                raise
            self.log.error("training %s failed: %s", name, exc)
            return TrainedNetwork(name=name, model=None, error=str(exc),
                                  train_seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - isolate per-model failures
            if self.fail_fast:
                raise
            self.log.error("training %s failed: %s", name, exc)
            return TrainedNetwork(name=name, model=None, error=str(exc),
                                  train_seconds=time.perf_counter() - started)

    # ── explainability ────────────────────────────────────────────────────────
    def _permutation(self, model, data: DLData) -> pd.DataFrame | None:
        """Permutation importance (drop in ROC-AUC) on the validation split."""
        if not self.explain_cfg.get("permutation", True):
            return None
        try:
            from sklearn.metrics import roc_auc_score
            repeats = int(self.explain_cfg.get("permutation_repeats", 3))
            X, y = data.numpy("val")
            rng = np.random.default_rng(self.random_state)
            base = roc_auc_score(y, predict_proba(model, X,
                                                  self.trainer.device))
            rows = []
            for j, feature in enumerate(data.features):
                drops = []
                for _ in range(repeats):
                    Xp = X.copy()
                    rng.shuffle(Xp[:, j])
                    drops.append(base - roc_auc_score(
                        y, predict_proba(model, Xp, self.trainer.device)))
                rows.append({"feature": feature,
                             "importance": float(np.mean(drops)),
                             "std": float(np.std(drops))})
            return (pd.DataFrame(rows)
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
        except Exception as exc:  # noqa: BLE001 - explainability best-effort
            self.log.warning("permutation importance failed: %s", exc)
            return None

    def _shap(self, model, data: DLData) -> dict | None:
        """Mean-|SHAP| per feature (None if shap missing/failed)."""
        if not self.explain_cfg.get("shap", True):
            return None
        if not _HAS_SHAP:
            self.log.warning("SHAP skipped: %s", _SHAP_REASON)
            return None
        try:
            max_samples = int(self.explain_cfg.get("shap_max_samples", 200))
            X, _ = data.numpy("val")
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), min(len(X), max_samples),
                             replace=False)
            sample = X[idx]

            def f(arr: np.ndarray) -> np.ndarray:
                return predict_proba(model,
                                     np.asarray(arr, dtype=np.float32),
                                     self.trainer.device)

            explainer = shap.Explainer(
                f, sample, feature_names=data.features)
            values = np.asarray(explainer(sample).values)
            mean_abs = np.abs(values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            return {"n_samples": len(sample),
                    "mean_abs_shap": {data.features[i]: float(mean_abs[i])
                                      for i in order}}
        except Exception as exc:  # noqa: BLE001
            self.log.warning("SHAP failed: %s", exc)
            return None

    # ── comparison & figures ──────────────────────────────────────────────────
    def _leaderboard(self, trained: list[TrainedNetwork]) -> pd.DataFrame:
        """Rank networks by test ROC-AUC, tie-broken by F1/recall/PR-AUC."""
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

    def _figures(self, trained: list[TrainedNetwork], data: DLData,
                 leaderboard: pd.DataFrame) -> list[str]:
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
        self.viz.comparison_bars(leaderboard, list(RANKING_METRICS))
        return list(self.viz.saved)
