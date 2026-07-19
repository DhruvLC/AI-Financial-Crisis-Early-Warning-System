"""Evaluation for the Self-Supervised Learning module.

Representation quality is judged the standard SSL way — the encoder is
**frozen** and simple downstream models are fit on its embeddings:

* **Linear probe** — logistic regression on frozen embeddings, scored on
  val/test with the shared ML metric suite (accuracy, precision, recall,
  F1, ROC-AUC, PR-AUC, …) via :class:`pipeline.ml.evaluation.
  ModelEvaluator` / :class:`ThresholdOptimizer` — keeping the numbers
  directly comparable with every earlier leaderboard.
* **KNN evaluation** (optional) — k-nearest-neighbour classifier on the
  embeddings, a hyperparameter-free sanity check of local structure.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from ingestion.logging_config import get_logger
from pipeline.deep_learning.evaluation import (METRIC_NAMES,  # noqa: F401
                                               ModelEvaluator,
                                               ThresholdOptimizer)

from .base import SSLError

__all__ = ["METRIC_NAMES", "ModelEvaluator", "ThresholdOptimizer",
           "LinearProbe", "KNNProbe"]

log = get_logger("ssl.evaluation")


class LinearProbe:
    """Logistic regression on frozen embeddings (the linear probe)."""

    def __init__(self, cfg: dict | None = None,
                 evaluator: ModelEvaluator | None = None,
                 thresholder: ThresholdOptimizer | None = None,
                 random_state: int = 42) -> None:
        cfg = cfg or {}
        self.max_iter = int(cfg.get("max_iter", 1000))
        self.C = float(cfg.get("C", 1.0))
        self.class_weight = cfg.get("class_weight", "balanced")
        self.random_state = int(random_state)
        self.evaluator = evaluator or ModelEvaluator()
        self.thresholder = thresholder or ThresholdOptimizer({})

    def evaluate(self, name: str, Z: dict[str, np.ndarray],
                 y: dict[str, np.ndarray]) -> tuple[dict, float, str]:
        """Fit on train embeddings, score every other split.

        Returns ``(evaluations by split, threshold, threshold_method)``;
        the threshold is optimized on the validation split.
        """
        if "train" not in Z:
            raise SSLError("linear probe needs train embeddings")
        clf = LogisticRegression(max_iter=self.max_iter, C=self.C,
                                 class_weight=self.class_weight,
                                 random_state=self.random_state)
        clf.fit(Z["train"], y["train"])

        val_split = "val" if "val" in Z else "train"
        val_proba = clf.predict_proba(Z[val_split])[:, 1]
        threshold, method = self.thresholder.optimize(y[val_split],
                                                      val_proba)
        evals = {}
        for split in Z:
            if split == "train":
                continue
            proba = clf.predict_proba(Z[split])[:, 1]
            evals[split] = self.evaluator.evaluate(
                f"{name}_linear_probe", split, y[split], proba, threshold)
        log.info("%s linear probe: %s", name,
                 {s: round(e.metrics.get("roc_auc", float("nan")), 4)
                  for s, e in evals.items()})
        return evals, float(threshold), method


class KNNProbe:
    """k-NN classifier on frozen embeddings (optional sanity check)."""

    def __init__(self, cfg: dict | None = None,
                 evaluator: ModelEvaluator | None = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.k = int(cfg.get("k", 15))
        if self.k < 1:
            raise SSLError(f"knn: k={self.k} must be >= 1")
        self.evaluator = evaluator or ModelEvaluator()

    def evaluate(self, name: str, Z: dict[str, np.ndarray],
                 y: dict[str, np.ndarray]) -> dict:
        if not self.enabled:
            return {}
        if "train" not in Z:
            raise SSLError("KNN probe needs train embeddings")
        k = min(self.k, len(Z["train"]))
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(Z["train"], y["train"])
        evals = {}
        for split in Z:
            if split == "train":
                continue
            proba = clf.predict_proba(Z[split])[:, 1]
            evals[split] = self.evaluator.evaluate(
                f"{name}_knn", split, y[split], proba, 0.5)
        log.info("%s knn(k=%d): %s", name, k,
                 {s: round(e.metrics.get("roc_auc", float("nan")), 4)
                  for s, e in evals.items()})
        return evals
