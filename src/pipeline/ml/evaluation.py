"""Model evaluation and threshold optimization for the ML module.

:class:`ModelEvaluator` computes the full binary-classification metric suite
(accuracy, precision, recall, F1, ROC-AUC, PR-AUC, balanced accuracy, MCC,
Cohen's kappa, log loss, Brier score) at a given decision threshold, and
:class:`ThresholdOptimizer` finds the optimal threshold on the validation
split (Youden index | max-F1 | custom).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, roc_curve)

from ingestion.logging_config import get_logger

from .base import EvaluationResult, MLError

log = get_logger("ml.evaluation")

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                "balanced_accuracy", "mcc", "cohen_kappa", "log_loss",
                "brier_score")


class ModelEvaluator:
    """Compute the configured metric suite for one model on one split."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = list(metrics or METRIC_NAMES)

    def evaluate(self, model_name: str, split: str, y_true: pd.Series,
                 y_proba: np.ndarray,
                 threshold: float = 0.5) -> EvaluationResult:
        """Score probabilities against truth at ``threshold``."""
        y_true = np.asarray(y_true)
        if len(y_true) == 0:
            raise MLError(f"cannot evaluate '{model_name}' on empty '{split}'")
        y_proba = np.clip(np.asarray(y_proba, dtype=float), 0.0, 1.0)
        y_pred = (y_proba >= threshold).astype(int)

        fns = {
            "accuracy": lambda: accuracy_score(y_true, y_pred),
            "precision": lambda: precision_score(y_true, y_pred,
                                                 zero_division=0),
            "recall": lambda: recall_score(y_true, y_pred, zero_division=0),
            "f1": lambda: f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": lambda: roc_auc_score(y_true, y_proba),
            "pr_auc": lambda: average_precision_score(y_true, y_proba),
            "balanced_accuracy": lambda: balanced_accuracy_score(y_true,
                                                                 y_pred),
            "mcc": lambda: matthews_corrcoef(y_true, y_pred),
            "cohen_kappa": lambda: cohen_kappa_score(y_true, y_pred),
            "log_loss": lambda: log_loss(y_true, y_proba, labels=[0, 1]),
            "brier_score": lambda: brier_score_loss(y_true, y_proba),
        }
        result = EvaluationResult(model_name=model_name, split=split,
                                  threshold=float(threshold))
        for name in self.metrics:
            fn = fns.get(name)
            if fn is None:
                result.notes.append(f"unknown metric '{name}' skipped")
                continue
            try:
                result.metrics[name] = float(fn())
            except ValueError as exc:  # e.g. single-class split for AUC
                result.metrics[name] = float("nan")
                result.notes.append(f"{name} failed: {exc}")
        result.confusion = confusion_matrix(y_true, y_pred,
                                            labels=[0, 1]).tolist()
        log.info("evaluated %s on %s @%.3f: roc_auc=%.4f f1=%.4f "
                 "recall=%.4f", model_name, split, threshold,
                 result.metrics.get("roc_auc", float("nan")),
                 result.metrics.get("f1", float("nan")),
                 result.metrics.get("recall", float("nan")))
        return result


class ThresholdOptimizer:
    """Pick the classification threshold on held-out probabilities."""

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.method = str(cfg.get("method", "youden")).lower()
        self.custom = float(cfg.get("custom_threshold", 0.5))

    def optimize(self, y_true: pd.Series,
                 y_proba: np.ndarray) -> tuple[float, str]:
        """Return ``(threshold, method_used)``."""
        if not self.enabled:
            return 0.5, "default"
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba, dtype=float)
        if self.method == "custom":
            return self.custom, "custom"
        if len(np.unique(y_true)) < 2:
            log.warning("threshold optimization skipped: single-class split")
            return 0.5, "default"
        if self.method == "youden":
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            best = thresholds[np.argmax(tpr - fpr)]
            return float(np.clip(best, 0.0, 1.0)), "youden"
        if self.method in ("max_f1", "f1"):
            grid = np.unique(np.clip(y_proba, 0.0, 1.0))
            if len(grid) > 200:                       # bounded search
                grid = np.quantile(grid, np.linspace(0, 1, 200))
            f1s = [f1_score(y_true, (y_proba >= t).astype(int),
                            zero_division=0) for t in grid]
            return float(grid[int(np.argmax(f1s))]), "max_f1"
        raise MLError(f"unsupported threshold method '{self.method}'")
