"""Evaluation and prediction for the Deep Learning module.

Reuses the classical ML metric suite — :class:`pipeline.ml.evaluation.
ModelEvaluator` and :class:`ThresholdOptimizer` cover accuracy, precision,
recall, F1, ROC-AUC, PR-AUC, confusion matrix, log loss, MCC, balanced
accuracy, and more — so classical and deep models are scored identically
and their leaderboards are directly comparable.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report

from ingestion.logging_config import get_logger
from pipeline.ml.evaluation import (METRIC_NAMES, ModelEvaluator,
                                    ThresholdOptimizer)

from .base import DLError

__all__ = ["METRIC_NAMES", "ModelEvaluator", "ThresholdOptimizer",
           "predict_proba", "classification_text_report"]

log = get_logger("dl.evaluation")


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: np.ndarray,
                  device: torch.device | str = "cpu",
                  batch_size: int = 1024) -> np.ndarray:
    """Positive-class probabilities from a trained network (batched)."""
    if len(X) == 0:
        raise DLError("cannot predict on empty input")
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise DLError("prediction input contains non-finite values")
    model = model.to(device)
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        batch = torch.as_tensor(X[i:i + batch_size]).to(device)
        logits = model(batch)
        if not torch.isfinite(logits).all():
            raise DLError("model produced non-finite logits — "
                          "training likely diverged")
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def classification_text_report(y_true: np.ndarray, y_proba: np.ndarray,
                               threshold: float = 0.5) -> str:
    """sklearn classification report at the given threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return classification_report(np.asarray(y_true), y_pred,
                                 labels=[0, 1],
                                 target_names=["healthy", "crisis"],
                                 zero_division=0)
