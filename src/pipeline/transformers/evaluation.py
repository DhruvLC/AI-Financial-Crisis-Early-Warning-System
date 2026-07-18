"""Evaluation and prediction for the Transformer Models module.

Pure reuse: the transformer stage scores with exactly the metric suite the
classical ML and deep-learning stages use — accuracy, precision, recall,
F1, ROC-AUC, PR-AUC, MCC, balanced accuracy, confusion matrix, log loss,
Brier — via :class:`pipeline.ml.evaluation.ModelEvaluator` /
:class:`ThresholdOptimizer`, plus the batched :func:`predict_proba` and
sklearn classification-text-report helpers from the deep-learning stage.
Keeping one metric implementation makes the three leaderboards directly
comparable.
"""
from __future__ import annotations

from pipeline.deep_learning.evaluation import (METRIC_NAMES,  # noqa: F401
                                               ModelEvaluator,
                                               ThresholdOptimizer,
                                               classification_text_report,
                                               predict_proba)

__all__ = ["METRIC_NAMES", "ModelEvaluator", "ThresholdOptimizer",
           "classification_text_report", "predict_proba"]
