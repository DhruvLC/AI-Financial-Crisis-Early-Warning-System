"""Core datatypes for the Machine Learning module.

Mirrors :mod:`pipeline.feature_engineering.base`: a small set of shared
containers plus a :class:`BaseModel` template that every concrete algorithm
under :mod:`pipeline.ml.models` subclasses. Keeping the plumbing here lets the
individual model classes stay small, uniform, and independently testable.

A model wrapper is **config-driven and estimator-agnostic**: it builds its
underlying scikit-learn-compatible estimator from the merged
default + user hyperparameters, exposes uniform ``fit`` / ``predict`` /
``predict_proba``, and reports availability (so optional back-ends such as
LightGBM or CatBoost degrade to a logged skip instead of an import error).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ingestion.logging_config import get_logger


class MLError(RuntimeError):
    """Raised for fatal ML problems (empty data, missing target/features,
    unsupported model, failed training) when the pipeline is configured to
    fail fast."""


@dataclass
class EvaluationResult:
    """Metrics + artefacts of evaluating one trained model on one split."""

    model_name: str
    split: str
    metrics: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    confusion: list[list[int]] | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "split": self.split,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "confusion_matrix": self.confusion,
            "notes": self.notes,
        }


@dataclass
class TrainedModel:
    """Everything one algorithm produces after training + evaluation."""

    name: str
    model: Any                                   # the fitted BaseModel wrapper
    hyperparameters: dict = field(default_factory=dict)
    tuning: dict = field(default_factory=dict)   # best CV params/score
    cv_scores: dict = field(default_factory=dict)
    evaluations: dict[str, EvaluationResult] = field(default_factory=dict)
    threshold: float = 0.5
    threshold_method: str = "default"
    train_seconds: float = 0.0
    feature_importance: pd.DataFrame | None = None
    permutation_importance: pd.DataFrame | None = None
    shap_summary: dict | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def metric(self, name: str, split: str = "test") -> float:
        ev = self.evaluations.get(split)
        return float(ev.metrics.get(name, float("nan"))) if ev else float("nan")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": "failed" if self.failed else "trained",
            "error": self.error,
            "hyperparameters": self.hyperparameters,
            "tuning": self.tuning,
            "cv_scores": self.cv_scores,
            "threshold": self.threshold,
            "threshold_method": self.threshold_method,
            "train_seconds": round(self.train_seconds, 3),
            "evaluations": {s: e.as_dict()
                            for s, e in self.evaluations.items()},
        }


class BaseModel(abc.ABC):
    """Template for one classification algorithm.

    Subclasses set :attr:`name` / :attr:`display_name`, provide sensible
    :meth:`default_params`, and build the underlying estimator in
    :meth:`_build`. Optional back-ends override :meth:`available`.
    """

    name: str = "base"
    display_name: str = "Base Model"
    supports_proba: bool = True

    def __init__(self, params: dict | None = None,
                 random_state: int = 42) -> None:
        self.random_state = random_state
        self.params = {**self.default_params(), **(params or {})}
        self.estimator = None
        self.feature_names: list[str] = []
        self.log = get_logger(f"ml.model.{self.name}")

    # ── to override ───────────────────────────────────────────────────────────
    @classmethod
    def available(cls) -> tuple[bool, str | None]:
        """Return (is_available, reason_if_not). Optional back-ends override."""
        return True, None

    def default_params(self) -> dict:
        return {}

    @abc.abstractmethod
    def _build(self) -> Any:
        """Construct and return the underlying (unfitted) estimator."""

    # ── uniform API ───────────────────────────────────────────────────────────
    def build(self) -> Any:
        self.estimator = self._build()
        return self.estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        if X.empty:
            raise MLError(f"{self.name}: cannot fit on an empty dataset")
        if self.estimator is None:
            self.build()
        self.feature_names = list(X.columns)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted(X)
        return np.asarray(self.estimator.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities (decision_function fallback → [0,1])."""
        self._check_fitted(X)
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(X))[:, 1]
        scores = np.asarray(self.estimator.decision_function(X), dtype=float)
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi > lo else np.full_like(scores, .5)

    def native_importance(self) -> pd.DataFrame | None:
        """Per-feature importance from the fitted estimator, if it has any."""
        est = self.estimator
        if est is None:
            return None
        values = None
        if hasattr(est, "feature_importances_"):
            values = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            values = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
        if values is None or len(values) != len(self.feature_names):
            return None
        return (pd.DataFrame({"feature": self.feature_names,
                              "importance": values})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _check_fitted(self, X: pd.DataFrame) -> None:
        if self.estimator is None:
            raise MLError(f"{self.name}: predict before fit")
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise MLError(
                f"{self.name}: {len(missing)} feature(s) missing at predict "
                f"time (e.g. {missing[:3]})")

    def get_params(self) -> dict:
        return dict(self.params)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
