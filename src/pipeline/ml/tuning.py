"""Hyperparameter tuning and cross-validation for the ML module.

Config-driven Grid / Random search over per-model search spaces, using either
Stratified K-fold or TimeSeriesSplit cross-validation. The tuner works on the
:class:`~pipeline.ml.base.BaseModel` wrappers: it builds the underlying
estimator, searches, and returns the best hyperparameters (merged into the
wrapper's params) plus the CV results.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit,
    cross_val_score)

from ingestion.logging_config import get_logger

from .base import BaseModel, MLError

log = get_logger("ml.tuning")

# Compact default search spaces — overridable per model via config
# (``ml.tuning.search_spaces.<model>``).
DEFAULT_SEARCH_SPACES: dict[str, dict[str, list]] = {
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {"max_depth": [4, 6, 8, 12],
                      "min_samples_leaf": [2, 5, 10]},
    "random_forest": {"n_estimators": [200, 400],
                      "max_depth": [None, 10, 20],
                      "min_samples_leaf": [1, 2, 5]},
    "extra_trees": {"n_estimators": [200, 400],
                    "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 2, 5]},
    "xgboost": {"n_estimators": [300, 600],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1]},
    "lightgbm": {"n_estimators": [300, 600],
                 "num_leaves": [15, 31, 63],
                 "learning_rate": [0.03, 0.05, 0.1]},
    "catboost": {"iterations": [300, 600],
                 "depth": [4, 6, 8],
                 "learning_rate": [0.03, 0.05, 0.1]},
    "svm": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
    "knn": {"n_neighbors": [5, 11, 15, 25],
            "weights": ["uniform", "distance"]},
    "naive_bayes": {"var_smoothing": [1e-11, 1e-9, 1e-7]},
    "mlp": {"hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2]},
}


def make_cv_splitter(cfg: dict, random_state: int = 42) -> Any:
    """Build the configured CV splitter (stratified_kfold | time_series)."""
    strategy = str(cfg.get("strategy", "stratified_kfold")).lower()
    n_splits = int(cfg.get("n_splits", 5))
    if strategy in ("stratified_kfold", "stratified", "kfold"):
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    if strategy in ("time_series", "timeseries", "time_series_split"):
        return TimeSeriesSplit(n_splits=n_splits)
    raise MLError(f"unsupported cv strategy '{strategy}'")


class HyperparameterTuner:
    """Grid / Random search over a model's hyperparameter space."""

    def __init__(self, cfg: dict, cv_cfg: dict,
                 random_state: int = 42) -> None:
        self.cfg = cfg or {}
        self.cv_cfg = cv_cfg or {}
        self.random_state = random_state
        self.enabled = bool(self.cfg.get("enabled", False))
        self.method = str(self.cfg.get("method", "random")).lower()
        self.scoring = self.cfg.get("scoring", "roc_auc")
        self.n_iter = int(self.cfg.get("n_iter", 15))
        self.spaces = {**DEFAULT_SEARCH_SPACES,
                       **(self.cfg.get("search_spaces") or {})}

    def tune(self, model: BaseModel, X: pd.DataFrame,
             y: pd.Series) -> dict:
        """Search; return ``{best_params, best_score, method}`` (may be {})."""
        space = self.spaces.get(model.name)
        if not self.enabled or not space:
            return {}
        cv = make_cv_splitter(self.cv_cfg, self.random_state)
        estimator = model.build()
        if self.method == "grid":
            search: Any = GridSearchCV(estimator, space, scoring=self.scoring,
                                       cv=cv, n_jobs=-1, refit=False)
        elif self.method == "random":
            search = RandomizedSearchCV(
                estimator, space, n_iter=self.n_iter, scoring=self.scoring,
                cv=cv, n_jobs=-1, refit=False,
                random_state=self.random_state)
        else:
            raise MLError(f"unsupported tuning method '{self.method}'")
        log.info("tuning %s via %s search (%s)", model.name, self.method,
                 self.scoring)
        search.fit(X, y)
        result = {"best_params": search.best_params_,
                  "best_score": float(search.best_score_),
                  "method": self.method, "scoring": self.scoring}
        model.params.update(search.best_params_)
        model.estimator = None                       # rebuild with best params
        log.info("tuned %s: %s=%.4f %s", model.name, self.scoring,
                 result["best_score"], search.best_params_)
        return result

    def cross_validate(self, model: BaseModel, X: pd.DataFrame,
                       y: pd.Series) -> dict:
        """Score the (already-parameterised) model across CV folds."""
        cv = make_cv_splitter(self.cv_cfg, self.random_state)
        scores = cross_val_score(model.build(), X, y, scoring=self.scoring,
                                 cv=cv, n_jobs=-1)
        model.estimator = None                       # leave wrapper unfitted
        result = {"scoring": self.scoring,
                  "scores": [float(s) for s in scores],
                  "mean": float(np.mean(scores)),
                  "std": float(np.std(scores))}
        log.info("cv %s: %s=%.4f ± %.4f", model.name, self.scoring,
                 result["mean"], result["std"])
        return result
