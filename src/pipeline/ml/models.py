"""Model zoo for the Machine Learning module.

One small :class:`~pipeline.ml.base.BaseModel` subclass per algorithm, each
with defaults tuned for the imbalanced financial-crisis target. Optional
back-ends (XGBoost, LightGBM, CatBoost) report their availability instead of
raising at import time, so the pipeline degrades gracefully to whatever is
installed.

``MODEL_REGISTRY`` maps config keys → classes; :func:`build_model` is the
single factory used by the pipeline, tuner, and tests.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .base import BaseModel, MLError

try:  # optional back-end
    from xgboost import XGBClassifier
    _HAS_XGB, _XGB_REASON = True, None
except Exception as exc:  # noqa: BLE001 - any import failure counts
    _HAS_XGB, _XGB_REASON = False, f"xgboost unavailable: {exc}"

try:  # optional back-end
    from lightgbm import LGBMClassifier
    _HAS_LGBM, _LGBM_REASON = True, None
except Exception as exc:  # noqa: BLE001
    _HAS_LGBM, _LGBM_REASON = False, f"lightgbm unavailable: {exc}"

try:  # optional back-end
    from catboost import CatBoostClassifier
    _HAS_CAT, _CAT_REASON = True, None
except Exception as exc:  # noqa: BLE001
    _HAS_CAT, _CAT_REASON = False, f"catboost unavailable: {exc}"


# ── linear ──────────────────────────────────────────────────────────────────
class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"
    display_name = "Logistic Regression"

    def default_params(self) -> dict:
        return {"C": 1.0, "max_iter": 2000, "class_weight": "balanced",
                "solver": "lbfgs"}

    def _build(self) -> Any:
        return LogisticRegression(random_state=self.random_state,
                                  **self.params)


# ── trees ───────────────────────────────────────────────────────────────────
class DecisionTreeModel(BaseModel):
    name = "decision_tree"
    display_name = "Decision Tree"

    def default_params(self) -> dict:
        return {"max_depth": 8, "min_samples_leaf": 5,
                "class_weight": "balanced"}

    def _build(self) -> Any:
        return DecisionTreeClassifier(random_state=self.random_state,
                                      **self.params)


class RandomForestModel(BaseModel):
    name = "random_forest"
    display_name = "Random Forest"

    def default_params(self) -> dict:
        return {"n_estimators": 400, "max_depth": None,
                "min_samples_leaf": 2, "class_weight": "balanced",
                "n_jobs": -1}

    def _build(self) -> Any:
        return RandomForestClassifier(random_state=self.random_state,
                                      **self.params)


class ExtraTreesModel(BaseModel):
    name = "extra_trees"
    display_name = "Extra Trees"

    def default_params(self) -> dict:
        return {"n_estimators": 400, "max_depth": None,
                "min_samples_leaf": 2, "class_weight": "balanced",
                "n_jobs": -1}

    def _build(self) -> Any:
        return ExtraTreesClassifier(random_state=self.random_state,
                                    **self.params)


# ── boosting ────────────────────────────────────────────────────────────────
class XGBoostModel(BaseModel):
    name = "xgboost"
    display_name = "XGBoost"

    @classmethod
    def available(cls) -> tuple[bool, str | None]:
        return _HAS_XGB, _XGB_REASON

    def default_params(self) -> dict:
        return {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.9, "colsample_bytree": 0.9,
                "scale_pos_weight": 1.0, "eval_metric": "logloss",
                "n_jobs": -1}

    def _build(self) -> Any:
        if not _HAS_XGB:
            raise MLError(_XGB_REASON or "xgboost unavailable")
        return XGBClassifier(random_state=self.random_state, **self.params)


class LightGBMModel(BaseModel):
    name = "lightgbm"
    display_name = "LightGBM"

    @classmethod
    def available(cls) -> tuple[bool, str | None]:
        return _HAS_LGBM, _LGBM_REASON

    def default_params(self) -> dict:
        return {"n_estimators": 600, "max_depth": -1, "learning_rate": 0.05,
                "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.9,
                "class_weight": "balanced", "n_jobs": -1, "verbosity": -1}

    def _build(self) -> Any:
        if not _HAS_LGBM:
            raise MLError(_LGBM_REASON or "lightgbm unavailable")
        return LGBMClassifier(random_state=self.random_state, **self.params)


class CatBoostModel(BaseModel):
    name = "catboost"
    display_name = "CatBoost"

    @classmethod
    def available(cls) -> tuple[bool, str | None]:
        return _HAS_CAT, _CAT_REASON

    def default_params(self) -> dict:
        return {"iterations": 600, "depth": 6, "learning_rate": 0.05,
                "auto_class_weights": "Balanced", "verbose": 0,
                "allow_writing_files": False}

    def _build(self) -> Any:
        if not _HAS_CAT:
            raise MLError(_CAT_REASON or "catboost unavailable")
        return CatBoostClassifier(random_state=self.random_state,
                                  **self.params)


# ── kernel / distance / probabilistic / neural ──────────────────────────────
class SVMModel(BaseModel):
    name = "svm"
    display_name = "Support Vector Machine"

    def default_params(self) -> dict:
        return {"C": 1.0, "kernel": "rbf", "gamma": "scale",
                "class_weight": "balanced", "probability": True}

    def _build(self) -> Any:
        return SVC(random_state=self.random_state, **self.params)


class KNNModel(BaseModel):
    name = "knn"
    display_name = "K-Nearest Neighbors"

    def default_params(self) -> dict:
        return {"n_neighbors": 15, "weights": "distance", "n_jobs": -1}

    def _build(self) -> Any:
        return KNeighborsClassifier(**self.params)   # no random_state


class NaiveBayesModel(BaseModel):
    name = "naive_bayes"
    display_name = "Gaussian Naive Bayes"

    def default_params(self) -> dict:
        return {"var_smoothing": 1e-9}

    def _build(self) -> Any:
        return GaussianNB(**self.params)             # no random_state


class MLPModel(BaseModel):
    name = "mlp"
    display_name = "Multi-Layer Perceptron"

    def default_params(self) -> dict:
        return {"hidden_layer_sizes": (64, 32), "activation": "relu",
                "alpha": 1e-3, "learning_rate_init": 1e-3, "max_iter": 500,
                "early_stopping": True}

    def _build(self) -> Any:
        return MLPClassifier(random_state=self.random_state, **self.params)


MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    cls.name: cls for cls in (
        LogisticRegressionModel, DecisionTreeModel, RandomForestModel,
        ExtraTreesModel, XGBoostModel, LightGBMModel, CatBoostModel,
        SVMModel, KNNModel, NaiveBayesModel, MLPModel,
    )
}


def build_model(name: str, params: dict | None = None,
                random_state: int = 42) -> BaseModel:
    """Factory: instantiate the registered model ``name`` (config key)."""
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise MLError(
            f"unsupported model '{name}' — expected one of "
            f"{sorted(MODEL_REGISTRY)}")
    ok, reason = cls.available()
    if not ok:
        raise MLError(reason or f"{name} unavailable")
    return cls(params=params, random_state=random_state)
