"""Stages 7 + Model Comparison — Traditional ML baselines.

Trains RandomForest and XGBoost, evaluates on a held-out set, and returns
the fitted models plus a comparison table. Deep-learning / transformer /
self-supervised models (diagram stages 8-12) are stubbed below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def _evaluate(name, model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "ROC_AUC": roc_auc_score(y, proba),
        "PR_AUC": average_precision_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }


def train_random_forest(X_train, y_train, cfg):
    p = cfg["models"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        class_weight=p["class_weight"],
        random_state=cfg["split"]["random_state"], n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, cfg):
    from xgboost import XGBClassifier

    p = cfg["models"]["xgboost"]
    # Handle class imbalance: ratio of negatives to positives.
    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos
    model = XGBClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        scale_pos_weight=neg / pos,
        eval_metric="aucpr", n_jobs=-1,
        random_state=cfg["split"]["random_state"],
    )
    model.fit(X_train, y_train)
    return model


def train_and_compare(train, val, cfg):
    """train/val are (X, y) tuples. Returns (models dict, comparison DataFrame)."""
    X_train, y_train = train
    X_val, y_val = val

    models = {
        "RandomForest": train_random_forest(X_train, y_train, cfg),
        "XGBoost": train_xgboost(X_train, y_train, cfg),
    }
    rows = [_evaluate(name, m, X_val, y_val) for name, m in models.items()]
    comparison = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    print("\n[models] Validation comparison:\n", comparison.to_string(index=False))
    return models, comparison


def best_model(models: dict, comparison: pd.DataFrame):
    name = comparison.iloc[0]["model"]
    return name, models[name]


# ── Phase 2 stubs (diagram stages 8-12) ──────────────────────────────────
def train_deep_mlp(*a, **k):        # stage 8
    raise NotImplementedError("Deep MLP — phase 2 (needs torch).")

def train_sequence_model(*a, **k):  # stage 9 (LSTM/GRU) — needs time-series data
    raise NotImplementedError("LSTM/GRU — phase 2.")

def train_transformer(*a, **k):     # stage 10 (TabTransformer / FT-Transformer)
    raise NotImplementedError("Transformer — phase 2.")

def self_supervised_pretrain(*a, **k):  # stage 11
    raise NotImplementedError("Self-supervised pretraining — phase 2.")
