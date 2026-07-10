"""Stage 6 — Feature Engineering.

Fit transforms on TRAIN only, then apply to val/test to avoid leakage.
Returns X/y matrices plus the fitted transformers for reuse at inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def _xy(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target].astype(int)
    return X, y


def build(train_df, val_df, test_df, cfg: dict):
    target = cfg["data"]["target_col"]
    fcfg = cfg["features"]

    X_train, y_train = _xy(train_df, target)
    X_val, y_val = _xy(val_df, target)
    X_test, y_test = _xy(test_df, target)

    cols = X_train.columns
    selector = None
    if fcfg.get("drop_low_variance"):
        selector = VarianceThreshold(fcfg.get("variance_threshold", 0.0))
        selector.fit(X_train)
        cols = X_train.columns[selector.get_support()]
        X_train, X_val, X_test = X_train[cols], X_val[cols], X_test[cols]
        print(f"[feat] Kept {len(cols)} features after variance filter")

    scaler = None
    if fcfg.get("scale"):
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=cols)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)
        print("[feat] Standardized features")

    transformers = {"selector": selector, "scaler": scaler, "columns": list(cols)}
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), transformers
