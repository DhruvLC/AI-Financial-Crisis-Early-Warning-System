"""Stage 3 — Data Preparation.

Cleaning (missing/duplicates), basic outlier handling, and a
train/validation/test split (random or time-based).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop duplicates, fill numeric NaNs with the column median."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if before != len(df):
        print(f"[prep] Dropped {before - len(df)} duplicate rows")

    if target_col not in df.columns:
        raise KeyError(
            f"target_col '{target_col}' not in columns. "
            f"Available (first 10): {list(df.columns)[:10]}"
        )

    feature_cols = [c for c in df.columns if c != target_col]
    numeric = df[feature_cols].select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(df[numeric].median())
    print(f"[prep] Filled NaNs in {len(numeric)} numeric columns")
    return df


def clip_outliers(df: pd.DataFrame, target_col: str, z: float = 5.0) -> pd.DataFrame:
    """Winsorize extreme values beyond +/- z std (per feature)."""
    feats = [c for c in df.columns if c != target_col]
    num = df[feats].select_dtypes(include=[np.number]).columns
    mean, std = df[num].mean(), df[num].std().replace(0, 1)
    lower, upper = mean - z * std, mean + z * std
    df[num] = df[num].clip(lower=lower, upper=upper, axis=1)
    return df


def split(df: pd.DataFrame, cfg: dict):
    """Return (train_df, val_df, test_df)."""
    s = cfg["split"]
    target = cfg["data"]["target_col"]

    if s.get("time_based") and s.get("date_col"):
        df = df.sort_values(s["date_col"]).reset_index(drop=True)
        n = len(df)
        test_start = int(n * (1 - s["test_size"]))
        val_start = int(test_start * (1 - s["val_size"]))
        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]
        print("[prep] Time-based split")
    else:
        train_df, test_df = train_test_split(
            df, test_size=s["test_size"],
            random_state=s["random_state"], stratify=df[target],
        )
        # carve validation out of the remaining train portion
        val_frac = s["val_size"] / (1 - s["test_size"])
        train_df, val_df = train_test_split(
            train_df, test_size=val_frac,
            random_state=s["random_state"], stratify=train_df[target],
        )
        print("[prep] Stratified random split")

    print(f"[prep] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))
