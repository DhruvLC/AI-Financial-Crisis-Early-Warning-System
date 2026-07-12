"""End-to-end runner for the AI Financial Crisis Early Warning System.

Wires diagram stages 2 -> 3 -> 6 -> 7 -> 13 -> Output.

Usage:
    python src/run_pipeline.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import os
import pickle

import pandas as pd
import yaml

from pipeline import (
    data_collection, data_validation, data_prep, features, models,
    explain, risk_score,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    target = cfg["data"]["target_col"]

    # Stage 2 — Data Collection
    df = data_collection.load(cfg)

    # Stage 4/5 — Data Validation & Quality (gate before any transformation)
    data_validation.validate(df, cfg)

    # Stage 3 — Data Preparation
    df = data_prep.clean(df, target)
    df = data_prep.clip_outliers(df, target)
    train_df, val_df, test_df = data_prep.split(df, cfg)

    # Stage 6 — Feature Engineering
    train, val, test, transformers = features.build(train_df, val_df, test_df, cfg)

    # Stage 7 + comparison — Traditional ML
    fitted, comparison = models.train_and_compare(train, val, cfg)
    best_name, best = models.best_model(fitted, comparison)
    print(f"\n[run] Best model on validation: {best_name}")

    # Final test-set evaluation of the winner
    X_test, y_test = test
    test_metrics = models._evaluate(best_name, best, X_test, y_test)
    print(f"[run] Test metrics: {test_metrics}")

    # Stage 13 — Explainability
    try:
        explain.shap_summary(best, X_test, cfg, best_name)
    except Exception as e:  # SHAP can be finicky across versions — don't crash the run
        print(f"[explain] Skipped SHAP ({e})")

    # Output — Risk scores for the test set
    scores = risk_score.score(best, X_test, cfg)
    reports_dir = cfg["output"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    comparison.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    scores.to_csv(os.path.join(reports_dir, "risk_scores.csv"), index=False)
    print(f"\n[run] Risk score sample:\n{scores.head().to_string(index=False)}")

    # Persist the winning model + transformers for inference
    models_dir = cfg["output"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
        pickle.dump({"model": best, "transformers": transformers,
                     "name": best_name}, f)
    print(f"[run] Saved model bundle -> {models_dir}/best_model.pkl")
    print("[run] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
