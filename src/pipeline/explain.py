"""Stage 13 — Explainable AI (SHAP + feature importance)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd


def shap_summary(model, X, cfg, model_name: str):
    """Compute SHAP values and save a summary plot + top-feature table."""
    import shap

    reports_dir = cfg["output"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    n = min(cfg["explainability"]["shap_max_samples"], len(X))
    Xs = X.sample(n, random_state=cfg["split"]["random_state"]) if n < len(X) else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    # Binary classifiers may return a list [class0, class1]; take positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, Xs, show=False)
    plot_path = os.path.join(reports_dir, f"shap_{model_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()

    importance = (
        pd.DataFrame({"feature": Xs.columns,
                      "mean_abs_shap": abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    table_path = os.path.join(reports_dir, f"feature_importance_{model_name}.csv")
    importance.to_csv(table_path, index=False)

    print(f"[explain] SHAP plot -> {plot_path}")
    print(f"[explain] Top drivers:\n{importance.head(10).to_string(index=False)}")
    return importance
