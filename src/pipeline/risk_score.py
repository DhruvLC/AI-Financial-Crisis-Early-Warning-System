"""Output — Financial Risk Score (0-100) + risk level."""
from __future__ import annotations

import pandas as pd


def score(model, X, cfg, ids=None) -> pd.DataFrame:
    """Map predicted bankruptcy probability to a 0-100 risk score + band."""
    scale = cfg["output"]["risk_score_scale"]
    proba = model.predict_proba(X)[:, 1]
    risk = (proba * scale).round(1)

    def band(r):
        if r < scale * 0.33:
            return "Low"
        if r < scale * 0.66:
            return "Medium"
        return "High"

    out = pd.DataFrame({
        "risk_score": risk,
        "probability": proba.round(4),
        "risk_level": [band(r) for r in risk],
    })
    if ids is not None:
        out.insert(0, "id", list(ids))
    return out
