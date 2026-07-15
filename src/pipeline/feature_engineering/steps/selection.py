"""Feature selection — keep the most informative features, drop the rest.

A configurable battery of selectors, each independently toggleable, run in
sequence on the train split (leak-safe: selections learned on train are
re-applied verbatim to val/test):

* **variance threshold** — drop (near-)constant features;
* **correlation-to-target** — drop features whose |corr(x, y)| falls below a
  floor (cheap univariate relevance filter);
* **mutual information** — keep the top-k by ``mutual_info_classif``;
* **ANOVA F-test** — keep the top-k by ``f_classif``;
* **chi-square** — keep the top-k by ``chi2`` (features min-max shifted to be
  non-negative, as chi2 requires);
* **RFE** — recursive feature elimination with a logistic-regression estimator;
* **model-based** — ``SelectFromModel`` over a random forest's importances.

The statistical selectors (MI / ANOVA / chi2) *vote*: a feature survives when
selected by at least ``min_votes`` of the enabled voters — more stable than any
single criterion. RFE and model-based selection then run on the survivors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE, SelectFromModel, SelectKBest, chi2, f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression

from ..base import FeatureEngineeringError, FeatureResult, FeatureStep


class FeatureSelection(FeatureStep):
    """Multi-technique feature selector (fit on train)."""

    name = "selection"

    # ── fit ───────────────────────────────────────────────────────────────────
    def _fit_transform(self, df: pd.DataFrame) -> FeatureResult:
        X, y = self.split_xy(df)
        if y is None:
            raise FeatureEngineeringError(
                "feature selection requires the target column")
        if X.shape[1] < 2:
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason="fewer than 2 numeric features")

        stats: dict = {}
        removed: list[str] = []

        # 1. Variance threshold — always first (constants break the others).
        X, dropped = self._variance_threshold(X)
        removed += dropped
        stats["variance_threshold"] = {"n_removed": len(dropped),
                                       "removed": dropped}

        # 2. Correlation-to-target floor.
        if self.cfg.get("correlation", True):
            X, dropped = self._correlation_floor(X, y)
            removed += dropped
            stats["correlation"] = {"n_removed": len(dropped),
                                    "removed": dropped}

        # 3. Statistical voting: MI / ANOVA / chi2 each nominate top-k.
        X, dropped, vote_stats = self._statistical_vote(X, y)
        removed += dropped
        stats["statistical_vote"] = vote_stats

        # 4. RFE on the survivors.
        if self.cfg.get("rfe", True) and X.shape[1] > 2:
            X, dropped = self._rfe(X, y)
            removed += dropped
            stats["rfe"] = {"n_removed": len(dropped), "removed": dropped}

        # 5. Model-based (SelectFromModel / random forest).
        if self.cfg.get("model_based", True) and X.shape[1] > 2:
            X, dropped = self._model_based(X, y)
            removed += dropped
            stats["model_based"] = {"n_removed": len(dropped),
                                    "removed": dropped}

        selected = list(X.columns)
        keep = [c for c in df.columns if c in selected or c == self.target_col
                or c not in self.numeric_features(df)]
        out = df[keep]

        result = FeatureResult(step=self.name, df=out, selected=selected,
                               removed=removed, stats=stats)
        result.params = {"n_selected": len(selected),
                         "min_votes": int(self.cfg.get("min_votes", 2))}
        result.note(f"selected {len(selected)} of "
                    f"{len(selected) + len(removed)} features")
        return result

    # ── individual selectors ──────────────────────────────────────────────────
    def _variance_threshold(self, X: pd.DataFrame):
        thresh = float(self.cfg.get("variance_threshold", 1e-8))
        variances = X.var()
        dropped = variances[variances <= thresh].index.tolist()
        return X.drop(columns=dropped), dropped

    def _correlation_floor(self, X: pd.DataFrame, y: pd.Series):
        floor = float(self.cfg.get("correlation_floor", 0.0))
        if floor <= 0:
            return X, []
        rel = X.corrwith(y).abs().fillna(0.0)
        dropped = rel[rel < floor].index.tolist()
        # Never drop everything — keep at least the strongest few.
        if len(dropped) >= X.shape[1]:
            dropped = rel.sort_values().index[:-5].tolist()
        return X.drop(columns=dropped), dropped

    def _statistical_vote(self, X: pd.DataFrame, y: pd.Series):
        k = min(int(self.cfg.get("top_k", 40)), X.shape[1])
        voters: dict[str, set] = {}

        if self.cfg.get("mutual_information", True):
            voters["mutual_information"] = self._top_k(
                X, y, mutual_info_classif, k)
        if self.cfg.get("anova", True):
            voters["anova"] = self._top_k(X, y, f_classif, k)
        if self.cfg.get("chi_square", True):
            # chi2 requires non-negative features — shift each to min 0.
            X_nn = X - X.min()
            voters["chi_square"] = self._top_k(X_nn, y, chi2, k)

        if not voters:
            return X, [], {"skipped": True}

        min_votes = min(int(self.cfg.get("min_votes", 2)), len(voters))
        votes = pd.Series(0, index=X.columns)
        for sel in voters.values():
            votes[list(sel)] += 1
        keep = votes[votes >= min_votes].index.tolist()
        if len(keep) < 2:  # degenerate vote — fall back to the union
            keep = sorted(set().union(*voters.values()))
        dropped = [c for c in X.columns if c not in keep]
        vote_stats = {
            "voters": {name: len(sel) for name, sel in voters.items()},
            "min_votes": min_votes,
            "n_removed": len(dropped),
            "n_kept": len(keep),
        }
        return X[keep], dropped, vote_stats

    def _top_k(self, X: pd.DataFrame, y: pd.Series, score_fn, k: int) -> set:
        try:
            selector = SelectKBest(score_fn, k=k).fit(X, y)
            return set(X.columns[selector.get_support()])
        except Exception as exc:  # noqa: BLE001 - one voter failing is OK
            self.log.warning("selector %s failed (%s) — skipping voter",
                             getattr(score_fn, "__name__", score_fn), exc)
            return set(X.columns)  # abstain: votes for everything

    def _rfe(self, X: pd.DataFrame, y: pd.Series):
        n_keep = min(int(self.cfg.get("rfe_n_features", 30)), X.shape[1])
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
        rfe = RFE(est, n_features_to_select=n_keep,
                  step=float(self.cfg.get("rfe_step", 0.1))).fit(X, y)
        keep = X.columns[rfe.support_].tolist()
        dropped = [c for c in X.columns if c not in keep]
        return X[keep], dropped

    def _model_based(self, X: pd.DataFrame, y: pd.Series):
        rf = RandomForestClassifier(
            n_estimators=int(self.cfg.get("model_n_estimators", 200)),
            class_weight="balanced",
            random_state=int(self.cfg.get("random_state", 42)),
            n_jobs=-1)
        sfm = SelectFromModel(
            rf, threshold=self.cfg.get("model_threshold", "median"),
            max_features=self.cfg.get("model_max_features")).fit(X, y)
        keep = X.columns[sfm.get_support()].tolist()
        if len(keep) < 2:
            return X, []
        dropped = [c for c in X.columns if c not in keep]
        return X[keep], dropped
