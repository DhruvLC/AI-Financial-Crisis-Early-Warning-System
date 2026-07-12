"""Feature generation — derive new features from the existing ones.

Creates, in a leak-safe fit-on-train fashion, several families of engineered
features (each independently toggleable via config):

* **log transforms** — ``sign(x)·log1p(|x|)`` for skewed features (seeded from
  the EDA ``distributions`` hints, or auto-detected);
* **ratio features** — pairwise ``a / b`` for a bounded set of the most
  informative base features;
* **interaction features** — pairwise products ``a · b`` (financial-ratio
  interactions);
* **polynomial features** — ``x**d`` powers up to a configurable degree;
* **difference features** — pairwise ``a − b``;
* **rolling-window features** — per-entity rolling mean/std over an ordered
  (time-series) dataset — *only* when an entity/date column is configured;
* **entity aggregates** — per-entity mean/std/min/max broadcast back to rows —
  *only* when an entity column is configured.

The set of generated column names is learned on train and re-created verbatim on
val/test so the three splits always share an identical feature schema. On the
cross-sectional bankruptcy dataset the rolling/entity families skip gracefully
(no entity/date column), while the log/ratio/interaction/polynomial families
apply.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..base import FeatureResult, FeatureStep

_EPS = 1e-9


class FeatureGeneration(FeatureStep):
    """Generate log / ratio / interaction / polynomial / difference / rolling /
    entity-aggregate features (fit on train)."""

    name = "generation"

    def __init__(self, cfg=None, target_col=None, hints=None) -> None:
        super().__init__(cfg, target_col, hints)
        self._plan: dict = {}          # fitted generation plan (reused on val/test)

    # ── fit ───────────────────────────────────────────────────────────────────
    def _fit_transform(self, df: pd.DataFrame) -> FeatureResult:
        num_cols = self.numeric_features(df)
        if not num_cols:
            return FeatureResult(step=self.name, df=df, skipped=True,
                                 skip_reason="no numeric features")

        base = self._base_features(df, num_cols)
        self._plan = {
            "log": self._plan_log(df, num_cols) if self.cfg.get(
                "log_transform", True) else [],
            "ratios": self._plan_pairs(base, self.cfg.get("ratios", True),
                                       int(self.cfg.get("max_ratio_pairs", 30))),
            "interactions": self._plan_pairs(
                base, self.cfg.get("interactions", True),
                int(self.cfg.get("max_interaction_pairs", 30))),
            "differences": self._plan_pairs(
                base, self.cfg.get("differences", True),
                int(self.cfg.get("max_difference_pairs", 20))),
            "polynomial": (base if self.cfg.get("polynomial", True) else []),
            "poly_degree": int(self.cfg.get("polynomial_degree", 2)),
            "rolling": self._plan_rolling(df, num_cols),
            "entity_agg": self._plan_entity(df, num_cols),
        }

        out, generated = self._apply_plan(df)
        result = FeatureResult(step=self.name, df=out, generated=generated)
        result.params = {"plan_counts": {
            "log": len(self._plan["log"]),
            "ratios": len(self._plan["ratios"]),
            "interactions": len(self._plan["interactions"]),
            "differences": len(self._plan["differences"]),
            "polynomial": len(self._plan["polynomial"]) * max(
                self._plan["poly_degree"] - 1, 0),
            "rolling": len(self._plan["rolling"]),
            "entity_agg": len(self._plan["entity_agg"]),
        }}
        result.stats = {"n_generated": len(generated),
                        "n_features_after": out.shape[1] - 1}
        result.note(f"generated {len(generated)} new feature(s)")
        return result

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out, _ = self._apply_plan(df)
        return out

    # ── plan builders ─────────────────────────────────────────────────────────
    def _base_features(self, df: pd.DataFrame, num_cols: list[str]) -> list[str]:
        """Pick a bounded, informative base set for pairwise generation.

        Prefers EDA-flagged discriminative ratios, then fills up to ``top_k``
        with the highest-variance columns so pairwise expansion stays tractable
        on wide tables (O(k^2)).
        """
        top_k = int(self.cfg.get("base_top_k", 12))
        prefer = [c for c in self.hints.get("top_discriminative", [])
                  if c in num_cols]
        if len(prefer) >= top_k:
            return prefer[:top_k]
        variances = df[num_cols].var().sort_values(ascending=False)
        fill = [c for c in variances.index if c not in prefer]
        return (prefer + fill)[:top_k]

    def _plan_log(self, df: pd.DataFrame, num_cols: list[str]) -> list[str]:
        hinted = [c for c in self.hints.get("skewed_features", [])
                  if c in num_cols]
        if hinted:
            return hinted
        # Fallback: auto-detect skew when no EDA hint is available.
        thresh = float(self.cfg.get("log_skew_threshold", 1.0))
        skew = df[num_cols].skew(numeric_only=True).abs()
        return skew[skew >= thresh].index.tolist()

    @staticmethod
    def _plan_pairs(base: list[str], enabled, max_pairs: int) -> list[tuple]:
        if not enabled or len(base) < 2:
            return []
        pairs = list(itertools.combinations(base, 2))
        return pairs[:max_pairs]

    def _plan_rolling(self, df: pd.DataFrame, num_cols: list[str]) -> list[str]:
        if not self.cfg.get("rolling", False):
            return []
        ent, date = self.cfg.get("entity_col"), self.cfg.get("date_col")
        if not ent or ent not in df.columns:
            return []
        cols = [c for c in num_cols if c not in (ent, date)]
        return cols[:int(self.cfg.get("max_rolling_features", 10))]

    def _plan_entity(self, df: pd.DataFrame, num_cols: list[str]) -> list[str]:
        if not self.cfg.get("entity_aggregates", False):
            return []
        ent = self.cfg.get("entity_col")
        if not ent or ent not in df.columns:
            return []
        cols = [c for c in num_cols if c != ent]
        return cols[:int(self.cfg.get("max_entity_features", 10))]

    # ── apply the fitted plan ─────────────────────────────────────────────────
    def _apply_plan(self, df: pd.DataFrame):
        out = df.copy()
        generated: list[str] = []
        p = self._plan

        for c in p.get("log", []):
            if c in out:
                name = f"log__{c}"
                out[name] = np.sign(out[c]) * np.log1p(out[c].abs())
                generated.append(name)

        for a, b in p.get("ratios", []):
            if a in out and b in out:
                name = f"ratio__{a}__over__{b}"
                out[name] = out[a] / (out[b].replace(0, np.nan) + _EPS)
                out[name] = out[name].replace([np.inf, -np.inf], np.nan)
                generated.append(name)

        for a, b in p.get("interactions", []):
            if a in out and b in out:
                name = f"mul__{a}__x__{b}"
                out[name] = out[a] * out[b]
                generated.append(name)

        for a, b in p.get("differences", []):
            if a in out and b in out:
                name = f"diff__{a}__minus__{b}"
                out[name] = out[a] - out[b]
                generated.append(name)

        degree = p.get("poly_degree", 2)
        for c in p.get("polynomial", []):
            if c not in out:
                continue
            for d in range(2, degree + 1):
                name = f"poly__{c}__pow{d}"
                out[name] = out[c] ** d
                generated.append(name)

        generated += self._apply_rolling(df, out, p)
        generated += self._apply_entity(df, out, p)

        # Any divisions/log of odd values may introduce NaNs — fill with the
        # train medians captured at fit time (leak-safe on val/test).
        if generated:
            out = self._fill_generated(out, generated)
        return out, generated

    def _apply_rolling(self, src: pd.DataFrame, out: pd.DataFrame, p) -> list[str]:
        cols = p.get("rolling", [])
        if not cols:
            return []
        ent = self.cfg.get("entity_col")
        date = self.cfg.get("date_col")
        window = int(self.cfg.get("rolling_window", 3))
        frame = src
        if date and date in frame.columns:
            frame = frame.sort_values([ent, date])
        gen = []
        grp = frame.groupby(ent, sort=False)
        for c in cols:
            for stat in ("mean", "std"):
                name = f"roll{window}_{stat}__{c}"
                rolled = grp[c].transform(
                    lambda s: s.rolling(window, min_periods=1).agg(stat))
                out[name] = rolled.reindex(out.index)
                gen.append(name)
        return gen

    def _apply_entity(self, src: pd.DataFrame, out: pd.DataFrame, p) -> list[str]:
        cols = p.get("entity_agg", [])
        if not cols:
            return []
        ent = self.cfg.get("entity_col")
        gen = []
        grp = src.groupby(ent, sort=False)
        for c in cols:
            for stat in ("mean", "std"):
                name = f"ent_{stat}__{c}"
                out[name] = grp[c].transform(stat).reindex(out.index)
                gen.append(name)
        return gen

    def _fill_generated(self, out: pd.DataFrame, generated: list[str]):
        if not hasattr(self, "_gen_medians") or not self._gen_medians:
            # Fit-time: learn and store medians of the generated columns.
            self._gen_medians = {c: float(out[c].median()) for c in generated
                                 if c in out}
        for c in generated:
            if c in out:
                out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(
                    self._gen_medians.get(c, 0.0))
        return out

    # medians of generated columns, learned on the train split
    _gen_medians: dict = {}
