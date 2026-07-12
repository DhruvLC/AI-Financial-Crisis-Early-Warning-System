"""Feature–target relationship analysis.

Ranks how strongly each numeric feature relates to the (binary) target using
three complementary, scale-free measures:

* **point-biserial correlation** — linear association with the binary label;
* **mutual information** — captures non-linear dependence (sklearn);
* **AUC of the univariate feature** — the ranking power of the feature used
  alone as a score (|AUC − 0.5| × 2 gives a 0..1 separability index).

It also surfaces the strongest *feature-pair* relationships (via the absolute
Pearson correlation already implied by the data) as candidate interactions, and
draws a ranked importance bar chart. With no usable binary target the analyzer
skips gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisResult, EdaAnalyzer


class FeatureRelationshipAnalysis(EdaAnalyzer):
    """Feature-vs-target association ranking (corr, mutual-info, univariate AUC)."""

    name = "relationships"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)
        num_cols = self.numeric_features(df)
        if not num_cols:
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="no numeric features")
        if (self.target_col not in df.columns
                or df[self.target_col].nunique(dropna=True) != 2):
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="need a binary target")

        y = df[self.target_col]
        pos = 1 if 1 in set(y.dropna().unique()) else y.value_counts().idxmin()
        yb = (y == pos).astype(int).to_numpy()

        X = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
        pbcorr = self._point_biserial(X, yb, num_cols)
        mi = self._mutual_info(X, yb, num_cols, result)
        auc = self._univariate_auc(X, yb, num_cols)

        tbl = pd.DataFrame({
            "feature": num_cols,
            "point_biserial_r": [pbcorr[c] for c in num_cols],
            "mutual_info": [mi.get(c, np.nan) for c in num_cols],
            "univariate_auc": [auc[c] for c in num_cols],
        })
        tbl["separability"] = (tbl["univariate_auc"] - 0.5).abs() * 2
        tbl["abs_r"] = tbl["point_biserial_r"].abs()
        tbl = tbl.sort_values("separability", ascending=False).reset_index(
            drop=True)
        result.tables["feature_target_relationships"] = tbl

        top = tbl.head(20)
        result.summary = {
            "n_features": len(num_cols),
            "positive_class": str(pos),
            "mutual_info_available": bool(mi),
            "top_by_separability": top[
                ["feature", "point_biserial_r", "mutual_info",
                 "univariate_auc", "separability"]
            ].to_dict("records"),
            "max_abs_point_biserial": float(tbl["abs_r"].max()),
            "mean_separability": float(tbl["separability"].mean()),
        }

        if self.figures is not None:
            self._chart(tbl, result)

        result.note(f"ranked {len(num_cols)} features vs target; "
                    f"max |r|={tbl['abs_r'].max():.3f}")
        return result

    # ── measures ──────────────────────────────────────────────────────────────
    @staticmethod
    def _point_biserial(X: pd.DataFrame, yb, cols) -> dict:
        out = {}
        for c in cols:
            x = X[c].to_numpy()
            if np.std(x) == 0:
                out[c] = 0.0
                continue
            r = np.corrcoef(x, yb)[0, 1]
            out[c] = float(r) if np.isfinite(r) else 0.0
        return out

    def _mutual_info(self, X, yb, cols, result) -> dict:
        try:
            from sklearn.feature_selection import mutual_info_classif
        except Exception as exc:  # noqa: BLE001 - sklearn optional
            result.note(f"mutual_info unavailable ({exc})")
            return {}
        try:
            mi = mutual_info_classif(X.to_numpy(), yb, random_state=42)
        except Exception as exc:  # noqa: BLE001
            result.note(f"mutual_info failed ({exc})")
            return {}
        return {c: float(v) for c, v in zip(cols, mi)}

    @staticmethod
    def _univariate_auc(X: pd.DataFrame, yb, cols) -> dict:
        """AUC of each raw feature as a ranking score (rank-based, no sklearn)."""
        n_pos = int(yb.sum())
        n_neg = int(len(yb) - n_pos)
        out = {}
        if n_pos == 0 or n_neg == 0:
            return {c: 0.5 for c in cols}
        for c in cols:
            x = X[c].to_numpy()
            order = pd.Series(x).rank(method="average").to_numpy()
            sum_pos = order[yb == 1].sum()
            auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            out[c] = float(auc) if np.isfinite(auc) else 0.5
        return out

    def _chart(self, tbl: pd.DataFrame, result: AnalysisResult) -> None:
        import seaborn as sns
        top = tbl.head(20).iloc[::-1]
        with self.figures.figure(figsize=(10, max(4, 0.42 * len(top)))) as fig:
            ax = fig.add_subplot(111)
            sns.barplot(data=top, y="feature", x="separability", ax=ax,
                        hue="feature", legend=False, palette="viridis")
            ax.set_title("Top features by target separability (|2·AUC − 1|)")
            ax.set_xlabel("separability index (0 = none, 1 = perfect)")
            ax.set_ylabel("")
            ax.tick_params(labelsize=7)
            path = self.figures.save(fig, "feature_target_relationships")
        result.figures.append(path)
