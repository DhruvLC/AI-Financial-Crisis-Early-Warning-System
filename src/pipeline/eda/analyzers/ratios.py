"""Financial-ratio analysis — domain grouping + target-conditioned behaviour.

The bankruptcy dataset is entirely composed of financial ratios (profitability,
liquidity, leverage, efficiency, cash-flow, …). This analyzer adds the *domain*
lens the generic numeric analyzers cannot:

* buckets every feature into a financial category by keyword matching;
* for each ratio, contrasts its mean between the solvent and bankrupt cohorts
  and reports the standardized mean difference (Cohen's d) — a scale-free
  measure of how strongly the ratio separates the two classes;
* ranks the most discriminative ratios per category.

Everything degrades gracefully: with no target it still emits the category map
and per-category coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisResult, EdaAnalyzer

# Ordered keyword → financial category. First match wins, so more specific
# buckets are listed before generic ones.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("profitability", ("roa", "roe", "profit", "margin", "return", "income",
                       "earning", "eps", "per share")),
    ("liquidity", ("current ratio", "quick ratio", "cash", "liquid",
                   "working capital", "acid")),
    ("leverage", ("debt", "liability", "liabilities", "equity", "borrow",
                  "leverage", "gearing", "solvency", "net worth")),
    ("efficiency", ("turnover", "asset turnover", "frequency", "expense rate",
                    "revenue per", "productivity", "efficiency")),
    ("cash_flow", ("cash flow", "cashflow", "operating cash")),
    ("growth", ("growth", "increase", "change rate")),
    ("interest_rate", ("interest rate", "interest-bearing", "tax rate")),
]


class FinancialRatioAnalysis(EdaAnalyzer):
    """Group financial ratios and measure how each separates the target classes."""

    name = "ratios"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(analyzer=self.name)
        num_cols = self.numeric_features(df)
        if not num_cols:
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="no numeric features")

        category = {c: self._categorize(c) for c in num_cols}
        cat_tbl = pd.DataFrame(
            {"feature": list(category), "category": list(category.values())})
        result.tables["ratio_categories"] = cat_tbl

        cat_counts = cat_tbl["category"].value_counts().to_dict()

        has_target = (self.target_col in df.columns
                      and df[self.target_col].nunique(dropna=True) == 2)
        discr_tbl = pd.DataFrame()
        if has_target:
            discr_tbl = self._discriminative(df, num_cols, category)
            result.tables["ratio_discrimination"] = discr_tbl

        summary: dict = {
            "n_ratios": len(num_cols),
            "categories": {k: int(v) for k, v in cat_counts.items()},
            "target_conditioned": bool(has_target),
        }
        if has_target and not discr_tbl.empty:
            top = discr_tbl.head(15)
            summary["top_discriminative"] = top[
                ["feature", "category", "cohens_d", "mean_solvent",
                 "mean_bankrupt"]
            ].to_dict("records")
            # Per-category mean |d| — which financial dimension matters most.
            cat_strength = (discr_tbl.assign(absd=discr_tbl["cohens_d"].abs())
                            .groupby("category")["absd"].mean()
                            .sort_values(ascending=False))
            summary["category_strength"] = {
                k: round(float(v), 4) for k, v in cat_strength.items()}

        result.summary = summary

        if self.figures is not None and has_target and not discr_tbl.empty:
            self._chart(discr_tbl, result)

        result.note(
            f"{len(num_cols)} ratios in {len(cat_counts)} categories"
            + ("; target-conditioned" if has_target else ""))
        return result

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _categorize(name: str) -> str:
        low = str(name).lower()
        for cat, keys in _CATEGORY_KEYWORDS:
            if any(k in low for k in keys):
                return cat
        return "other"

    def _discriminative(self, df, num_cols, category) -> pd.DataFrame:
        y = df[self.target_col]
        # Positive class == 1 (bankrupt) by convention; else the minority class.
        pos = 1 if 1 in set(y.dropna().unique()) else y.value_counts().idxmin()
        bankrupt = y == pos
        solvent = ~bankrupt
        rows = []
        for c in num_cols:
            a = df.loc[solvent, c].dropna()
            b = df.loc[bankrupt, c].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = a.mean(), b.mean()
            # Pooled SD for Cohen's d.
            na, nb = len(a), len(b)
            va, vb = a.var(ddof=1), b.var(ddof=1)
            pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) \
                if (na + nb - 2) > 0 else 0.0
            d = float((mb - ma) / pooled) if pooled and pooled > 0 else 0.0
            rows.append({
                "feature": c,
                "category": category[c],
                "mean_solvent": float(ma),
                "mean_bankrupt": float(mb),
                "cohens_d": d,
                "abs_cohens_d": abs(d),
            })
        tbl = pd.DataFrame(rows)
        if not tbl.empty:
            tbl = tbl.sort_values("abs_cohens_d", ascending=False).reset_index(
                drop=True)
        return tbl

    def _chart(self, discr_tbl: pd.DataFrame, result: AnalysisResult) -> None:
        import seaborn as sns
        top = discr_tbl.head(20).iloc[::-1]
        with self.figures.figure(figsize=(10, max(4, 0.42 * len(top)))) as fig:
            ax = fig.add_subplot(111)
            colors = ["#c0392b" if d > 0 else "#2c7fb8" for d in top["cohens_d"]]
            ax.barh(top["feature"].astype(str).str.slice(0, 40),
                    top["cohens_d"], color=colors)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title("Most discriminative ratios (Cohen's d: bankrupt − solvent)")
            ax.set_xlabel("Cohen's d  (>0 higher in bankrupt firms)")
            ax.tick_params(labelsize=7)
            path = self.figures.save(fig, "ratio_discrimination")
        result.figures.append(path)
