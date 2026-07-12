"""Target-variable analysis — class distribution, imbalance, charts."""
from __future__ import annotations

import pandas as pd

from ..base import AnalysisResult, EdaAnalyzer


class TargetAnalysis(EdaAnalyzer):
    """Bankruptcy class distribution, imbalance ratio, and distribution charts."""

    name = "target"

    def _analyze(self, df: pd.DataFrame) -> AnalysisResult:
        if self.target_col not in df.columns:
            return AnalysisResult(analyzer=self.name, skipped=True,
                                  skip_reason="no target column")
        result = AnalysisResult(analyzer=self.name)
        tgt = df[self.target_col]
        counts = tgt.value_counts(dropna=False).sort_index()
        n = int(len(tgt))
        pct = (counts / n * 100.0).round(4)

        tbl = pd.DataFrame({
            "class": [str(k) for k in counts.index],
            "count": counts.values.astype(int),
            "percentage": pct.values,
        })
        result.tables["target_distribution"] = tbl

        majority = int(counts.max())
        minority = int(counts.min())
        imbalance_ratio = round(majority / max(minority, 1), 3)
        # For the canonical binary bankruptcy target: 1 == positive (bankrupt).
        positive = int(counts.get(1, 0)) if 1 in counts.index else minority
        negative = n - positive
        result.summary = {
            "n_samples": n,
            "n_classes": int(tgt.nunique(dropna=True)),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "class_percentages": {str(k): float(v) for k, v in pct.items()},
            "majority_count": majority,
            "minority_count": minority,
            "imbalance_ratio": imbalance_ratio,
            "positive_count": positive,
            "negative_count": negative,
            "positive_pct": round(positive / n * 100, 4),
            "is_imbalanced": bool(imbalance_ratio >= 5),
        }

        if self.figures is not None:
            self._chart(tbl, result)

        result.note(f"{n} samples; imbalance ratio {imbalance_ratio}:1 "
                    f"(positive {result.summary['positive_pct']}%)")
        return result

    def _chart(self, tbl: pd.DataFrame, result: AnalysisResult) -> None:
        import seaborn as sns
        with self.figures.figure(figsize=(11, 5)) as fig:
            ax1 = fig.add_subplot(121)
            sns.barplot(data=tbl, x="class", y="count", ax=ax1,
                        hue="class", legend=False, palette="rocket")
            ax1.set_title(f"'{self.target_col}' class counts")
            for i, v in enumerate(tbl["count"]):
                ax1.text(i, v, str(int(v)), ha="center", va="bottom")

            ax2 = fig.add_subplot(122)
            ax2.pie(tbl["count"], labels=tbl["class"], autopct="%1.2f%%",
                    startangle=90, colors=sns.color_palette("rocket",
                                                            len(tbl)))
            ax2.set_title("Class proportion")
            path = self.figures.save(fig, "target_distribution")
        result.figures.append(path)
