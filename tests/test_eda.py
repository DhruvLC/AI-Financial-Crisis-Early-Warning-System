"""Unit tests for the Exploratory Data Analysis module (``src/pipeline/eda``).

Run with the project's venv, no extra deps required::

    .venv/bin/python -m unittest discover -s tests -v

Figures are disabled by default in these tests (``figures=None``) so they stay
fast and headless; a couple of cases exercise the FigureManager explicitly.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Make ``src`` importable exactly like the run_*.py entry points do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline.eda import (  # noqa: E402
    ANALYZER_REGISTRY, AnalysisResult, BusinessInsightsEngine, DEFAULT_ORDER,
    EdaAnalyzer, EdaError, EdaReport, EdaRunner, FigureManager,
)
from pipeline.eda.analyzers import (  # noqa: E402
    CorrelationAnalysis, DescriptiveStatistics, DimensionalityAnalysis,
    FeatureDistributionAnalysis, FeatureRelationshipAnalysis,
    FinancialRatioAnalysis, MissingValueAnalysis, OutlierAnalysis,
    DatasetOverview, TargetAnalysis,
)

TARGET = "Bankrupt?"


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Toy financial-ratio-ish frame with a strong-ish, imbalanced signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    df = pd.DataFrame({
        "ROA(C)": rng.normal(0, 1, n) - 1.5 * y,        # separates the classes
        "Operating Gross Margin": rng.normal(5, 2, n),
        "Debt ratio %": rng.normal(0, 1, n) + 1.2 * y,  # separates the classes
        "Cash flow rate": rng.normal(0, 1, n),
        "Total Asset Turnover": rng.exponential(2, n),  # skewed
        TARGET: y,
    })
    return df


class TestBaseAndResult(unittest.TestCase):
    def test_analysis_result_as_dict(self):
        r = AnalysisResult(analyzer="x")
        r.tables["t"] = pd.DataFrame({"a": [1]})
        r.note("hello")
        d = r.as_dict()
        self.assertEqual(d["analyzer"], "x")
        self.assertEqual(d["status"], "completed")
        self.assertIn("t", d["tables"])
        self.assertIn("hello", d["notes"])

    def test_empty_dataset_raises(self):
        with self.assertRaises(EdaError):
            DatasetOverview(target_col=TARGET).run(pd.DataFrame())

    def test_missing_target_raises(self):
        with self.assertRaises(EdaError):
            DescriptiveStatistics(target_col=TARGET).run(
                pd.DataFrame({"a": [1, 2, 3]}))

    def test_disabled_analyzer_skips(self):
        res = DatasetOverview(cfg={"enabled": False}, target_col=TARGET).run(
            _frame(20))
        self.assertTrue(res.skipped)

    def test_crash_normalised_to_eda_error(self):
        class Boom(EdaAnalyzer):
            name = "boom"

            def _analyze(self, df):
                raise ValueError("kaboom")

        with self.assertRaises(EdaError):
            Boom(target_col=TARGET).run(_frame(20))

    def test_input_not_mutated(self):
        df = _frame(50)
        before = df.copy()
        DescriptiveStatistics(target_col=TARGET).run(df)
        pd.testing.assert_frame_equal(df, before)


class TestOverviewTargetDescriptive(unittest.TestCase):
    def test_overview(self):
        res = DatasetOverview(target_col=TARGET).run(_frame())
        self.assertEqual(res.summary["n_rows"], 400)
        self.assertEqual(res.summary["n_numeric_features"], 5)
        self.assertTrue(res.summary["target"]["is_binary"])

    def test_target_imbalance(self):
        res = TargetAnalysis(target_col=TARGET).run(_frame())
        self.assertGreaterEqual(res.summary["imbalance_ratio"], 1.0)
        self.assertIn("positive_pct", res.summary)

    def test_descriptive_columns(self):
        res = DescriptiveStatistics(target_col=TARGET).run(_frame())
        tbl = res.tables["descriptive_statistics"]
        for col in ("mean", "median", "std", "skewness", "kurtosis", "p95"):
            self.assertIn(col, tbl.columns)
        self.assertEqual(len(tbl), 5)


class TestQualityAnalyzers(unittest.TestCase):
    def test_missing_counts(self):
        df = _frame(100)
        df.loc[:9, "Cash flow rate"] = np.nan
        res = MissingValueAnalysis(target_col=TARGET).run(df)
        self.assertEqual(res.summary["total_missing_cells"], 10)
        self.assertEqual(res.summary["n_features_with_missing"], 1)

    def test_outliers_and_isolation_forest(self):
        res = OutlierAnalysis(cfg={"isolation_forest": True},
                              target_col=TARGET).run(_frame())
        self.assertIn("outlier_summary", res.tables)
        self.assertIn("mean_iqr_outlier_pct", res.summary)
        iso = res.summary.get("isolation_forest", {})
        self.assertTrue(iso.get("available"))
        self.assertGreaterEqual(iso.get("n_anomalies", 0), 0)

    def test_outliers_no_numeric_skips(self):
        df = pd.DataFrame({"c": ["a", "b", "c"], TARGET: [0, 1, 0]})
        res = OutlierAnalysis(target_col=TARGET).run(df)
        self.assertTrue(res.skipped)


class TestStructureAnalyzers(unittest.TestCase):
    def test_distribution_flags_skew(self):
        res = FeatureDistributionAnalysis(target_col=TARGET).run(_frame())
        self.assertIn("distribution_shape", res.tables)
        # The exponential column should be flagged as highly skewed.
        self.assertGreaterEqual(res.summary["n_highly_skewed"], 1)

    def test_correlation_and_vif(self):
        res = CorrelationAnalysis(target_col=TARGET).run(_frame())
        self.assertIn("correlation_pearson", res.tables)
        self.assertIn("multicollinearity_vif", res.tables)
        self.assertIn("n_high_correlation_pairs", res.summary)

    def test_correlation_needs_two_features(self):
        df = pd.DataFrame({"a": [1.0, 2, 3], TARGET: [0, 1, 0]})
        res = CorrelationAnalysis(target_col=TARGET).run(df)
        self.assertTrue(res.skipped)


class TestDomainAnalyzers(unittest.TestCase):
    def test_ratios_categorize_and_discriminate(self):
        res = FinancialRatioAnalysis(target_col=TARGET).run(_frame())
        cats = res.tables["ratio_categories"]["category"].tolist()
        self.assertIn("profitability", cats)
        self.assertIn("leverage", cats)
        self.assertTrue(res.summary["target_conditioned"])
        self.assertIn("top_discriminative", res.summary)

    def test_relationships_ranking(self):
        res = FeatureRelationshipAnalysis(target_col=TARGET).run(_frame())
        tbl = res.tables["feature_target_relationships"]
        self.assertIn("univariate_auc", tbl.columns)
        self.assertIn("separability", tbl.columns)
        # separability lies in [0, 1]
        self.assertTrue((tbl["separability"] >= 0).all())
        self.assertTrue((tbl["separability"] <= 1.0001).all())

    def test_dimensionality_pca(self):
        res = DimensionalityAnalysis(cfg={"projection": False},
                                     target_col=TARGET).run(_frame())
        self.assertIn("pca_explained_variance", res.tables)
        cum = res.tables["pca_explained_variance"]["cumulative_variance"]
        self.assertAlmostEqual(cum.iloc[-1], 1.0, places=4)
        self.assertIn("n_components_95pct", res.summary)


class TestInsightsEngine(unittest.TestCase):
    def _run_all(self, df):
        results = {}
        for name in DEFAULT_ORDER:
            cls = ANALYZER_REGISTRY[name]
            results[name] = cls(target_col=TARGET).run(df)
        return results

    def test_generates_prioritized_insights(self):
        results = self._run_all(_frame())
        payload = BusinessInsightsEngine().generate(results)
        self.assertGreater(payload["n_insights"], 0)
        self.assertIn("headline", payload)
        # sorted by severity: criticals (if any) come first
        ranks = [{"critical": 0, "warning": 1, "info": 2}[i["severity"]]
                 for i in payload["insights"]]
        self.assertEqual(ranks, sorted(ranks))

    def test_engine_never_raises_on_empty(self):
        payload = BusinessInsightsEngine().generate({})
        self.assertEqual(payload["n_insights"], 0)


class TestFiguresAndReport(unittest.TestCase):
    def test_figure_manager_saves_png(self):
        with tempfile.TemporaryDirectory() as d:
            fm = FigureManager(figures_dir=d)
            res = TargetAnalysis(target_col=TARGET, figures=fm).run(_frame())
            self.assertTrue(res.figures)
            self.assertTrue(os.path.exists(res.figures[0]))
            self.assertTrue(res.figures[0].endswith(".png"))

    def test_report_writes_all_formats(self):
        with tempfile.TemporaryDirectory() as d:
            results = [DatasetOverview(target_col=TARGET).run(_frame()),
                       TargetAnalysis(target_col=TARGET).run(_frame())]
            ins = BusinessInsightsEngine().generate(
                {r.analyzer: r for r in results})
            meta = {"dataset": "toy", "n_rows": 400, "n_cols": 6,
                    "target_col": TARGET, "n_analyzers_run": 2,
                    "n_analyzers_skipped": 0}
            out = EdaReport(d).write(results, ins, meta)
            for key in ("json", "md", "html"):
                self.assertTrue(os.path.exists(out[key]))
            self.assertTrue(out["csv"])
            self.assertTrue(os.path.isdir(os.path.join(d, "statistics")))


class TestRunnerEndToEnd(unittest.TestCase):
    def test_runner_full_suite(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"eda": {"reports_dir": d,
                           "figures": {"enabled": False}}}
            result = EdaRunner(cfg, target_col=TARGET).run(_frame(), "toy")
            self.assertEqual(len(result.results), len(DEFAULT_ORDER))
            self.assertGreater(result.insights["n_insights"], 0)
            self.assertTrue(os.path.exists(result.outputs["json"]))
            self.assertTrue(os.path.exists(result.outputs["md"]))
            self.assertTrue(os.path.exists(result.outputs["html"]))

    def test_runner_fail_fast_off_isolates_errors(self):
        # A frame with only the target -> most analyzers skip, none crash.
        df = pd.DataFrame({TARGET: [0, 1] * 30})
        with tempfile.TemporaryDirectory() as d:
            cfg = {"eda": {"reports_dir": d, "fail_fast": False,
                           "figures": {"enabled": False}}}
            result = EdaRunner(cfg, target_col=TARGET).run(df, "toy")
            self.assertEqual(len(result.results), len(DEFAULT_ORDER))


if __name__ == "__main__":
    unittest.main(verbosity=2)
