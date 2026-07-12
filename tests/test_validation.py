"""Unit tests for the Data Validation module (``src/validation``).

Run with the project's venv, no extra deps required::

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Make ``src`` importable exactly like the run_*.py entry points do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from validation.base import CheckOutcome, Severity  # noqa: E402
from validation.checks import (  # noqa: E402
    DuplicateDetector, FinancialValidator, MissingValueAnalyzer,
    OutlierDetector, SchemaValidator, TimeSeriesValidator,
)
from validation.quality import QualityScorer  # noqa: E402
from validation.runner import DataValidationRunner  # noqa: E402
from validation.schemas import (  # noqa: E402
    ColumnSpec, FinancialSpec, SourceSchema, schema_for,
)

CTX = {"now": datetime.now(timezone.utc), "source": "test"}


def _codes(outcome: CheckOutcome) -> set[str]:
    return {f.code for f in outcome.findings}


class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        self.spec = SourceSchema(
            source="t",
            columns=[ColumnSpec("id", "string"), ColumnSpec("value", "numeric"),
                     ColumnSpec("note", "string", required=False)],
        )
        self.check = SchemaValidator()

    def test_valid_schema(self):
        df = pd.DataFrame({"id": ["a"], "value": [1.0]})
        out = self.check.run(df, self.spec, CTX)
        self.assertEqual(out.worst, Severity.INFO)
        self.assertEqual(out.metrics["schema_score"], 1.0)

    def test_missing_required_is_error(self):
        df = pd.DataFrame({"id": ["a"]})
        out = self.check.run(df, self.spec, CTX)
        self.assertIn("missing_required_columns", _codes(out))
        self.assertEqual(out.worst, Severity.ERROR)

    def test_unexpected_column_warns_when_not_dynamic(self):
        df = pd.DataFrame({"id": ["a"], "value": [1.0], "surprise": [9]})
        out = self.check.run(df, self.spec, CTX)
        self.assertIn("unexpected_columns", _codes(out))

    def test_dtype_mismatch_warns(self):
        df = pd.DataFrame({"id": ["a"], "value": ["not_a_number"]})
        out = self.check.run(df, self.spec, CTX)
        self.assertIn("dtype_mismatch", _codes(out))


class TestMissingValues(unittest.TestCase):
    def test_required_mostly_missing_is_error(self):
        spec = SourceSchema(source="t", columns=[ColumnSpec("value", "numeric")])
        df = pd.DataFrame({"value": [np.nan] * 9 + [1.0]})
        out = MissingValueAnalyzer().run(df, spec, CTX)
        self.assertIn("required_column_mostly_missing", _codes(out))

    def test_completeness_metric(self):
        spec = SourceSchema(source="t")
        df = pd.DataFrame({"a": [1, 2, np.nan, 4]})   # 25% missing
        out = MissingValueAnalyzer().run(df, spec, CTX)
        self.assertAlmostEqual(out.metrics["completeness_score"], 0.75, places=3)


class TestDuplicates(unittest.TestCase):
    def test_duplicate_rows(self):
        spec = SourceSchema(source="t")
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        out = DuplicateDetector().run(df, spec, CTX)
        self.assertIn("duplicate_rows", _codes(out))
        self.assertEqual(out.metrics["duplicate_rows"], 1)

    def test_duplicate_entity_timestamps(self):
        spec = SourceSchema(source="t", entity_column="eid", date_columns=["date"])
        df = pd.DataFrame({
            "eid": ["A", "A", "B"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
            "v": [1, 2, 3],
        })
        out = DuplicateDetector().run(df, spec, CTX)
        self.assertIn("duplicate_entity_timestamps", _codes(out))


class TestOutliers(unittest.TestCase):
    def test_flags_injected_outliers(self):
        spec = SourceSchema(source="t")
        rng = np.random.RandomState(0)
        vals = np.r_[rng.normal(0, 1, 200), np.repeat(1000.0, 40)]
        df = pd.DataFrame({"x": vals})
        out = OutlierDetector({"outliers": {"isolation_forest": False}}).run(
            df, spec, CTX)
        self.assertIn("iqr_outliers", _codes(out))
        self.assertGreater(out.metrics["mean_iqr_outlier_pct"], 0.0)

    def test_skips_small_data(self):
        spec = SourceSchema(source="t")
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = OutlierDetector({"outliers": {"isolation_forest": False}}).run(
            df, spec, CTX)
        self.assertTrue(out.skipped)


class TestFinancial(unittest.TestCase):
    def test_negative_and_nonpositive(self):
        spec = SourceSchema(
            source="t",
            financial=FinancialSpec(nonneg_columns=["revenue"],
                                    positive_columns=["assets"]),
        )
        df = pd.DataFrame({"revenue": [100.0, -5.0], "assets": [10.0, 0.0]})
        out = FinancialValidator().run(df, spec, CTX)
        self.assertIn("negative_value", _codes(out))
        self.assertIn("nonpositive_value", _codes(out))
        self.assertEqual(out.worst, Severity.ERROR)

    def test_future_date_flagged(self):
        spec = SourceSchema(source="t", date_columns=["date"],
                            financial=FinancialSpec())
        future = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) \
            + pd.Timedelta(days=365)
        df = pd.DataFrame({"date": [pd.Timestamp("2020-01-01"), future]})
        out = FinancialValidator().run(df, spec, CTX)
        self.assertIn("future_date", _codes(out))

    def test_invalid_fiscal_year(self):
        spec = SourceSchema(
            source="t",
            financial=FinancialSpec(fiscal_year_column="fy"),
        )
        df = pd.DataFrame({"fy": [2020, 1800, 3000]})
        out = FinancialValidator().run(df, spec, CTX)
        self.assertIn("invalid_fiscal_year", _codes(out))

    def test_concept_rules_long_format(self):
        spec = SourceSchema(
            source="t",
            financial=FinancialSpec(concept_column="concept", value_column="value",
                                    concept_rules={"Assets": "positive"}),
        )
        df = pd.DataFrame({"concept": ["Assets", "Assets"], "value": [10.0, -3.0]})
        out = FinancialValidator().run(df, spec, CTX)
        self.assertIn("invalid_concept_value", _codes(out))


class TestTimeSeries(unittest.TestCase):
    def test_unordered_and_duplicate(self):
        spec = SourceSchema(source="t", date_columns=["date"])
        df = pd.DataFrame({"date": pd.to_datetime(
            ["2020-01-03", "2020-01-01", "2020-01-01", "2020-01-02"])})
        out = TimeSeriesValidator().run(df, spec, CTX)
        self.assertIn("not_chronological", _codes(out))
        self.assertIn("duplicate_timestamps", _codes(out))

    def test_ordered_series_ok(self):
        spec = SourceSchema(source="t", date_columns=["date"])
        df = pd.DataFrame({"date": pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])})
        out = TimeSeriesValidator().run(df, spec, CTX)
        self.assertEqual(out.metrics["timeliness_score"], 1.0)


class TestQualityScorer(unittest.TestCase):
    def test_perfect_scores_100_grade_a(self):
        outcomes = [
            CheckOutcome("schema", metrics={"schema_score": 1.0}),
            CheckOutcome("missing_values", metrics={"completeness_score": 1.0}),
            CheckOutcome("duplicates", metrics={"uniqueness_score": 1.0}),
            CheckOutcome("financial", metrics={"validity_score": 1.0}),
            CheckOutcome("outliers", metrics={"mean_iqr_outlier_pct": 0.0}),
            CheckOutcome("time_series", metrics={"timeliness_score": 1.0}),
        ]
        score, grade, _ = QualityScorer().score(outcomes)
        self.assertAlmostEqual(score, 100.0, places=6)
        self.assertEqual(grade, "A")

    def test_grade_thresholds(self):
        self.assertEqual(QualityScorer.grade(95), "A")
        self.assertEqual(QualityScorer.grade(85), "B")
        self.assertEqual(QualityScorer.grade(72), "C")
        self.assertEqual(QualityScorer.grade(61), "D")
        self.assertEqual(QualityScorer.grade(40), "F")

    def test_missing_component_defaults_to_one(self):
        # Only schema present; others default to 1.0 -> still a high score.
        score, _, comps = QualityScorer().score(
            [CheckOutcome("schema", metrics={"schema_score": 1.0})])
        self.assertEqual(comps["completeness"], 1.0)
        self.assertAlmostEqual(score, 100.0, places=6)


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.interim = os.path.join(self.tmp.name, "interim")
        self.reports = os.path.join(self.tmp.name, "reports")
        os.makedirs(self.interim)
        self.storage = {"interim_dir": self.interim,
                        "metadata_layer_dir": self.tmp.name}
        self.cfg = {"reports_dir": self.reports,
                    "outliers": {"isolation_forest": False}}

    def tearDown(self):
        self.tmp.cleanup()

    def _runner(self):
        return DataValidationRunner(self.storage, self.cfg)

    def test_missing_dataset_reported_absent(self):
        reports = self._runner().run(only=["fred"])
        self.assertEqual(len(reports), 1)
        self.assertFalse(reports[0].present)

    def test_corrupted_file_isolated(self):
        bad = os.path.join(self.interim, "fred.parquet")
        with open(bad, "w") as f:
            f.write("this is not parquet")
        report = self._runner().run(only=["fred"])[0]
        self.assertTrue(report.present)
        self.assertIsNotNone(report.load_error)
        self.assertFalse(report.is_valid)

    def test_end_to_end_writes_reports(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "GDP": np.linspace(1, 100, 100),
        })
        df.to_csv(os.path.join(self.interim, "fred.csv"), index=False)
        reports = self._runner().run(only=["fred"])
        r = reports[0]
        self.assertTrue(r.present and r.is_valid)
        self.assertGreater(r.quality_score, 0)
        self.assertTrue(os.path.exists(os.path.join(self.reports, "fred.json")))
        self.assertTrue(os.path.exists(os.path.join(self.reports, "_summary.json")))
        self.assertTrue(os.path.exists(os.path.join(self.reports, "_summary.md")))

    def test_fail_fast_raises_on_errors(self):
        from validation.runner import DataValidationError
        # A required column fully missing -> schema error -> not valid.
        df = pd.DataFrame({"wrong": [1, 2, 3]})
        df.to_csv(os.path.join(self.interim, "imf.csv"), index=False)
        cfg = {**self.cfg, "fail_fast": True}
        with self.assertRaises(DataValidationError):
            DataValidationRunner(self.storage, cfg).run(only=["imf"])


class TestSchemaFor(unittest.TestCase):
    def test_known_and_unknown(self):
        self.assertEqual(schema_for("fred").source, "fred")
        unknown = schema_for("does_not_exist")
        self.assertTrue(unknown.dynamic_columns)


if __name__ == "__main__":
    unittest.main()
