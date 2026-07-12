"""Unit tests for the Data Preparation (preprocessing) module
(``src/pipeline/preprocessing``).

Run with the project's venv, no extra deps required::

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Make ``src`` importable exactly like the run_*.py entry points do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline.preprocessing import (  # noqa: E402
    CategoricalEncoder, DataCleaner, DuplicateRemover, FeatureScaler,
    Imputer, OutlierTreatment, PreprocessingError, PreprocessingPipeline,
    PreprocessingReport, STEP_REGISTRY,
)
from pipeline.preprocessing.lineage import LineageTracker  # noqa: E402

TARGET = "Bankrupt?"


def _toy_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "f_num": rng.normal(0, 1, n),
        "f_num2": rng.normal(10, 2, n),
        "f_cat": rng.choice(["a", "b", "c"], n),
        TARGET: rng.integers(0, 2, n),
    })
    return df


class TestCleaning(unittest.TestCase):
    def test_drops_invalid_target_and_inf(self):
        df = _toy_frame(60)
        df.loc[0, TARGET] = np.nan
        df.loc[1, "f_num"] = np.inf
        step = DataCleaner(cfg={}, target_col=TARGET)
        res = step.fit_transform(df)
        self.assertEqual(res.stats["dropped_invalid_target"], 1)
        self.assertEqual(res.stats["dropped_inf_rows"], 1)
        self.assertEqual(len(res.df), 58)

    def test_normalizes_null_like_tokens(self):
        df = _toy_frame(30)
        df.loc[0, "f_cat"] = "  NA "
        step = DataCleaner(cfg={"lowercase_categoricals": False}, target_col=TARGET)
        res = step.fit_transform(df)
        self.assertTrue(pd.isna(res.df.loc[0, "f_cat"]))


class TestDuplicates(unittest.TestCase):
    def test_removes_full_row_duplicates(self):
        df = _toy_frame(20)
        dup = pd.concat([df, df.iloc[:5]], ignore_index=True)
        step = DuplicateRemover(cfg={}, target_col=TARGET)
        res = step.fit_transform(dup)
        self.assertEqual(res.stats["removed_rows"], 5)
        self.assertEqual(len(res.df), 20)


class TestImputer(unittest.TestCase):
    def test_median_impute_fit_on_train(self):
        train = _toy_frame(100)
        train.loc[:9, "f_num"] = np.nan
        step = Imputer(cfg={"strategy": "median"}, target_col=TARGET)
        res = step.fit_transform(train)
        self.assertEqual(res.df["f_num"].isna().sum(), 0)
        # transform a held-out frame with its own NaNs -> uses train medians
        val = _toy_frame(20, seed=1)
        val.loc[0, "f_num"] = np.nan
        out = step.transform(val)
        self.assertEqual(out["f_num"].isna().sum(), 0)

    def test_rejects_unknown_strategy(self):
        with self.assertRaises(PreprocessingError):
            Imputer(cfg={"strategy": "bogus"}, target_col=TARGET)


class TestOutliers(unittest.TestCase):
    def test_winsorize_clips_bounds(self):
        train = _toy_frame(200)
        train.loc[0, "f_num"] = 1000.0
        step = OutlierTreatment(cfg={"method": "winsorize"}, target_col=TARGET)
        res = step.fit_transform(train)
        self.assertLess(res.df["f_num"].max(), 1000.0)

    def test_filter_removes_only_from_train(self):
        train = _toy_frame(200)
        step = OutlierTreatment(
            cfg={"method": "zscore_filter", "zscore_threshold": 2.0},
            target_col=TARGET)
        res = step.fit_transform(train)
        self.assertLessEqual(len(res.df), 200)
        # val/test must be untouched by a filter step
        val = _toy_frame(50, seed=2)
        out = step.transform(val)
        self.assertEqual(len(out), 50)

    def test_rejects_unknown_method(self):
        with self.assertRaises(PreprocessingError):
            OutlierTreatment(cfg={"method": "bogus"}, target_col=TARGET)


class TestEncoding(unittest.TestCase):
    def test_onehot_expands_and_aligns_unseen(self):
        train = _toy_frame(100)
        step = CategoricalEncoder(cfg={"method": "onehot"}, target_col=TARGET)
        res = step.fit_transform(train)
        self.assertNotIn("f_cat", res.df.columns)
        self.assertIn("f_cat=a", res.df.columns)
        # held-out frame missing a category still aligns to fitted columns
        val = _toy_frame(30, seed=3)
        val["f_cat"] = "a"
        out = step.transform(val)
        self.assertListEqual(list(out.columns), list(res.df.columns))

    def test_ordinal_maps_and_handles_unknown(self):
        train = _toy_frame(60)
        step = CategoricalEncoder(
            cfg={"method": "ordinal", "unknown_value": -1}, target_col=TARGET)
        step.fit_transform(train)
        val = _toy_frame(10, seed=4)
        val.loc[0, "f_cat"] = "zzz"   # unseen category
        out = step.transform(val)
        self.assertEqual(out.loc[0, "f_cat"], -1)

    def test_skips_high_cardinality(self):
        df = _toy_frame(60)
        df["f_id"] = [f"id{i}" for i in range(len(df))]
        step = CategoricalEncoder(
            cfg={"method": "onehot", "max_cardinality": 5}, target_col=TARGET)
        res = step.fit_transform(df)
        self.assertIn("f_id", res.params["skipped_high_cardinality"])
        self.assertIn("f_id", res.df.columns)


class TestScaling(unittest.TestCase):
    def test_standard_scaler_zero_mean(self):
        train = _toy_frame(200)
        step = FeatureScaler(cfg={"method": "standard"}, target_col=TARGET)
        res = step.fit_transform(train)
        self.assertAlmostEqual(res.df["f_num"].mean(), 0.0, places=6)
        self.assertNotIn(TARGET, res.params["numeric_columns"])

    def test_transform_requires_fitted_columns(self):
        train = _toy_frame(80)
        step = FeatureScaler(cfg={"method": "minmax"}, target_col=TARGET)
        step.fit_transform(train)
        val = _toy_frame(20, seed=5).drop(columns=["f_num2"])
        with self.assertRaises(PreprocessingError):
            step.transform(val)

    def test_none_method_skips(self):
        train = _toy_frame(50)
        step = FeatureScaler(cfg={"method": "none"}, target_col=TARGET)
        res = step.fit_transform(train)
        self.assertTrue(res.skipped)


class TestLineage(unittest.TestCase):
    def test_records_shapes(self):
        tracker = LineageTracker()
        df = _toy_frame(40)
        tracker.start(df)
        step = DuplicateRemover(cfg={}, target_col=TARGET)
        res = step.fit_transform(df)
        tracker.record(res, df, res.df)
        tracker.finish(res.df)
        d = tracker.as_dict()
        self.assertEqual(d["n_steps"], 1)
        self.assertEqual(d["initial_shape"]["rows"], 40)


class TestPipelineEndToEnd(unittest.TestCase):
    def _cfg(self, **pp) -> dict:
        return {
            "data": {"target_col": TARGET, "processed_dir": "data/processed"},
            "split": {"test_size": 0.2, "val_size": 0.2, "random_state": 42,
                      "time_based": False, "date_col": None},
            "preprocessing": pp,
        }

    def test_full_run_leak_safe_and_numeric(self):
        df = _toy_frame(400)
        df.loc[:19, "f_num"] = np.nan          # missing values
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)  # duplicates
        cfg = self._cfg(
            imputation={"strategy": "median"},
            outliers={"method": "winsorize"},
            encoding={"method": "onehot"},
            scaling={"method": "standard"})
        pipe = PreprocessingPipeline(cfg, target_col=TARGET)
        res = pipe.run(df)

        # all three splits share identical columns (alignment)
        self.assertListEqual(list(res.train.columns), list(res.val.columns))
        self.assertListEqual(list(res.train.columns), list(res.test.columns))
        # no missing values anywhere
        for frame in (res.train, res.val, res.test):
            self.assertEqual(int(frame.isna().sum().sum()), 0)
        # categorical column encoded away
        self.assertNotIn("f_cat", res.train.columns)
        # lineage captured every executed step
        self.assertGreaterEqual(res.lineage["n_applied"], 5)

    def test_report_writes_files(self):
        import tempfile
        df = _toy_frame(200)
        cfg = self._cfg(scaling={"method": "none"})
        res = PreprocessingPipeline(cfg, target_col=TARGET).run(df)
        with tempfile.TemporaryDirectory() as d:
            reporter = PreprocessingReport(d)
            jp, mp = reporter.write(res, TARGET)
            self.assertTrue(os.path.exists(jp))
            self.assertTrue(os.path.exists(mp))
            self.assertIn("steps", res.report)

    def test_empty_frame_raises(self):
        cfg = self._cfg()
        with self.assertRaises(PreprocessingError):
            PreprocessingPipeline(cfg, target_col=TARGET).run(
                pd.DataFrame({TARGET: []}))

    def test_registry_has_all_steps(self):
        for name in ("cleaning", "duplicates", "imputation", "outliers",
                     "encoding", "scaling"):
            self.assertIn(name, STEP_REGISTRY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
