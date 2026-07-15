"""Unit tests for the Feature Engineering module
(``src/pipeline/feature_engineering``).

Run with the project's venv, no extra deps required::

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Make ``src`` importable exactly like the run_*.py entry points do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline.feature_engineering import (  # noqa: E402
    DimensionalityReduction, FeatureEngineeringError,
    FeatureEngineeringPipeline, FeatureEngineeringReport, FeatureGeneration,
    FeatureImportance, FeatureSelection, FeatureStore, FEATURE_STEPS,
    MulticollinearityFilter, STEP_REGISTRY,
)
from pipeline.feature_engineering.lineage import FeatureLineageTracker  # noqa: E402

TARGET = "Bankrupt?"
RNG = np.random.default_rng(42)


def make_df(n=300, seed=42) -> pd.DataFrame:
    """Synthetic frame with signal, redundancy, and skew — like the real data."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    f1 = y * 2.0 + rng.normal(0, 1, n)               # informative
    f2 = -y * 1.5 + rng.normal(0, 1, n)              # informative
    f3 = f1 * 0.999 + rng.normal(0, 0.001, n)        # ~duplicate of f1
    f4 = rng.normal(0, 1, n)                         # noise
    f5 = np.exp(rng.normal(0, 1.5, n))               # skewed
    f6 = np.full(n, 3.14)                            # constant
    return pd.DataFrame({TARGET: y, "f1": f1, "f2": f2, "f3": f3,
                         "f4": f4, "f5": f5, "f6": f6})


class TestGeneration(unittest.TestCase):
    def test_generates_and_replays_same_schema(self):
        df = make_df()
        step = FeatureGeneration(cfg={"base_top_k": 4}, target_col=TARGET)
        res = step.fit_transform(df)
        self.assertGreater(len(res.generated), 0)
        # val/test replay produces the identical column set
        val = step.transform(make_df(seed=7))
        self.assertEqual(list(res.df.columns), list(val.columns))
        # no NaN/inf leaks into generated columns
        gen = res.df[res.generated]
        self.assertFalse(gen.isna().any().any())
        self.assertTrue(np.isfinite(gen.to_numpy()).all())

    def test_disabled_families(self):
        cfg = {"log_transform": False, "ratios": False, "interactions": False,
               "differences": False, "polynomial": False}
        res = FeatureGeneration(cfg=cfg, target_col=TARGET).fit_transform(make_df())
        self.assertEqual(res.generated, [])


class TestMulticollinearity(unittest.TestCase):
    def test_drops_near_duplicate(self):
        df = make_df()
        step = MulticollinearityFilter(cfg={"correlation_threshold": 0.95},
                                       target_col=TARGET)
        res = step.fit_transform(df)
        # exactly one of the f1/f3 duplicate pair is dropped
        self.assertEqual(len({"f1", "f3"} & set(res.removed)), 1)
        self.assertIn(TARGET, res.df.columns)

    def test_vif_reduces_below_threshold(self):
        df = make_df()
        step = MulticollinearityFilter(
            cfg={"correlation_filter": False, "vif_threshold": 10.0},
            target_col=TARGET)
        res = step.fit_transform(df)
        final_vif = res.stats["final_max_vif"]
        if final_vif is not None:
            self.assertLessEqual(final_vif, 10.0 + 1e-6)

    def test_transform_applies_same_drops(self):
        df = make_df()
        step = MulticollinearityFilter(target_col=TARGET)
        res = step.fit_transform(df)
        other = step.transform(make_df(seed=9))
        self.assertEqual(list(res.df.columns), list(other.columns))


class TestSelection(unittest.TestCase):
    def test_drops_constant_and_keeps_signal(self):
        df = make_df()
        step = FeatureSelection(cfg={"rfe": False, "model_based": False},
                                target_col=TARGET)
        res = step.fit_transform(df)
        self.assertIn("f6", res.removed)          # constant dropped
        self.assertIn("f1", res.selected)         # informative kept
        self.assertIn(TARGET, res.df.columns)

    def test_rfe_and_model_based_bound_features(self):
        df = make_df()
        step = FeatureSelection(cfg={"rfe_n_features": 3, "top_k": 5},
                                target_col=TARGET)
        res = step.fit_transform(df)
        self.assertLessEqual(len(res.selected), 5)

    def test_missing_target_raises(self):
        df = make_df().drop(columns=[TARGET])
        with self.assertRaises(FeatureEngineeringError):
            FeatureSelection(target_col=TARGET).fit_transform(df)


class TestReduction(unittest.TestCase):
    def test_pca_append_hits_variance_target(self):
        df = make_df()
        step = DimensionalityReduction(
            cfg={"method": "pca", "explained_variance": 0.9, "mode": "append"},
            target_col=TARGET)
        res = step.fit_transform(df)
        self.assertGreaterEqual(res.stats["explained_variance"], 0.9)
        self.assertTrue(all(c.startswith("pca__") for c in res.generated))
        # append mode keeps originals
        self.assertIn("f1", res.df.columns)
        # transform yields the same schema
        val = step.transform(make_df(seed=3))
        self.assertEqual(list(res.df.columns), list(val.columns))

    def test_svd_replace_mode(self):
        df = make_df()
        step = DimensionalityReduction(
            cfg={"method": "svd", "n_components": 3, "mode": "replace"},
            target_col=TARGET)
        res = step.fit_transform(df)
        self.assertNotIn("f1", res.df.columns)
        self.assertEqual(len(res.generated), 3)
        self.assertIn(TARGET, res.df.columns)

    def test_method_none_skips(self):
        res = DimensionalityReduction(cfg={"method": "none"},
                                      target_col=TARGET).fit_transform(make_df())
        self.assertTrue(res.skipped)


class TestImportance(unittest.TestCase):
    def test_identity_and_scores(self):
        df = make_df()
        step = FeatureImportance(cfg={"n_estimators": 50, "shap": False},
                                 target_col=TARGET)
        res = step.fit_transform(df)
        pd.testing.assert_frame_equal(res.df, df)   # analysis-only
        self.assertIsNotNone(step.importances)
        # informative features rank above pure noise
        table = step.importances
        self.assertLess(table.loc["f1", "mean_rank"],
                        table.loc["f6", "mean_rank"])
        # transform is identity
        pd.testing.assert_frame_equal(step.transform(df), df)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_versioning_roundtrip(self):
        store = FeatureStore(self.dir)
        df = make_df(50)
        rec1 = store.save({"train": df}, metadata={"note": "first"})
        rec2 = store.save({"train": df}, metadata={"note": "second"})
        self.assertEqual((rec1["version"], rec2["version"]), ("v001", "v002"))
        self.assertEqual(store.latest_version(), "v002")
        splits, meta = store.load()
        self.assertEqual(meta["note"], "second")
        pd.testing.assert_frame_equal(splits["train"], df)

    def test_empty_store_raises(self):
        with self.assertRaises(FileNotFoundError):
            FeatureStore(self.dir).load()


class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cfg(self):
        return {"feature_engineering": {
            "fail_fast": True,
            "eda_report_path": os.path.join(self.tmp, "absent.json"),
            "generation": {"base_top_k": 4},
            "selection": {"top_k": 10, "rfe_n_features": 5,
                          "model_n_estimators": 50},
            "importance": {"n_estimators": 50, "shap": False},
            "store": {"enabled": True, "dir": os.path.join(self.tmp, "store")},
        }}

    def test_full_run(self):
        train, val, test = make_df(300), make_df(80, seed=1), make_df(80, seed=2)
        pipe = FeatureEngineeringPipeline(self._cfg(), target_col=TARGET)
        result = pipe.run(train, val, test)
        # identical schema across splits, target preserved
        self.assertEqual(list(result.train.columns), list(result.val.columns))
        self.assertEqual(list(result.train.columns), list(result.test.columns))
        self.assertIn(TARGET, result.train.columns)
        # all five steps recorded in lineage
        self.assertEqual(result.lineage["n_steps"], len(FEATURE_STEPS))
        # store persisted a version
        self.assertIsNotNone(result.store_record)
        self.assertEqual(result.store_record["version"], "v001")
        # reports write cleanly
        reporter = FeatureEngineeringReport(os.path.join(self.tmp, "reports"))
        json_path, md_path = reporter.write(result, TARGET)
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))

    def test_empty_split_raises(self):
        pipe = FeatureEngineeringPipeline(self._cfg(), target_col=TARGET)
        with self.assertRaises(FeatureEngineeringError):
            pipe.run(make_df(), make_df().iloc[:0], make_df())

    def test_fail_fast_off_continues(self):
        cfg = self._cfg()
        cfg["feature_engineering"]["fail_fast"] = False
        train = make_df().drop(columns=[TARGET])
        train[TARGET] = 0  # single-class target breaks supervised steps
        pipe = FeatureEngineeringPipeline(cfg, target_col=TARGET)
        result = pipe.run(train, make_df(80, seed=1), make_df(80, seed=2))
        self.assertIsNotNone(result)


class TestRegistryAndLineage(unittest.TestCase):
    def test_registry_names(self):
        self.assertEqual(
            set(STEP_REGISTRY),
            {"generation", "multicollinearity", "selection",
             "reduction", "importance"})

    def test_lineage_math(self):
        tracker = FeatureLineageTracker()
        df = make_df()
        tracker.start(df)
        step = FeatureGeneration(cfg={"base_top_k": 3}, target_col=TARGET)
        res = step.fit_transform(df)
        tracker.record(res, df, res.df)
        tracker.finish(res.df)
        d = tracker.as_dict()
        self.assertEqual(d["n_steps"], 1)
        self.assertEqual(d["cols_added"],
                         res.df.shape[1] - df.shape[1])


if __name__ == "__main__":
    unittest.main()
