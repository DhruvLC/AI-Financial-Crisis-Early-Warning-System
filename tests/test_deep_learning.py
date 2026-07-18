"""Unit tests for the Deep Learning module (``src/pipeline/deep_learning``).

Run with the project's venv (needs torch)::

    .venv/bin/python -m unittest tests.test_deep_learning -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

# Make ``src`` importable exactly like the run_*.py entry points do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline.deep_learning import (  # noqa: E402
    DLDataLoader, DLError, DLPipeline, DLPredictor, DLModelRegistry,
    DLReport, EarlyStopping, NETWORK_REGISTRY, TabularDataset, Trainer,
    build_loss, build_network, build_optimizer, build_scheduler,
    load_checkpoint, predict_proba, seed_all,
)
from pipeline.deep_learning.base import TrainedNetwork  # noqa: E402
from pipeline.feature_engineering.store import FeatureStore  # noqa: E402

TARGET = "Bankrupt?"


def make_df(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """Synthetic frame with signal + noise — like the engineered data."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    return pd.DataFrame({
        "f1": y * 2.0 + rng.normal(0, 1, n),
        "f2": -y * 1.5 + rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
        "f4": rng.normal(0, 1, n),
        TARGET: y,
    })


def make_splits() -> dict[str, pd.DataFrame]:
    return {"train": make_df(300, 1), "val": make_df(120, 2),
            "test": make_df(120, 3)}


def small_cfg(tmp: str) -> dict:
    """Minimal fast config used by the pipeline tests."""
    return {
        "data": {"target_col": TARGET},
        "feature_engineering": {"store": {"dir": os.path.join(tmp,
                                                              "features")}},
        "deep_learning": {
            "fail_fast": True,
            "random_state": 42,
            "networks": ["mlp"],
            "data": {"batch_size": 32},
            "model_params": {"mlp": {"hidden_layers": [8],
                                     "dropout": 0.0}},
            "training": {"epochs": 3, "device": "cpu", "log_every": 10},
            "early_stopping": {"patience": 5},
            "checkpoint": {"dir": os.path.join(tmp, "models")},
            "explainability": {"permutation": True,
                               "permutation_repeats": 1, "shap": False},
            "output": {
                "models_dir": os.path.join(tmp, "models"),
                "reports_dir": os.path.join(tmp, "reports"),
                "figures_dir": os.path.join(tmp, "reports", "figures"),
            },
        },
    }


class TmpDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="dl_test_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


# ── data loading ────────────────────────────────────────────────────────────
class TestDataLoading(TmpDirTest):
    def _store(self) -> str:
        store_dir = os.path.join(self.tmp, "features")
        splits = make_splits()
        FeatureStore(store_dir).save(
            splits, {"target_col": TARGET,
                     "features": [c for c in splits["train"].columns
                                  if c != TARGET]})
        return store_dir

    def test_tabular_dataset(self) -> None:
        df = make_df(50)
        ds = TabularDataset(df.drop(columns=[TARGET]), df[TARGET])
        self.assertEqual(len(ds), 50)
        X, y = ds[0]
        self.assertEqual(X.dtype, torch.float32)
        self.assertEqual(X.shape, (4,))
        self.assertIn(float(y), (0.0, 1.0))

    def test_tabular_dataset_rejects_empty_and_nan(self) -> None:
        df = make_df(10)
        with self.assertRaises(DLError):
            TabularDataset(df.drop(columns=[TARGET]).iloc[:0],
                           df[TARGET].iloc[:0])
        bad = df.drop(columns=[TARGET]).copy()
        bad.iloc[0, 0] = np.nan
        with self.assertRaises(DLError):
            TabularDataset(bad, df[TARGET])

    def test_loader_builds_batches_and_pos_weight(self) -> None:
        data = DLDataLoader(self._store(), TARGET, batch_size=32).load()
        self.assertEqual(data.n_features, 4)
        self.assertEqual(set(data.loaders), {"train", "val", "test"})
        X, y = next(iter(data.loaders["train"]))
        self.assertEqual(X.shape, (32, 4))
        self.assertGreater(data.pos_weight, 1.0)   # imbalanced target

    def test_loader_rejects_bad_batch_size(self) -> None:
        with self.assertRaises(DLError):
            DLDataLoader(self.tmp, TARGET, batch_size=0)

    def test_shuffle_reproducible(self) -> None:
        store = self._store()
        first = [y.sum().item() for _, y in
                 DLDataLoader(store, TARGET, batch_size=64,
                              seed=7).load().loaders["train"]]
        second = [y.sum().item() for _, y in
                  DLDataLoader(store, TARGET, batch_size=64,
                               seed=7).load().loaders["train"]]
        self.assertEqual(first, second)


# ── models ──────────────────────────────────────────────────────────────────
class TestNetworks(unittest.TestCase):
    def test_registry_contents(self) -> None:
        self.assertEqual(set(NETWORK_REGISTRY),
                         {"mlp", "deep_fc", "residual", "wide_deep"})

    def test_forward_pass_all_networks(self) -> None:
        X = torch.randn(16, 10)
        for name in NETWORK_REGISTRY:
            model = build_network(name, 10)
            model.eval()
            out = model(X)
            self.assertEqual(out.shape, (16,), name)
            self.assertTrue(torch.isfinite(out).all(), name)

    def test_configurable_params(self) -> None:
        model = build_network("mlp", 6,
                              {"hidden_layers": [12, 5],
                               "activation": "gelu", "dropout": 0.5,
                               "batch_norm": False,
                               "initialization": "xavier"})
        self.assertEqual(model.params["hidden_layers"], [12, 5])
        out = model.eval()(torch.randn(4, 6))
        self.assertEqual(out.shape, (4,))

    def test_all_activations(self) -> None:
        for act in ("relu", "gelu", "elu", "leakyrelu", "selu"):
            model = build_network("mlp", 4, {"activation": act,
                                             "hidden_layers": [6]})
            self.assertEqual(model.eval()(torch.randn(3, 4)).shape, (3,))

    def test_invalid_config_raises(self) -> None:
        with self.assertRaises(DLError):
            build_network("nope", 4)
        with self.assertRaises(DLError):
            build_network("mlp", 4, {"activation": "tanhh"})
        with self.assertRaises(DLError):
            build_network("mlp", 0)
        with self.assertRaises(DLError):
            build_network("mlp", 4, {"initialization": "bogus"})


# ── losses / optimizers / schedulers / early stopping ───────────────────────
class TestTrainingComponents(unittest.TestCase):
    def test_losses(self) -> None:
        logits = torch.randn(8)
        targets = (torch.rand(8) > 0.5).float()
        for name in ("bce", "bce_with_logits", "weighted_bce", "focal"):
            loss = build_loss({"name": name}, pos_weight=3.0)
            val = loss(logits, targets)
            self.assertTrue(torch.isfinite(val), name)
        with self.assertRaises(DLError):
            build_loss({"name": "hinge"})

    def test_optimizers(self) -> None:
        model = build_network("mlp", 4, {"hidden_layers": [4]})
        for name in ("adam", "adamw", "sgd", "rmsprop"):
            opt = build_optimizer(model, {"name": name, "lr": 0.01})
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)
        with self.assertRaises(DLError):
            build_optimizer(model, {"name": "lion"})

    def test_schedulers(self) -> None:
        model = build_network("mlp", 4, {"hidden_layers": [4]})
        opt = build_optimizer(model, {"name": "adam"})
        for name, plateau in (("plateau", True), ("cosine", False),
                              ("step", False), ("exponential", False)):
            sched, is_plateau = build_scheduler(opt, {"name": name}, 10)
            self.assertIsNotNone(sched, name)
            self.assertEqual(is_plateau, plateau, name)
        sched, _ = build_scheduler(opt, {"name": "none"}, 10)
        self.assertIsNone(sched)
        with self.assertRaises(DLError):
            build_scheduler(opt, {"name": "cyclic"}, 10)

    def test_early_stopping(self) -> None:
        model = build_network("mlp", 4, {"hidden_layers": [4]})
        es = EarlyStopping(patience=2, min_delta=0.0)
        self.assertFalse(es.step(1, 1.0, model))
        self.assertFalse(es.step(2, 0.5, model))   # improvement
        self.assertFalse(es.step(3, 0.6, model))   # 1 bad epoch
        self.assertTrue(es.step(4, 0.7, model))    # patience exhausted
        self.assertEqual(es.best_epoch, 2)
        es.restore(model)                          # restores without error


# ── trainer ─────────────────────────────────────────────────────────────────
class TestTrainer(TmpDirTest):
    def _loaders(self):
        data_dir = os.path.join(self.tmp, "features")
        splits = make_splits()
        FeatureStore(data_dir).save(
            splits, {"target_col": TARGET,
                     "features": [c for c in splits["train"].columns
                                  if c != TARGET]})
        return DLDataLoader(data_dir, TARGET, batch_size=32).load()

    def test_training_loop_and_checkpoints(self) -> None:
        seed_all(0)
        data = self._loaders()
        model = build_network("mlp", data.n_features,
                              {"hidden_layers": [8], "dropout": 0.0})
        trainer = Trainer({"training": {"epochs": 3, "device": "cpu"},
                           "checkpoint": {"dir": self.tmp}})
        history, ckpts = trainer.fit("mlp", model, data.loaders["train"],
                                     data.loaders["val"])
        self.assertEqual(len(history.epochs), 3)
        self.assertGreater(history.best_epoch, 0)
        for tag in ("best", "last"):
            self.assertTrue(os.path.exists(ckpts[tag]), tag)
        state = load_checkpoint(ckpts["best"])
        self.assertIn("model_state", state)

    def test_resume_training(self) -> None:
        seed_all(0)
        data = self._loaders()
        model = build_network("mlp", data.n_features,
                              {"hidden_layers": [8], "dropout": 0.0})
        trainer = Trainer({"training": {"epochs": 2, "device": "cpu"},
                           "checkpoint": {"dir": self.tmp}})
        _, ckpts = trainer.fit("mlp", model, data.loaders["train"],
                               data.loaders["val"])
        trainer2 = Trainer({"training": {"epochs": 4, "device": "cpu"},
                            "checkpoint": {"dir": self.tmp}})
        history, _ = trainer2.fit("mlp", model, data.loaders["train"],
                                  data.loaders["val"],
                                  resume_from=ckpts["last"])
        self.assertEqual(history.epochs[0].epoch, 3)  # resumed after 2

    def test_corrupt_checkpoint_raises(self) -> None:
        path = os.path.join(self.tmp, "bad.pt")
        with open(path, "wb") as f:
            f.write(b"not a checkpoint")
        with self.assertRaises(DLError):
            load_checkpoint(path)
        with self.assertRaises(DLError):
            load_checkpoint(os.path.join(self.tmp, "missing.pt"))


# ── prediction & evaluation ─────────────────────────────────────────────────
class TestPrediction(unittest.TestCase):
    def test_predict_proba_shape_and_range(self) -> None:
        model = build_network("mlp", 5, {"hidden_layers": [6]})
        proba = predict_proba(model.eval(),
                              np.random.default_rng(0)
                              .normal(size=(40, 5)).astype(np.float32))
        self.assertEqual(proba.shape, (40,))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_predict_rejects_bad_input(self) -> None:
        model = build_network("mlp", 3, {"hidden_layers": [4]})
        with self.assertRaises(DLError):
            predict_proba(model, np.empty((0, 3), dtype=np.float32))
        bad = np.ones((4, 3), dtype=np.float32)
        bad[0, 0] = np.inf
        with self.assertRaises(DLError):
            predict_proba(model, bad)


# ── pipeline end-to-end (synthetic) ─────────────────────────────────────────
class TestPipeline(TmpDirTest):
    def _run(self):
        cfg = small_cfg(self.tmp)
        splits = make_splits()
        FeatureStore(cfg["feature_engineering"]["store"]["dir"]).save(
            splits, {"target_col": TARGET,
                     "features": [c for c in splits["train"].columns
                                  if c != TARGET]})
        return DLPipeline(cfg).run(), cfg

    def test_full_pipeline(self) -> None:
        result, cfg = self._run()
        self.assertEqual(len(result.trained), 1)
        best = result.best
        self.assertFalse(best.failed)
        self.assertIn("test", best.evaluations)
        self.assertIn("roc_auc", best.evaluations["test"].metrics)
        self.assertIsNotNone(best.permutation_importance)
        self.assertGreater(len(result.figures), 0)
        self.assertGreater(len(result.reports), 0)

        # registry artefact suite
        mdir = cfg["deep_learning"]["output"]["models_dir"]
        for fname in ("best_model.pt", "last_model.pt", "registry.json",
                      "training_config.json", "metrics.json",
                      "history.json", "feature_metadata.json"):
            self.assertTrue(os.path.exists(os.path.join(mdir, fname)),
                            fname)

        # reports
        rdir = cfg["deep_learning"]["output"]["reports_dir"]
        for fname in ("deep_learning_report.json",
                      "deep_learning_report.md",
                      "deep_learning_report.html", "leaderboard.csv",
                      "metrics_summary.csv"):
            self.assertTrue(os.path.exists(os.path.join(rdir, fname)),
                            fname)

        # predictor round-trip on the registered best model
        predictor = DLPredictor(mdir, device="cpu").load()
        X = make_df(30, 9).drop(columns=[TARGET])
        proba = predictor.predict_proba(X)
        self.assertEqual(proba.shape, (30,))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        labels = predictor.predict(X)
        self.assertTrue(set(np.unique(labels)) <= {0, 1})

    def test_predictor_rejects_missing_features(self) -> None:
        _, cfg = self._run()
        predictor = DLPredictor(
            cfg["deep_learning"]["output"]["models_dir"],
            device="cpu").load()
        with self.assertRaises(DLError):
            predictor.predict_proba(pd.DataFrame({"f1": [0.1]}))


# ── registry & report units ─────────────────────────────────────────────────
class TestRegistryAndReport(TmpDirTest):
    def test_corrupt_registry_raises(self) -> None:
        reg = DLModelRegistry(self.tmp)
        with open(reg.registry_path, "w") as f:
            f.write("{broken json")
        with self.assertRaises(DLError):
            reg.entries()

    def test_empty_registry_best_raises(self) -> None:
        with self.assertRaises(DLError):
            DLModelRegistry(self.tmp).best_checkpoint()


if __name__ == "__main__":
    unittest.main()
