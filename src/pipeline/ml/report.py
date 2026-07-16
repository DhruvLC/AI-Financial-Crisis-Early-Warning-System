"""Reporting for the ML module.

Turns a :class:`~pipeline.ml.pipeline.MLResult` into the full report suite
under ``reports/ml/`` — the same JSON+Markdown+CSV shape as the preprocessing
and feature-engineering reports:

* ``training_report.md`` / ``ml_report.json`` — per-model training summary
* ``evaluation_report.md`` — the full metric suite per model per split
* ``leaderboard.csv`` / ``metrics_summary.csv`` — machine-readable rankings
* ``best_model_report.md`` — the winner in detail
* ``feature_importance_<model>.csv`` + ``feature_importance_report.md``
* ``shap_report.md`` (when SHAP produced values)
* ``model_card_<model>.md`` — one model card per trained model
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from ingestion.logging_config import get_logger

log = get_logger("ml.report")


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a frame as a GitHub-flavoured Markdown table (no tabulate)."""
    def fmt(v) -> str:
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


class MLReport:
    """Persist the ML run's report suite."""

    def __init__(self, reports_dir: str = "reports/ml") -> None:
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        self.written: list[str] = []

    # ── entry point ───────────────────────────────────────────────────────────
    def write(self, result) -> list[str]:
        """Write every report; return the list of paths."""
        self._json_report(result)
        self._training_report(result)
        self._evaluation_report(result)
        self._leaderboard(result)
        self._metrics_summary(result)
        self._best_model_report(result)
        self._importance_reports(result)
        self._shap_report(result)
        self._model_cards(result)
        log.info("wrote %d ML reports under %s", len(self.written),
                 self.reports_dir)
        return list(self.written)

    # ── individual reports ────────────────────────────────────────────────────
    def _json_report(self, result) -> None:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": result.dataset.version,
            "target_col": result.dataset.target_col,
            "n_features": len(result.dataset.features),
            "splits": {n: len(d) for n, d in result.dataset.splits.items()},
            "models": [t.as_dict() for t in result.trained],
            "failed": {t.name: t.error for t in result.trained if t.failed},
            "leaderboard": result.leaderboard.to_dict(orient="records")
            if result.leaderboard is not None else [],
            "best_model": result.best.name if result.best else None,
            "figures": result.figures,
        }
        self._dump_json("ml_report.json", report)

    def _training_report(self, result) -> None:
        lines = self._header("Training Report", result)
        lines += ["| Model | Status | Train (s) | CV mean | Threshold |",
                  "|---|---|---|---|---|"]
        for t in result.trained:
            cv = t.cv_scores.get("mean")
            cv_txt = f"{cv:.4f}" if cv is not None else "n/a"
            lines.append(
                f"| {t.name} | {'FAILED' if t.failed else 'trained'} "
                f"| {t.train_seconds:.2f} | {cv_txt} "
                f"| {t.threshold:.3f} ({t.threshold_method}) |")
        for t in result.trained:
            if t.failed:
                lines += ["", f"**{t.name} failed:** `{t.error}`"]
        for t in result.trained:
            if not t.failed and t.tuning:
                lines += ["", f"### {t.name} — tuning",
                          f"- method: {t.tuning.get('method')}",
                          f"- best CV {t.tuning.get('scoring')}: "
                          f"{t.tuning.get('best_score'):.4f}",
                          f"- best params: `{t.tuning.get('best_params')}`"]
        self._dump_md("training_report.md", lines)

    def _evaluation_report(self, result) -> None:
        lines = self._header("Evaluation Report", result)
        for t in result.trained:
            if t.failed:
                continue
            lines += ["", f"## {t.name}"]
            for split, ev in t.evaluations.items():
                lines += [f"", f"### {split} (threshold {ev.threshold:.3f})",
                          "| Metric | Value |", "|---|---|"]
                lines += [f"| {m} | {v:.4f} |"
                          for m, v in ev.metrics.items()]
                if ev.confusion:
                    (tn, fp), (fn, tp) = ev.confusion
                    lines += ["", f"Confusion: TN={tn} FP={fp} FN={fn} "
                                  f"TP={tp}"]
        self._dump_md("evaluation_report.md", lines)

    def _leaderboard(self, result) -> None:
        if result.leaderboard is None or result.leaderboard.empty:
            return
        path = os.path.join(self.reports_dir, "leaderboard.csv")
        result.leaderboard.to_csv(path, index=False)
        self.written.append(path)
        lines = self._header("Leaderboard", result)
        lines += ["Ranked by test ROC-AUC (ties broken by F1, recall, "
                  "PR-AUC).", "",
                  _md_table(result.leaderboard)]
        self._dump_md("leaderboard.md", lines)

    def _metrics_summary(self, result) -> None:
        rows = []
        for t in result.trained:
            if t.failed:
                continue
            for split, ev in t.evaluations.items():
                rows.append({"model": t.name, "split": split,
                             "threshold": ev.threshold, **ev.metrics})
        if rows:
            path = os.path.join(self.reports_dir, "metrics_summary.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            self.written.append(path)

    def _best_model_report(self, result) -> None:
        best = result.best
        if best is None:
            return
        lines = self._header("Best Model Report", result)
        lines += [f"## {best.name}", "",
                  f"- Dataset version: {result.dataset.version}",
                  f"- Threshold: {best.threshold:.3f} "
                  f"({best.threshold_method})",
                  f"- Training time: {best.train_seconds:.2f}s",
                  f"- Hyperparameters: `{best.hyperparameters}`", ""]
        for split, ev in best.evaluations.items():
            lines += [f"### {split}", "| Metric | Value |", "|---|---|"]
            lines += [f"| {m} | {v:.4f} |" for m, v in ev.metrics.items()]
            lines.append("")
        if best.feature_importance is not None:
            lines += ["### Top features", "",
                      _md_table(best.feature_importance.head(15))]
        self._dump_md("best_model_report.md", lines)

    def _importance_reports(self, result) -> None:
        lines = self._header("Feature Importance Report", result)
        for t in result.trained:
            for kind, df in (("native", t.feature_importance),
                             ("permutation", t.permutation_importance)):
                if df is None or df.empty:
                    continue
                path = os.path.join(
                    self.reports_dir,
                    f"feature_importance_{t.name}_{kind}.csv")
                df.to_csv(path, index=False)
                self.written.append(path)
                lines += ["", f"## {t.name} — {kind}", "",
                          _md_table(df.head(15))]
        self._dump_md("feature_importance_report.md", lines)

    def _shap_report(self, result) -> None:
        with_shap = [t for t in result.trained if t.shap_summary]
        if not with_shap:
            return
        lines = self._header("SHAP Report", result)
        for t in with_shap:
            top = list(t.shap_summary["mean_abs_shap"].items())[:15]
            lines += ["", f"## {t.name} "
                          f"({t.shap_summary['n_samples']} samples)", "",
                      "| Feature | mean |SHAP| |", "|---|---|"]
            lines += [f"| {feat} | {val:.4f} |" for feat, val in top]
        self._dump_md("shap_report.md", lines)

    def _model_cards(self, result) -> None:
        for t in result.trained:
            if t.failed:
                continue
            test = t.evaluations.get("test")
            lines = [f"# Model Card — {t.name}", "",
                     f"Generated: {datetime.now(timezone.utc).isoformat()}",
                     "", "## Overview",
                     f"- **Task**: binary financial-crisis (bankruptcy) "
                     f"prediction",
                     f"- **Algorithm**: {t.name}",
                     f"- **Dataset version**: {result.dataset.version}",
                     f"- **Features**: {len(result.dataset.features)}",
                     f"- **Decision threshold**: {t.threshold:.3f} "
                     f"({t.threshold_method})", "",
                     "## Hyperparameters", f"`{t.hyperparameters}`", "",
                     "## Performance (test)"]
            if test:
                lines += ["| Metric | Value |", "|---|---|"]
                lines += [f"| {m} | {v:.4f} |"
                          for m, v in test.metrics.items()]
            lines += ["", "## Intended use & limitations",
                      "- Early-warning signal for financial distress; "
                      "not a standalone credit decisioning system.",
                      "- Trained on a heavily imbalanced sample; monitor "
                      "calibration and drift before production use."]
            self._dump_md(f"model_card_{t.name}.md", lines)

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _header(self, title: str, result) -> list[str]:
        return [f"# {title}", "",
                f"Generated: {datetime.now(timezone.utc).isoformat()}",
                f"Dataset: feature store {result.dataset.version} "
                f"(target `{result.dataset.target_col}`)", ""]

    def _dump_md(self, name: str, lines: list[str]) -> None:
        path = os.path.join(self.reports_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.written.append(path)

    def _dump_json(self, name: str, payload: dict) -> None:
        path = os.path.join(self.reports_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        self.written.append(path)
