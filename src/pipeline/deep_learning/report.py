"""Reporting for the Deep Learning module.

Turns a :class:`~pipeline.deep_learning.pipeline.DLResult` into the report
suite under ``reports/deep_learning/`` — the same JSON+Markdown+CSV shape as
the ML reports, plus an HTML rendering:

* ``deep_learning_report.json``  — the full machine-readable run record
* ``deep_learning_report.md``    — architecture / training / metrics summary
* ``deep_learning_report.html``  — the Markdown report rendered to HTML
* ``leaderboard.csv`` / ``metrics_summary.csv`` — machine-readable rankings
* ``training_history_<model>.csv`` — per-epoch traces
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone

import pandas as pd

from ingestion.logging_config import get_logger
from pipeline.ml.report import _md_table

log = get_logger("dl.report")


class DLReport:
    """Persist the deep-learning run's report suite."""

    def __init__(self, reports_dir: str = "reports/deep_learning") -> None:
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        self.written: list[str] = []

    # ── entry point ───────────────────────────────────────────────────────────
    def write(self, result) -> list[str]:
        """Write every report; return the list of paths."""
        self._json_report(result)
        md_lines = self._markdown_report(result)
        self._html_report(md_lines)
        self._leaderboard_csv(result)
        self._metrics_summary(result)
        self._history_csvs(result)
        log.info("wrote %d deep-learning reports under %s",
                 len(self.written), self.reports_dir)
        return list(self.written)

    # ── individual reports ────────────────────────────────────────────────────
    def _json_report(self, result) -> None:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": result.data.version,
            "target_col": result.data.target_col,
            "n_features": result.data.n_features,
            "splits": {n: len(t) for n, t in result.data.tensors.items()},
            "models": [t.as_dict() for t in result.trained],
            "failed": {t.name: t.error for t in result.trained if t.failed},
            "leaderboard": result.leaderboard.to_dict(orient="records")
            if result.leaderboard is not None else [],
            "best_model": result.best.name if result.best else None,
            "figures": result.figures,
        }
        self._dump_json("deep_learning_report.json", report)

    def _markdown_report(self, result) -> list[str]:
        lines = ["# Deep Learning Report", "",
                 f"Generated: {datetime.now(timezone.utc).isoformat()}",
                 f"Dataset: feature store {result.data.version} "
                 f"(target `{result.data.target_col}`, "
                 f"{result.data.n_features} features)", ""]

        lines += ["## Training summary", "",
                  "| Model | Status | Params | Device | Epochs | Best epoch "
                  "| Train (s) | Threshold |",
                  "|---|---|---|---|---|---|---|---|"]
        for t in result.trained:
            h = t.history
            lines.append(
                f"| {t.name} | {'FAILED' if t.failed else 'trained'} "
                f"| {t.n_parameters:,} | {t.device} "
                f"| {len(h.epochs) if h else 0} "
                f"| {h.best_epoch if h else '-'} "
                f"| {t.train_seconds:.2f} "
                f"| {t.threshold:.3f} ({t.threshold_method}) |")
        for t in result.trained:
            if t.failed:
                lines += ["", f"**{t.name} failed:** `{t.error}`"]

        for t in result.trained:
            if t.failed:
                continue
            lines += ["", f"## {t.name}", "", "### Architecture",
                      f"`{t.architecture}`", "", "### Hyperparameters",
                      f"`{t.hyperparameters}`"]
            for split, ev in t.evaluations.items():
                lines += ["", f"### Metrics — {split} "
                              f"(threshold {ev.threshold:.3f})",
                          "| Metric | Value |", "|---|---|"]
                lines += [f"| {m} | {v:.4f} |"
                          for m, v in ev.metrics.items()]
                if ev.confusion:
                    (tn, fp), (fn, tp) = ev.confusion
                    lines += ["", f"Confusion: TN={tn} FP={fp} FN={fn} "
                                  f"TP={tp}"]
            if t.permutation_importance is not None:
                lines += ["", "### Top features (permutation)", "",
                          _md_table(t.permutation_importance.head(15))]
            if t.shap_summary:
                top = list(t.shap_summary["mean_abs_shap"].items())[:15]
                lines += ["", "### Top features (SHAP)", "",
                          "| Feature | mean abs SHAP |", "|---|---|"]
                lines += [f"| {feat} | {val:.4f} |" for feat, val in top]

        if result.leaderboard is not None and not result.leaderboard.empty:
            lines += ["", "## Leaderboard (test)", "",
                      _md_table(result.leaderboard)]
        if result.best is not None:
            lines += ["", "## Best model", "",
                      f"- **Network**: {result.best.name}",
                      f"- **Best epoch**: "
                      f"{result.best.history.best_epoch}",
                      f"- **Test ROC-AUC**: "
                      f"{result.best.metric('roc_auc'):.4f}",
                      f"- **Threshold**: {result.best.threshold:.3f} "
                      f"({result.best.threshold_method})"]
        if result.figures:
            lines += ["", "## Figures", ""]
            lines += [f"- `{p}`" for p in result.figures]
        self._dump_md("deep_learning_report.md", lines)
        return lines

    def _html_report(self, md_lines: list[str]) -> None:
        """Minimal Markdown → HTML rendering (headings, tables, code)."""
        body: list[str] = []
        in_table = False
        for line in md_lines:
            stripped = line.strip()
            if stripped.startswith("|"):
                cells = [html.escape(c.strip())
                         for c in stripped.strip("|").split("|")]
                if set("".join(cells)) <= set("-: "):
                    continue                         # separator row
                tag = "td" if in_table else "th"
                if not in_table:
                    body.append("<table>")
                    in_table = True
                body.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>"
                                             for c in cells) + "</tr>")
                continue
            if in_table:
                body.append("</table>")
                in_table = False
            if stripped.startswith("###"):
                body.append(f"<h3>{html.escape(stripped[3:].strip())}</h3>")
            elif stripped.startswith("##"):
                body.append(f"<h2>{html.escape(stripped[2:].strip())}</h2>")
            elif stripped.startswith("#"):
                body.append(f"<h1>{html.escape(stripped[1:].strip())}</h1>")
            elif stripped.startswith("- "):
                body.append(f"<li>{html.escape(stripped[2:])}</li>")
            elif stripped:
                body.append(f"<p>{html.escape(stripped)}</p>")
        if in_table:
            body.append("</table>")
        doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
               "<title>Deep Learning Report</title><style>"
               "body{font-family:sans-serif;margin:2em;max-width:960px}"
               "table{border-collapse:collapse;margin:1em 0}"
               "th,td{border:1px solid #ccc;padding:4px 10px;"
               "text-align:left}th{background:#f0f0f0}"
               "</style></head><body>" + "\n".join(body)
               + "</body></html>")
        path = os.path.join(self.reports_dir, "deep_learning_report.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        self.written.append(path)

    def _leaderboard_csv(self, result) -> None:
        if result.leaderboard is None or result.leaderboard.empty:
            return
        path = os.path.join(self.reports_dir, "leaderboard.csv")
        result.leaderboard.to_csv(path, index=False)
        self.written.append(path)

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

    def _history_csvs(self, result) -> None:
        for t in result.trained:
            if t.failed or not t.history or not t.history.epochs:
                continue
            rows = [{"epoch": e.epoch, "train_loss": e.train_loss,
                     "val_loss": e.val_loss, "lr": e.lr,
                     "seconds": e.seconds, **{f"val_{k}": v
                                              for k, v in
                                              e.val_metrics.items()}}
                    for e in t.history.epochs]
            path = os.path.join(self.reports_dir,
                                f"training_history_{t.name}.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            self.written.append(path)

    # ── plumbing ──────────────────────────────────────────────────────────────
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
