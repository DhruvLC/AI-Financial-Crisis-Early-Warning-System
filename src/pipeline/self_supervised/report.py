"""Reporting for the Self-Supervised Learning module.

Extends :class:`pipeline.deep_learning.report.DLReport` — reusing its
JSON / Markdown / HTML / CSV writers wholesale — renaming the artefacts
to ``self_supervised_report.*`` and appending the SSL-specific sections:
augmentation pipeline, loss configuration, representation exports +
embedding statistics, and KNN probe metrics. Reports land under
``reports/self_supervised/``:

* ``self_supervised_report.json`` / ``.md`` / ``.html``
* ``leaderboard.csv`` / ``metrics_summary.csv``
* ``training_history_<encoder>.csv`` — per-epoch pretraining traces
"""
from __future__ import annotations

import json
import os

from ingestion.logging_config import get_logger
from pipeline.deep_learning.report import DLReport

__all__ = ["SSLReport"]

log = get_logger("ssl.report")


class SSLReport(DLReport):
    """Persist the self-supervised run's report suite."""

    def __init__(self,
                 reports_dir: str = "reports/self_supervised") -> None:
        super().__init__(reports_dir)

    # ── entry point ───────────────────────────────────────────────────────────
    def write(self, result) -> list[str]:
        """Write every report; return the list of paths."""
        self._json_report(result)
        self._adopt("deep_learning_report.json",
                    "self_supervised_report.json",
                    lambda payload: {
                        **payload,
                        "augmentations": result.augmentations,
                        "loss": result.loss_config,
                        "representations": self._representation_dict(result),
                    })

        md_lines = self._markdown_report(result)
        md_lines[0] = "# Self-Supervised Learning Report"
        md_lines += self._ssl_sections(result)
        self._dump_md("self_supervised_report.md", md_lines)
        old_md = os.path.join(self.reports_dir, "deep_learning_report.md")
        self.written.remove(old_md)
        os.remove(old_md)

        self._html_report(md_lines)
        self._adopt("deep_learning_report.html",
                    "self_supervised_report.html")

        self._leaderboard_csv(result)
        self._metrics_summary(result)
        self._history_csvs(result)
        log.info("wrote %d self-supervised reports under %s",
                 len(self.written), self.reports_dir)
        return list(self.written)

    # ── SSL-specific sections ─────────────────────────────────────────────────
    @staticmethod
    def _representation_dict(result) -> dict:
        return {t.name: {"paths": t.representations,
                         "embedding_stats": t.embedding_stats,
                         "knn_metrics": {s: e.metrics for s, e in
                                         t.knn_evaluations.items()}}
                for t in result.trained if not t.failed}

    def _ssl_sections(self, result) -> list[str]:
        lines = ["", "## Augmentation pipeline", "",
                 "| Augmentation | Parameters |", "|---|---|"]
        for aug in result.augmentations:
            aug = dict(aug)
            name = aug.pop("name", "?")
            lines.append(f"| {name} | `{aug}` |")
        lines += ["", "## Contrastive loss", "",
                  f"`{result.loss_config}`"]
        for t in result.trained:
            if t.failed:
                continue
            lines += ["", f"## Representations — {t.name}", "",
                      "| Split | Path | Samples | Dim | Mean L2 norm |",
                      "|---|---|---|---|---|"]
            for split, path in t.representations.items():
                st = t.embedding_stats.get(split, {})
                lines.append(f"| {split} | `{path}` "
                             f"| {st.get('n_samples', '-')} "
                             f"| {st.get('dim', '-')} "
                             f"| {st.get('mean_l2_norm', float('nan')):.3f}"
                             f" |")
            if t.knn_evaluations:
                lines += ["", f"### KNN probe — {t.name}", "",
                          "| Split | ROC-AUC | F1 | Accuracy |",
                          "|---|---|---|---|"]
                for split, ev in t.knn_evaluations.items():
                    m = ev.metrics
                    lines.append(
                        f"| {split} "
                        f"| {m.get('roc_auc', float('nan')):.4f} "
                        f"| {m.get('f1', float('nan')):.4f} "
                        f"| {m.get('accuracy', float('nan')):.4f} |")
        return lines

    # ── plumbing (same adoption pattern as the transformer report) ────────────
    def _adopt(self, old_name: str, new_name: str,
               transform=None) -> None:
        """Rename an artefact the parent writer produced (optionally
        transforming JSON payloads) and fix the written-paths list."""
        old = os.path.join(self.reports_dir, old_name)
        new = os.path.join(self.reports_dir, new_name)
        if transform is not None:
            with open(old, encoding="utf-8") as f:
                payload = transform(json.load(f))
            with open(new, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.remove(old)
        else:
            os.replace(old, new)
        self.written = [new if p == old else p for p in self.written]
