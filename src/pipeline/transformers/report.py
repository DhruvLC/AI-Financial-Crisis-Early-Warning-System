"""Reporting for the Transformer Models module.

Extends :class:`pipeline.deep_learning.report.DLReport` — reusing its
JSON / Markdown / HTML / CSV writers wholesale — renaming the artefacts to
``transformer_report.*`` and appending the transformer-specific sections:
attention analysis per model and the comparison with the classical ML and
deep-learning leaderboards. Reports land under ``reports/transformers/``:

* ``transformer_report.json`` / ``.md`` / ``.html``
* ``leaderboard.csv`` / ``metrics_summary.csv``
* ``training_history_<model>.csv`` — per-epoch traces
"""
from __future__ import annotations

import json
import os

import pandas as pd

from ingestion.logging_config import get_logger
from pipeline.deep_learning.report import DLReport
from pipeline.ml.report import _md_table

__all__ = ["TransformerReport"]

log = get_logger("transformers.report")


class TransformerReport(DLReport):
    """Persist the transformer run's report suite."""

    def __init__(self, reports_dir: str = "reports/transformers") -> None:
        super().__init__(reports_dir)

    # ── entry point ───────────────────────────────────────────────────────────
    def write(self, result) -> list[str]:
        """Write every report; return the list of paths."""
        self._json_report(result)
        self._adopt("deep_learning_report.json", "transformer_report.json",
                    lambda payload: {**payload,
                                     "attention": self._attention_dict(result),
                                     "cross_family_comparison":
                                     self._comparison_records(result)})

        md_lines = self._markdown_report(result)
        md_lines[0] = "# Transformer Models Report"
        md_lines += self._attention_sections(result)
        md_lines += self._comparison_section(result)
        self._dump_md("transformer_report.md", md_lines)
        self.written.remove(os.path.join(self.reports_dir,
                                         "deep_learning_report.md"))
        os.remove(os.path.join(self.reports_dir, "deep_learning_report.md"))

        self._html_report(md_lines)
        self._adopt("deep_learning_report.html", "transformer_report.html")

        self._leaderboard_csv(result)
        self._metrics_summary(result)
        self._history_csvs(result)
        log.info("wrote %d transformer reports under %s",
                 len(self.written), self.reports_dir)
        return list(self.written)

    # ── transformer-specific sections ─────────────────────────────────────────
    @staticmethod
    def _attention_dict(result) -> dict:
        return {t.name: t.attention.as_dict()
                for t in result.trained
                if not t.failed and getattr(t, "attention", None)}

    def _attention_sections(self, result) -> list[str]:
        lines: list[str] = []
        for t in result.trained:
            attn = getattr(t, "attention", None)
            if t.failed or attn is None:
                continue
            lines += ["", f"## Attention analysis — {t.name}", "",
                      f"Aggregated over {attn.n_samples} validation "
                      f"samples (mean over heads, layers, and queries).",
                      "", "| Layer | Mean attention entropy |", "|---|---|"]
            lines += [f"| {layer} | {ent:.4f} |"
                      for layer, ent in attn.entropy.items()]
            top = list(attn.feature_attention.items())[:15]
            if top:
                lines += ["", "### Top features by attention received", "",
                          "| Feature | Mean attention |", "|---|---|"]
                lines += [f"| {feat} | {val:.4f} |" for feat, val in top]
        return lines

    def _comparison_records(self, result) -> list[dict]:
        rows = []
        for family, board in (getattr(result, "family_boards", None)
                              or {}).items():
            if board is None or board.empty:
                continue
            for _, r in board.iterrows():
                rows.append({"family": family, "model": str(r["model"]),
                             **{m: float(r[m]) for m in
                                ("roc_auc", "f1", "recall", "pr_auc")
                                if m in r and pd.notna(r[m])}})
        return sorted(rows, key=lambda r: -r.get("roc_auc", 0.0))

    def _comparison_section(self, result) -> list[str]:
        records = self._comparison_records(result)
        if not records:
            return []
        board = pd.DataFrame(records)
        return ["", "## Comparison with previous ML and DL models", "",
                "Test-set metrics of every registered model across the "
                "classical ML, deep-learning, and transformer stages:", "",
                _md_table(board)]

    # ── plumbing ──────────────────────────────────────────────────────────────
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
