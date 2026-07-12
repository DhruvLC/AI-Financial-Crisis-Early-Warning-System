"""Reporting for the preprocessing module.

Turns a :class:`~pipeline.preprocessing.pipeline.PreprocessResult` into a
machine-readable JSON report (per-step params/stats + the full lineage trail +
final split shapes) and a short human-readable Markdown summary, written under
``reports/preprocessing/`` — the same shape as the validation reports.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ingestion.logging_config import get_logger

log = get_logger("preprocessing.report")


class PreprocessingReport:
    """Persist the preprocessing run's JSON + Markdown reports."""

    def __init__(self, reports_dir: str = "reports/preprocessing") -> None:
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def build(self, result, target_col: str) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_col": target_col,
            "splits": {
                "train": {"rows": len(result.train),
                          "cols": result.train.shape[1]},
                "val": {"rows": len(result.val), "cols": result.val.shape[1]},
                "test": {"rows": len(result.test), "cols": result.test.shape[1]},
            },
            "steps": [r.as_dict() for r in result.step_results],
            "lineage": result.lineage,
        }

    def write(self, result, target_col: str) -> tuple[str, str]:
        report = self.build(result, target_col)
        result.report = report
        json_path = os.path.join(self.reports_dir, "preprocessing.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        md_path = os.path.join(self.reports_dir, "preprocessing.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._markdown(report))
        log.info("wrote preprocessing reports: %s, %s", json_path, md_path)
        return json_path, md_path

    # ── markdown ─────────────────────────────────────────────────────────────
    @staticmethod
    def _markdown(report: dict) -> str:
        lin = report["lineage"]
        init, final = lin["initial_shape"], lin["final_shape"]
        lines = [
            "# Preprocessing Report",
            "",
            f"_Generated: {report['generated_at']}_",
            "",
            f"- Target column: `{report['target_col']}`",
            f"- Initial shape: {init['rows']} rows × {init['cols']} cols",
            f"- Final (train) shape: {final['rows']} rows × {final['cols']} cols",
            f"- Rows removed: {lin['rows_removed']}  |  "
            f"Feature cols added: {lin['cols_added']}",
            f"- Steps applied: {lin['n_applied']}  |  skipped: {lin['n_skipped']}",
            "",
            "## Splits",
            "",
            "| Split | Rows | Cols |",
            "|-------|------|------|",
        ]
        for name, s in report["splits"].items():
            lines.append(f"| {name} | {s['rows']} | {s['cols']} |")
        lines += ["", "## Transformation lineage", "",
                  "| # | Step | Status | Rows Δ | Cols Δ | Notes |",
                  "|---|------|--------|--------|--------|-------|"]
        step_notes = {s["step"]: s.get("notes", []) for s in report["steps"]}
        for r in lin["trail"]:
            note = "; ".join(step_notes.get(r["step"], [])) or (
                r["skip_reason"] or "")
            lines.append(
                f"| {r['order']} | {r['step']} | {r['status']} | "
                f"{r['rows_delta']:+d} | {r['cols_delta']:+d} | {note} |")
        lines.append("")
        return "\n".join(lines)
