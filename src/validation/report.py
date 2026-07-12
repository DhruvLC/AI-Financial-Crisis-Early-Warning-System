"""Report generation for the Data Validation module.

Writes one JSON report per dataset plus a combined corpus summary (JSON +
human-readable Markdown) under ``reports/validation/``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ingestion.logging_config import get_logger

from .base import DatasetReport

log = get_logger("validation.report")


class ReportGenerator:
    """Persist per-dataset and corpus-level validation reports."""

    def __init__(self, reports_dir: str = "reports/validation") -> None:
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def write_dataset(self, report: DatasetReport) -> str:
        path = os.path.join(self.reports_dir, f"{report.source}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.as_dict(), f, indent=2, default=str)
        log.info("wrote dataset report: %s", path)
        return path

    def write_summary(self, reports: list[DatasetReport]) -> tuple[str, str]:
        summary = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "n_datasets": len(reports),
            "n_present": sum(r.present for r in reports),
            "n_valid": sum(r.is_valid for r in reports),
            "datasets": [
                {
                    "source": r.source,
                    "present": r.present,
                    "n_rows": r.n_rows,
                    "n_cols": r.n_cols,
                    "quality_score": round(r.quality_score, 2),
                    "quality_grade": r.quality_grade,
                    "n_errors": r.n_errors,
                    "n_warnings": r.n_warnings,
                    "is_valid": r.is_valid,
                    "load_error": r.load_error,
                }
                for r in reports
            ],
        }
        json_path = os.path.join(self.reports_dir, "_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        md_path = os.path.join(self.reports_dir, "_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._markdown(summary))
        log.info("wrote corpus summary: %s / %s", json_path, md_path)
        return json_path, md_path

    @staticmethod
    def _markdown(summary: dict) -> str:
        lines = [
            "# Data Validation Summary",
            "",
            f"_Generated: {summary['validated_at']}_",
            "",
            f"- Datasets: **{summary['n_datasets']}**  ",
            f"- Present: **{summary['n_present']}**  ",
            f"- Valid (no errors): **{summary['n_valid']}**",
            "",
            "| Source | Present | Rows | Cols | Score | Grade | Errors | Warnings |",
            "|--------|:-------:|-----:|-----:|------:|:-----:|-------:|---------:|",
        ]
        for d in summary["datasets"]:
            present = "✓" if d["present"] else "—"
            score = f"{d['quality_score']:.1f}" if d["present"] else "—"
            lines.append(
                f"| {d['source']} | {present} | {d['n_rows']} | {d['n_cols']} | "
                f"{score} | {d['quality_grade']} | {d['n_errors']} | "
                f"{d['n_warnings']} |"
            )
        return "\n".join(lines) + "\n"
