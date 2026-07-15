"""Reporting for the feature-engineering module.

Turns a :class:`~pipeline.feature_engineering.pipeline.FeatureEngineeringResult`
into a machine-readable JSON report (per-step params/stats + the full lineage
trail + final split shapes + importance table) plus a human-readable Markdown
summary and a per-feature importance CSV, written under
``reports/feature_engineering/`` — the same shape as the preprocessing reports.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ingestion.logging_config import get_logger

log = get_logger("features.report")


class FeatureEngineeringReport:
    """Persist the feature-engineering run's JSON + Markdown + CSV reports."""

    def __init__(self,
                 reports_dir: str = "reports/feature_engineering") -> None:
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def build(self, result, target_col: str) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_col": target_col,
            "splits": {
                name: {"rows": len(df), "cols": df.shape[1]}
                for name, df in (("train", result.train),
                                 ("val", result.val), ("test", result.test))},
            "final_features": [c for c in result.train.columns
                               if c != target_col],
            "steps": [r.as_dict() for r in result.step_results],
            "lineage": result.lineage,
            "eda_hints_used": bool(result.hints),
            "store_version": (result.store_record or {}).get("version"),
        }

    def write(self, result, target_col: str) -> tuple[str, str]:
        report = self.build(result, target_col)
        result.report = report
        json_path = os.path.join(self.reports_dir, "feature_engineering.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        md_path = os.path.join(self.reports_dir, "feature_engineering.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._markdown(report))
        if result.importances is not None:
            csv_path = os.path.join(self.reports_dir, "feature_importance.csv")
            result.importances.rename_axis("feature").to_csv(csv_path)
            log.info("wrote importance table: %s", csv_path)
        log.info("wrote feature-engineering reports: %s, %s",
                 json_path, md_path)
        return json_path, md_path

    # ── markdown ─────────────────────────────────────────────────────────────
    @staticmethod
    def _markdown(report: dict) -> str:
        lin = report["lineage"]
        init, final = lin["initial_shape"], lin["final_shape"]
        lines = [
            "# Feature Engineering Report",
            "",
            f"_Generated: {report['generated_at']}_",
            "",
            f"- Target column: `{report['target_col']}`",
            f"- Initial (train) shape: {init['rows']} rows × "
            f"{init['cols']} cols",
            f"- Final (train) shape: {final['rows']} rows × "
            f"{final['cols']} cols",
            f"- Features generated: {lin['n_generated_total']}  |  "
            f"removed: {lin['n_removed_total']}",
            f"- Steps applied: {lin['n_applied']}  |  "
            f"skipped: {lin['n_skipped']}",
            f"- EDA hints used: {'yes' if report['eda_hints_used'] else 'no'}"
            f"  |  Feature-store version: {report['store_version'] or '—'}",
            "",
            "## Splits",
            "",
            "| Split | Rows | Cols |",
            "|-------|------|------|",
        ]
        for name, s in report["splits"].items():
            lines.append(f"| {name} | {s['rows']} | {s['cols']} |")

        lines += ["", "## Transformation lineage", "",
                  "| # | Step | Status | Cols Δ | Generated | Removed | Notes |",
                  "|---|------|--------|--------|-----------|---------|-------|"]
        step_notes = {s["step"]: s.get("notes", []) for s in report["steps"]}
        for r in lin["trail"]:
            note = "; ".join(step_notes.get(r["step"], [])) or (
                r["skip_reason"] or "")
            lines.append(
                f"| {r['order']} | {r['step']} | {r['status']} | "
                f"{r['cols_delta']:+d} | {r['n_generated']} | "
                f"{r['n_removed']} | {note} |")

        # Top features by consensus importance, when the step ran.
        imp = next((s for s in report["steps"]
                    if s["step"] == "importance" and s["status"] == "applied"),
                   None)
        if imp and imp["stats"].get("top_features"):
            methods = imp["stats"]["methods"]
            lines += ["", "## Top features by importance", "",
                      "| Feature | " + " | ".join(methods) + " |",
                      "|---------|" + "|".join("------" for _ in methods) + "|"]
            for row in imp["stats"]["top_features"][:15]:
                cells = " | ".join(f"{row.get(m, 0):.4f}" for m in methods)
                lines.append(f"| {row['feature']} | {cells} |")
        lines.append("")
        return "\n".join(lines)
