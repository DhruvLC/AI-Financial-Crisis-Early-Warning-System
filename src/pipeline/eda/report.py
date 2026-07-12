"""Automated EDA reporting — Markdown, HTML, JSON, and CSV.

Consumes the ordered list of :class:`~pipeline.eda.base.AnalysisResult` objects
plus the business-insights payload and emits, under ``reports/eda/``:

* ``eda_report.json``  — the full machine-readable record (summaries, notes,
  figures, insights);
* ``eda_report.md``    — a human-readable Markdown narrative;
* ``eda_report.html``  — a self-contained styled HTML version (with embedded
  figures) for sharing;
* ``statistics/*.csv`` — every analyzer table, one CSV each, for spreadsheet use.

Mirrors the shape/conventions of
:class:`pipeline.preprocessing.report.PreprocessingReport`.
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone

import pandas as pd

from ingestion.logging_config import get_logger

log = get_logger("eda.report")

_SEVERITY_EMOJI = {"critical": "🔴", "warning": "🟠", "info": "🔵"}


class EdaReport:
    """Persist the EDA run's JSON, Markdown, HTML and per-table CSV reports."""

    def __init__(self, reports_dir: str = "reports/eda") -> None:
        self.reports_dir = reports_dir
        self.stats_dir = os.path.join(reports_dir, "statistics")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────
    def write(self, results: list, insights: dict, meta: dict) -> dict:
        """Write all report formats; return a dict of output paths."""
        csv_paths = self._write_csvs(results)
        report = self._build(results, insights, meta, csv_paths)

        json_path = os.path.join(self.reports_dir, "eda_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        md_path = os.path.join(self.reports_dir, "eda_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._markdown(report))

        html_path = os.path.join(self.reports_dir, "eda_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self._html(report))

        log.info("wrote EDA reports: %s, %s, %s (+%d CSVs)",
                 json_path, md_path, html_path, len(csv_paths))
        return {"json": json_path, "md": md_path, "html": html_path,
                "csv": csv_paths, "report": report}

    # ── assembly ──────────────────────────────────────────────────────────────
    def _write_csvs(self, results: list) -> list[str]:
        paths = []
        for r in results:
            for name, tbl in getattr(r, "tables", {}).items():
                if not isinstance(tbl, pd.DataFrame):
                    continue
                fname = f"{r.analyzer}__{name}.csv"
                path = os.path.join(self.stats_dir, fname)
                try:
                    tbl.to_csv(path, index=False)
                    paths.append(path)
                except Exception as exc:  # noqa: BLE001 - never abort on one table
                    log.warning("could not write CSV %s (%s)", path, exc)
        return paths

    def _build(self, results, insights, meta, csv_paths) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": meta,
            "analyzers": [r.as_dict() for r in results],
            "insights": insights,
            "figures": [fig for r in results for fig in getattr(r, "figures", [])],
            "statistics_csvs": [os.path.basename(p) for p in csv_paths],
        }

    # ── Markdown ──────────────────────────────────────────────────────────────
    def _markdown(self, report: dict) -> str:
        meta = report["meta"]
        ins = report["insights"]
        L = [
            "# Exploratory Data Analysis Report",
            "",
            f"_Generated: {report['generated_at']}_",
            "",
            "## Dataset",
            "",
            f"- Source split: `{meta.get('dataset')}`",
            f"- Shape: {meta.get('n_rows')} rows × {meta.get('n_cols')} cols",
            f"- Target column: `{meta.get('target_col')}`",
            f"- Analyzers run: {meta.get('n_analyzers_run')} "
            f"({meta.get('n_analyzers_skipped')} skipped)",
            f"- Figures generated: {len(report['figures'])}",
            "",
            "## Executive summary",
            "",
            f"**{ins.get('headline', 'n/a')}**",
            "",
            f"Insights: {ins.get('severity_counts', {})}",
            "",
        ]

        if ins.get("insights"):
            L += ["### Prioritized insights", "",
                  "| Severity | Category | Finding | Recommendation |",
                  "|----------|----------|---------|----------------|"]
            for i in ins["insights"]:
                emoji = _SEVERITY_EMOJI.get(i["severity"], "")
                L.append(
                    f"| {emoji} {i['severity']} | {i['category']} | "
                    f"{i['message']} | {i['recommendation']} |")
            L.append("")

        L += ["## Analyzer details", ""]
        for a in report["analyzers"]:
            L.append(f"### {a['analyzer']}  ({a['status']})")
            L.append("")
            if a["status"] == "skipped":
                L += [f"_Skipped: {a.get('skip_reason')}_", ""]
                continue
            for note in a.get("notes", []):
                L.append(f"- {note}")
            summary = a.get("summary", {})
            if summary:
                L += ["", "```json", json.dumps(
                    self._compact(summary), indent=2, default=str)[:4000],
                    "```", ""]
            if a.get("tables"):
                L.append("Tables: " + ", ".join(
                    f"`statistics/{a['analyzer']}__{t}.csv`"
                    for t in a["tables"]))
                L.append("")

        if report["figures"]:
            L += ["## Figures", ""]
            for fig in report["figures"]:
                rel = os.path.relpath(fig, self.reports_dir)
                L.append(f"![{os.path.basename(fig)}]({rel})")
                L.append("")
        return "\n".join(L)

    # ── HTML ──────────────────────────────────────────────────────────────────
    def _html(self, report: dict) -> str:
        meta = report["meta"]
        ins = report["insights"]
        e = html.escape
        parts = [
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>EDA Report — Financial Crisis EWS</title>",
            "<style>",
            "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"
            "sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;"
            "color:#1a1a1a;line-height:1.5}",
            "h1,h2,h3{color:#12314f}h1{border-bottom:3px solid #2c7fb8;"
            "padding-bottom:.3rem}",
            "table{border-collapse:collapse;width:100%;margin:1rem 0;"
            "font-size:.9rem}th,td{border:1px solid #ddd;padding:.45rem;"
            "text-align:left;vertical-align:top}th{background:#12314f;"
            "color:#fff}tr:nth-child(even){background:#f6f8fa}",
            ".sev-critical{color:#c0392b;font-weight:700}"
            ".sev-warning{color:#d35400;font-weight:700}"
            ".sev-info{color:#2471a3}",
            ".card{background:#eef4fb;border-left:5px solid #2c7fb8;"
            "padding:.8rem 1rem;margin:1rem 0;border-radius:4px}",
            "img{max-width:100%;height:auto;border:1px solid #ddd;"
            "border-radius:4px;margin:.5rem 0}",
            "pre{background:#0f1b2b;color:#d6e2f0;padding:1rem;overflow:auto;"
            "border-radius:6px;font-size:.8rem}code{font-size:.85rem}",
            "</style></head><body>",
            "<h1>Exploratory Data Analysis Report</h1>",
            f"<p><em>Generated: {e(report['generated_at'])}</em></p>",
            "<h2>Dataset</h2><ul>",
            f"<li>Source split: <code>{e(str(meta.get('dataset')))}</code></li>",
            f"<li>Shape: {meta.get('n_rows')} rows × {meta.get('n_cols')} "
            "cols</li>",
            f"<li>Target: <code>{e(str(meta.get('target_col')))}</code></li>",
            f"<li>Analyzers run: {meta.get('n_analyzers_run')} "
            f"({meta.get('n_analyzers_skipped')} skipped)</li>",
            f"<li>Figures: {len(report['figures'])}</li></ul>",
            "<h2>Executive summary</h2>",
            f"<div class='card'><strong>{e(str(ins.get('headline', '')))}"
            "</strong></div>",
        ]

        if ins.get("insights"):
            parts.append("<h3>Prioritized insights</h3>")
            parts.append("<table><tr><th>Severity</th><th>Category</th>"
                         "<th>Finding</th><th>Recommendation</th></tr>")
            for i in ins["insights"]:
                parts.append(
                    f"<tr><td class='sev-{e(i['severity'])}'>"
                    f"{e(i['severity'])}</td><td>{e(i['category'])}</td>"
                    f"<td>{e(i['message'])}</td>"
                    f"<td>{e(i['recommendation'])}</td></tr>")
            parts.append("</table>")

        parts.append("<h2>Analyzer details</h2>")
        for a in report["analyzers"]:
            parts.append(f"<h3>{e(a['analyzer'])} "
                         f"<small>({e(a['status'])})</small></h3>")
            if a["status"] == "skipped":
                parts.append(f"<p><em>Skipped: {e(str(a.get('skip_reason')))}"
                             "</em></p>")
                continue
            if a.get("notes"):
                parts.append("<ul>" + "".join(
                    f"<li>{e(str(nt))}</li>" for nt in a["notes"]) + "</ul>")
            if a.get("summary"):
                blob = json.dumps(self._compact(a["summary"]), indent=2,
                                  default=str)[:4000]
                parts.append(f"<pre>{e(blob)}</pre>")

        if report["figures"]:
            parts.append("<h2>Figures</h2>")
            for fig in report["figures"]:
                rel = os.path.relpath(fig, self.reports_dir)
                parts.append(f"<h4>{e(os.path.basename(fig))}</h4>"
                             f"<img src='{e(rel)}' alt='{e(os.path.basename(fig))}'>")
        parts.append("</body></html>")
        return "\n".join(parts)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _compact(summary: dict) -> dict:
        """Trim long list-of-record blocks so embedded JSON stays readable."""
        out = {}
        for k, v in summary.items():
            if isinstance(v, list) and len(v) > 8:
                out[k] = v[:8] + [f"... (+{len(v) - 8} more)"]
            else:
                out[k] = v
        return out
