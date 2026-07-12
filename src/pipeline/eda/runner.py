"""EDA orchestrator.

Sequences every :class:`~pipeline.eda.base.EdaAnalyzer` over a processed
DataFrame, runs the :class:`~pipeline.eda.insights.BusinessInsightsEngine`, and
hands the collected results to :class:`~pipeline.eda.report.EdaReport`.

Design parallels :class:`validation.runner.DataValidationRunner` and
:class:`pipeline.preprocessing.pipeline.PreprocessingPipeline`: config-driven
construction, per-analyzer exception isolation honouring ``fail_fast``, uniform
logging via the shared logging config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ingestion.logging_config import get_logger

from .base import AnalysisResult, EdaError
from .insights import BusinessInsightsEngine
from .plotting import FigureManager
from .report import EdaReport
from .analyzers import ANALYZER_REGISTRY, DEFAULT_ORDER


@dataclass
class EdaResult:
    """Everything an EDA run produces."""

    results: list = field(default_factory=list)          # list[AnalysisResult]
    insights: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)          # written file paths

    def by_name(self) -> dict:
        return {r.analyzer: r for r in self.results}


class EdaRunner:
    """Run the full EDA suite and emit the multi-format report."""

    def __init__(self, cfg: dict, target_col: str) -> None:
        self.cfg = cfg or {}
        self.eda_cfg = dict(self.cfg.get("eda", {}))
        self.target_col = target_col
        self.log = get_logger("eda.runner")
        self.fail_fast = bool(self.eda_cfg.get("fail_fast", False))

        fig_cfg = dict(self.eda_cfg.get("figures", {}))
        self.figures = FigureManager(
            figures_dir=fig_cfg.get("dir", "reports/eda/figures"),
            dpi=int(fig_cfg.get("dpi", 150)),
            style=fig_cfg.get("style", "whitegrid"),
            palette=fig_cfg.get("palette", "deep"),
            enabled=bool(fig_cfg.get("enabled", True)),
        ) if bool(fig_cfg.get("enabled", True)) else None

        self.reports_dir = self.eda_cfg.get("reports_dir", "reports/eda")
        self.order = self.eda_cfg.get("analyzers", DEFAULT_ORDER)

    # ── build one analyzer ────────────────────────────────────────────────────
    def _make(self, name: str):
        cls = ANALYZER_REGISTRY[name]
        acfg = dict(self.eda_cfg.get(name, {}))
        return cls(cfg=acfg, target_col=self.target_col, figures=self.figures)

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame, dataset_name: str = "processed") -> EdaResult:
        if df is None or len(df) == 0:
            raise EdaError("EDA received an empty dataset")

        self.log.info("starting EDA on '%s' (%d rows x %d cols)",
                      dataset_name, len(df), df.shape[1])
        results: list[AnalysisResult] = []
        for name in self.order:
            if name not in ANALYZER_REGISTRY:
                self.log.warning("unknown analyzer '%s' — skipping", name)
                continue
            try:
                results.append(self._make(name).run(df))
            except EdaError as exc:
                if self.fail_fast:
                    raise
                self.log.error("analyzer '%s' failed (continuing): %s",
                               name, exc)
                results.append(AnalysisResult(
                    analyzer=name, skipped=True,
                    skip_reason=f"error: {exc}"))

        insights = BusinessInsightsEngine(
            self.eda_cfg.get("insights", {})).generate(
            {r.analyzer: r for r in results})

        n_run = sum(1 for r in results if not r.skipped)
        n_skip = sum(1 for r in results if r.skipped)
        meta = {
            "dataset": dataset_name,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "target_col": self.target_col,
            "n_analyzers_run": n_run,
            "n_analyzers_skipped": n_skip,
        }

        reporter = EdaReport(self.reports_dir)
        out = reporter.write(results, insights, meta)

        self.log.info("EDA complete — %d analyzer(s) run, %d skipped, "
                      "%d figure(s), %d insight(s)",
                      n_run, n_skip, len(out["report"]["figures"]),
                      insights["n_insights"])
        return EdaResult(results=results, insights=insights,
                         report=out["report"],
                         outputs={k: out[k] for k in ("json", "md", "html",
                                                      "csv")})
