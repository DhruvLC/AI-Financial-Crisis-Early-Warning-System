"""Bridge from the EDA module's report into Feature-Engineering hints.

The EDA stage already computed, per feature, the information this module would
otherwise recompute: highly-correlated pairs and high-VIF features
(``correlation`` analyzer), skewed features that benefit from log transforms
(``distributions``), the most discriminative financial ratios (``ratios``), and
per-feature target separability (``relationships``). Rather than duplicate that
work, :class:`EdaInsightLoader` reads ``reports/eda/eda_report.json`` and exposes
a compact ``hints`` dict the feature steps consult (e.g. the log-transform step
seeds its candidate list from the EDA-flagged skewed features).

Everything is best-effort: a missing/oddly-shaped report yields empty hints, and
the pipeline still works from first principles — the EDA report only *seeds*
decisions, it never gates them.
"""
from __future__ import annotations

import json
import os

from ingestion.logging_config import get_logger

log = get_logger("features.eda_insights")


class EdaInsightLoader:
    """Load and normalize EDA findings into feature-engineering hints."""

    def __init__(self, report_path: str = "reports/eda/eda_report.json") -> None:
        self.report_path = report_path

    def load(self) -> dict:
        """Return a hints dict; empty if the EDA report is absent/unreadable."""
        if not os.path.exists(self.report_path):
            log.info("no EDA report at %s — proceeding without hints",
                     self.report_path)
            return {}
        try:
            with open(self.report_path, encoding="utf-8") as f:
                report = json.load(f)
        except Exception as exc:  # noqa: BLE001 - hints are optional
            log.warning("could not read EDA report (%s) — no hints", exc)
            return {}

        summaries = {a.get("analyzer"): a.get("summary", {})
                     for a in report.get("analyzers", [])}
        hints = {
            "skewed_features": self._skewed(summaries),
            "high_correlation_pairs": self._corr_pairs(summaries),
            "high_vif_features": self._high_vif(summaries),
            "top_discriminative": self._discriminative(summaries),
            "generated_at": report.get("generated_at"),
        }
        log.info("EDA hints: %d skewed, %d corr-pairs, %d high-VIF features",
                 len(hints["skewed_features"]),
                 len(hints["high_correlation_pairs"]),
                 len(hints["high_vif_features"]))
        return hints

    # ── extractors (each defensive against missing keys) ──────────────────────
    @staticmethod
    def _skewed(summaries: dict) -> list[str]:
        s = summaries.get("distributions", {}) or {}
        return list(s.get("highly_skewed", []) or [])

    @staticmethod
    def _corr_pairs(summaries: dict) -> list[dict]:
        s = summaries.get("correlation", {}) or {}
        return list(s.get("top_pairs", []) or [])

    @staticmethod
    def _high_vif(summaries: dict) -> list[str]:
        s = summaries.get("correlation", {}) or {}
        return list(s.get("high_vif_features", []) or [])

    @staticmethod
    def _discriminative(summaries: dict) -> list[str]:
        s = summaries.get("ratios", {}) or {}
        return [r.get("feature") for r in (s.get("top_discriminative", []) or [])
                if r.get("feature")]
