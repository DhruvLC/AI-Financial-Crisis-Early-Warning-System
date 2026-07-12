"""Data-quality scoring: turn check metrics into a 0-100 score + letter grade.

The score is a weighted blend of five component scores, each in [0, 1], pulled
from the metrics that the individual checks emit:

    schema        — required columns present, dtypes correct
    completeness  — 1 - missing fraction
    uniqueness    — 1 - duplicate-row fraction
    validity      — financial sanity + (dampened) outlier rate
    timeliness    — chronological order / temporal integrity

Components that don't apply to a source (e.g. timeliness for the cross-sectional
bankruptcy table) default to 1.0 so they neither help nor hurt.
"""
from __future__ import annotations

from .base import CheckOutcome

DEFAULT_WEIGHTS = {
    "schema": 0.20,
    "completeness": 0.25,
    "uniqueness": 0.15,
    "validity": 0.25,
    "timeliness": 0.15,
}

# Map component name -> (check name, metric key) it is read from.
_COMPONENT_SOURCES = {
    "schema": ("schema", "schema_score"),
    "completeness": ("missing_values", "completeness_score"),
    "uniqueness": ("duplicates", "uniqueness_score"),
    "timeliness": ("time_series", "timeliness_score"),
}


class QualityScorer:
    """Compute a weighted quality score from a dataset's check outcomes."""

    def __init__(self, weights: dict | None = None) -> None:
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    @staticmethod
    def grade(score: float) -> str:
        for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
            if score >= cutoff:
                return letter
        return "F"

    def score(self, outcomes: list[CheckOutcome]) -> tuple[float, str, dict]:
        by_name = {o.check: o for o in outcomes}
        components: dict[str, float] = {}

        for comp, (check, key) in _COMPONENT_SOURCES.items():
            o = by_name.get(check)
            components[comp] = self._metric(o, key, default=1.0)

        # Validity blends financial validity with a dampened outlier penalty.
        fin = by_name.get("financial")
        out = by_name.get("outliers")
        validity = self._metric(fin, "validity_score", default=1.0)
        outlier_rate = self._metric(out, "mean_iqr_outlier_pct", default=0.0)
        components["validity"] = max(0.0, validity - 0.5 * outlier_rate)

        total_w = sum(self.weights.values()) or 1.0
        score = 100.0 * sum(
            self.weights.get(c, 0.0) * v for c, v in components.items()
        ) / total_w
        score = max(0.0, min(100.0, score))
        return score, self.grade(score), components

    @staticmethod
    def _metric(outcome: CheckOutcome | None, key: str, default: float) -> float:
        if outcome is None or outcome.skipped:
            return default
        val = outcome.metrics.get(key)
        return float(val) if isinstance(val, (int, float)) else default
