"""Feature-transformation lineage tracking.

Mirrors :class:`pipeline.preprocessing.lineage.LineageTracker`. Every feature
step emits a compact :pyattr:`FeatureResult.transformation` record; this tracker
accumulates them into an ordered, auditable trail of exactly how the feature set
evolved — which columns were generated, removed, or projected, in order, with
per-step column deltas. This is what lets an engineered dataset be reproduced,
explained, and debugged after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureLineageRecord:
    """One entry in the feature lineage trail (one feature step)."""

    order: int
    step: str
    status: str                 # "applied" | "skipped"
    skip_reason: str | None
    cols_before: int
    cols_after: int
    n_generated: int
    n_removed: int
    stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "step": self.step,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
            "cols_delta": self.cols_after - self.cols_before,
            "n_generated": self.n_generated,
            "n_removed": self.n_removed,
            "stats": self.stats,
        }


class FeatureLineageTracker:
    """Accumulate an ordered, auditable trail of feature transformations."""

    def __init__(self) -> None:
        self._records: list[FeatureLineageRecord] = []
        self._initial_shape: tuple[int, int] | None = None
        self._final_shape: tuple[int, int] | None = None

    def start(self, df: pd.DataFrame) -> None:
        self._initial_shape = (len(df), df.shape[1])

    def record(self, result, before: pd.DataFrame, after: pd.DataFrame) -> None:
        self._records.append(FeatureLineageRecord(
            order=len(self._records) + 1,
            step=result.step,
            status="skipped" if result.skipped else "applied",
            skip_reason=result.skip_reason,
            cols_before=before.shape[1],
            cols_after=after.shape[1],
            n_generated=len(result.generated),
            n_removed=len(result.removed),
            stats=result.stats,
        ))

    def finish(self, df: pd.DataFrame) -> None:
        self._final_shape = (len(df), df.shape[1])

    @property
    def records(self) -> list[FeatureLineageRecord]:
        return list(self._records)

    def as_dict(self) -> dict:
        init = self._initial_shape or (0, 0)
        final = self._final_shape or init
        return {
            "initial_shape": {"rows": init[0], "cols": init[1]},
            "final_shape": {"rows": final[0], "cols": final[1]},
            "cols_added": sum(max(r.cols_after - r.cols_before, 0)
                              for r in self._records),
            "cols_removed": sum(max(r.cols_before - r.cols_after, 0)
                                for r in self._records),
            "n_generated_total": sum(r.n_generated for r in self._records),
            "n_removed_total": sum(r.n_removed for r in self._records),
            "n_steps": len(self._records),
            "n_applied": sum(r.status == "applied" for r in self._records),
            "n_skipped": sum(r.status == "skipped" for r in self._records),
            "trail": [r.as_dict() for r in self._records],
        }
