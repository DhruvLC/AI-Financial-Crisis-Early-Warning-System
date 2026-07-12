"""Data lineage tracking for the preprocessing pipeline.

Every step emits a compact :pyattr:`StepResult.transformation` record; the
:class:`LineageTracker` accumulates them into an ordered, auditable trail of
exactly what happened to the data — the transformations applied, in order, with
their fitted parameters and row/column deltas. This is what lets a preprocessed
dataset be reproduced, explained, and debugged after the fact.

The tracker also snapshots the frame *shape* before and after the run so the
report can show the net effect (rows removed, feature columns added by encoding,
etc.) at a glance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class LineageRecord:
    """One entry in the lineage trail (one preprocessing step)."""

    order: int
    step: str
    status: str                 # "applied" | "skipped"
    skip_reason: str | None
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    params: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "step": self.step,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_delta": self.rows_after - self.rows_before,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
            "cols_delta": self.cols_after - self.cols_before,
            "params": self.params,
            "stats": self.stats,
        }


class LineageTracker:
    """Accumulate an ordered, auditable trail of preprocessing transformations."""

    def __init__(self) -> None:
        self._records: list[LineageRecord] = []
        self._initial_shape: tuple[int, int] | None = None
        self._final_shape: tuple[int, int] | None = None

    def start(self, df: pd.DataFrame) -> None:
        self._initial_shape = (len(df), df.shape[1])

    def record(self, result, before: pd.DataFrame, after: pd.DataFrame) -> None:
        """Append one step's transformation record given its before/after frames."""
        self._records.append(LineageRecord(
            order=len(self._records) + 1,
            step=result.step,
            status="skipped" if result.skipped else "applied",
            skip_reason=result.skip_reason,
            rows_before=len(before),
            rows_after=len(after),
            cols_before=before.shape[1],
            cols_after=after.shape[1],
            params=result.params,
            stats=result.stats,
        ))

    def finish(self, df: pd.DataFrame) -> None:
        self._final_shape = (len(df), df.shape[1])

    # ── views ────────────────────────────────────────────────────────────────
    @property
    def records(self) -> list[LineageRecord]:
        return list(self._records)

    def as_dict(self) -> dict:
        init = self._initial_shape or (0, 0)
        final = self._final_shape or init
        # Actual rows removed by steps — summed from per-step deltas so the
        # train/val/test split (which shrinks the final frame but removes
        # nothing) is not miscounted as removal.
        rows_removed = sum(max(r.rows_before - r.rows_after, 0)
                           for r in self._records)
        cols_added = sum(max(r.cols_after - r.cols_before, 0)
                         for r in self._records)
        return {
            "initial_shape": {"rows": init[0], "cols": init[1]},
            "final_shape": {"rows": final[0], "cols": final[1]},
            "rows_removed": rows_removed,
            "cols_added": cols_added,
            "n_steps": len(self._records),
            "n_applied": sum(r.status == "applied" for r in self._records),
            "n_skipped": sum(r.status == "skipped" for r in self._records),
            "trail": [r.as_dict() for r in self._records],
        }
