"""Declarative schema + semantic contracts for every ingested source.

Each :class:`SourceSchema` describes what a well-formed interim dataset for
that source looks like: its columns and expected dtypes, which column is the
canonical entity key, which columns are dates vs. integer years, whether it is
a time series, and — for financial feeds — the semantic rules
(:class:`FinancialSpec`) used by the financial validator.

Sources whose exact columns are config-driven (FRED series, World Bank
indicators, Alpha Vantage OHLCV) set ``dynamic_columns=True`` and only pin the
stable structural columns as required; the rest are accepted as expected extras.

This is the per-dataset source of truth. It is intentionally richer than
``ingestion.cross_validation.SCHEMA_REGISTRY`` (which only needs coverage/range
facts for the corpus-level layer); the two describe different validation tiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# dtype "kinds" checked via pandas.api.types (see checks/schema.py)
NUMERIC, INTEGER, FLOAT = "numeric", "integer", "float"
DATETIME, STRING, BOOL, ANY = "datetime", "string", "bool", "any"


@dataclass
class ColumnSpec:
    name: str
    dtype: str = ANY          # one of the kind constants above
    required: bool = True


@dataclass
class FinancialSpec:
    """Semantic rules for financial validation (all fields optional)."""
    # Columns that must be >= 0 (e.g. revenue, volume) — negatives are invalid.
    nonneg_columns: list[str] = field(default_factory=list)
    # Columns that must be strictly > 0 (e.g. total assets, share price).
    positive_columns: list[str] = field(default_factory=list)
    # Ratio columns mapped to plausible (low, high) bounds.
    ratio_bounds: dict = field(default_factory=dict)
    # Integer fiscal-year column, validated against [min_year, current+1].
    fiscal_year_column: str | None = None
    # Long-format feeds (SEC EDGAR): a `concept` column + a `value` column,
    # with per-concept sign rules, e.g. {"Assets": "positive", "Revenues": "nonneg"}.
    concept_column: str | None = None
    value_column: str | None = None
    concept_rules: dict = field(default_factory=dict)


@dataclass
class SourceSchema:
    source: str
    columns: list[ColumnSpec] = field(default_factory=list)
    entity_column: str | None = None
    date_columns: list[str] = field(default_factory=list)   # datetime-like
    year_columns: list[str] = field(default_factory=list)   # integer year
    time_series_keys: list[str] = field(default_factory=list)  # group keys for TS
    freshness_days: int | None = None
    dynamic_columns: bool = False       # extra numeric columns are expected
    financial: FinancialSpec | None = None

    # convenience accessors ---------------------------------------------------
    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    @property
    def known_columns(self) -> set[str]:
        return {c.name for c in self.columns}

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)


# Sensible fiscal-year floor for modern filings.
MIN_FISCAL_YEAR = 1970

SOURCE_SCHEMAS: dict[str, SourceSchema] = {
    # ── SEC EDGAR: long-format XBRL company facts ───────────────────────────
    "sec_edgar": SourceSchema(
        source="sec_edgar",
        columns=[
            ColumnSpec("cik", STRING),
            ColumnSpec("entity_id", STRING),
            ColumnSpec("entity_name", STRING, required=False),
            ColumnSpec("concept", STRING),
            ColumnSpec("unit", STRING, required=False),
            ColumnSpec("value", NUMERIC),
            ColumnSpec("fy", NUMERIC, required=False),
            ColumnSpec("fp", STRING, required=False),
            ColumnSpec("form", STRING, required=False),
            ColumnSpec("date", DATETIME),
            ColumnSpec("filed", ANY, required=False),
        ],
        entity_column="entity_id",
        date_columns=["date"],
        time_series_keys=["cik", "concept"],
        financial=FinancialSpec(
            fiscal_year_column="fy",
            concept_column="concept",
            value_column="value",
            concept_rules={
                "Assets": "positive",
                "Liabilities": "nonneg",
                "Revenues": "nonneg",
                "CashAndCashEquivalentsAtCarryingValue": "nonneg",
            },
        ),
    ),
    # ── Kaggle Company Bankruptcy: wide labelled financial ratios ───────────
    "kaggle_bankruptcy": SourceSchema(
        source="kaggle_bankruptcy",
        columns=[ColumnSpec("Bankrupt?", INTEGER)],
        dynamic_columns=True,             # ~95 ratio feature columns
        financial=FinancialSpec(
            # These are ratios; the dataset is min-max scaled to [0, 1].
            # Values well outside that band indicate corruption.
            ratio_bounds={},              # populated generically at run time
        ),
    ),
    # ── FRED macro indicators (date + config-driven series) ─────────────────
    "fred": SourceSchema(
        source="fred",
        columns=[ColumnSpec("date", DATETIME)],
        date_columns=["date"],
        time_series_keys=[],
        freshness_days=400,
        dynamic_columns=True,
    ),
    # ── Yahoo Finance OHLCV ─────────────────────────────────────────────────
    "yahoo_finance": SourceSchema(
        source="yahoo_finance",
        columns=[
            ColumnSpec("date", DATETIME),
            ColumnSpec("ticker", STRING),
            ColumnSpec("entity_id", STRING),
            ColumnSpec("Close", NUMERIC),
        ],
        entity_column="entity_id",
        date_columns=["date"],
        time_series_keys=["entity_id"],
        freshness_days=30,
        dynamic_columns=True,
        financial=FinancialSpec(
            positive_columns=["Open", "High", "Low", "Close"],
            nonneg_columns=["Volume"],
        ),
    ),
    # ── Alpha Vantage market data ───────────────────────────────────────────
    "alpha_vantage": SourceSchema(
        source="alpha_vantage",
        columns=[
            ColumnSpec("date", DATETIME),
            ColumnSpec("symbol", STRING),
            ColumnSpec("entity_id", STRING, required=False),
        ],
        entity_column="entity_id",
        date_columns=["date"],
        time_series_keys=["symbol"],
        freshness_days=30,
        dynamic_columns=True,
        financial=FinancialSpec(
            positive_columns=["open", "high", "low", "close"],
            nonneg_columns=["volume"],
        ),
    ),
    # ── Kaggle Financial News headlines ─────────────────────────────────────
    "kaggle_news": SourceSchema(
        source="kaggle_news",
        columns=[ColumnSpec("Date", DATETIME, required=False)],
        date_columns=["Date"],
        dynamic_columns=True,
    ),
    # ── Kaggle Stock Market OHLCV ───────────────────────────────────────────
    "kaggle_stock": SourceSchema(
        source="kaggle_stock",
        columns=[
            ColumnSpec("Date", DATETIME),
            ColumnSpec("Open", NUMERIC, required=False),
            ColumnSpec("High", NUMERIC, required=False),
            ColumnSpec("Low", NUMERIC, required=False),
            ColumnSpec("Close", NUMERIC, required=False),
            ColumnSpec("Volume", NUMERIC, required=False),
            ColumnSpec("ticker", STRING, required=False),
        ],
        entity_column="ticker",
        date_columns=["Date"],
        time_series_keys=["ticker"],
        dynamic_columns=True,
        financial=FinancialSpec(
            positive_columns=["Open", "High", "Low", "Close"],
            nonneg_columns=["Volume"],
        ),
    ),
    # ── World Bank (entity + integer year + indicators) ─────────────────────
    "world_bank": SourceSchema(
        source="world_bank",
        columns=[
            ColumnSpec("entity_id", STRING),
            ColumnSpec("country", STRING, required=False),
            ColumnSpec("date", INTEGER),
        ],
        entity_column="entity_id",
        year_columns=["date"],
        time_series_keys=["entity_id"],
        dynamic_columns=True,
        financial=FinancialSpec(nonneg_columns=["GDP", "Population"]),
    ),
    # ── IMF (entity + integer year + value) ─────────────────────────────────
    "imf": SourceSchema(
        source="imf",
        columns=[
            ColumnSpec("entity_id", STRING),
            ColumnSpec("date", INTEGER),
            ColumnSpec("value", NUMERIC),
        ],
        entity_column="entity_id",
        year_columns=["date"],
        time_series_keys=["entity_id"],
        dynamic_columns=True,
    ),
    # ── OECD (entity + period + value) ──────────────────────────────────────
    "oecd": SourceSchema(
        source="oecd",
        columns=[
            ColumnSpec("entity_id", STRING),
            ColumnSpec("date", ANY),
            ColumnSpec("value", NUMERIC),
        ],
        entity_column="entity_id",
        date_columns=[],
        time_series_keys=["entity_id"],
        dynamic_columns=True,
    ),
}


def schema_for(source: str) -> SourceSchema:
    """Return the registered schema, or a permissive default for unknown feeds."""
    return SOURCE_SCHEMAS.get(source) or SourceSchema(source=source,
                                                       dynamic_columns=True)
