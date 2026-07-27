"""Frontend utilities — CSV parsing, payload building, result framing."""
from __future__ import annotations

import io
from typing import List, Optional, Tuple

import pandas as pd


def parse_csv(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Parse an uploaded CSV. Returns ``(df, error_message)``."""
    try:
        df = pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001 - user-supplied file
        return None, f"Could not parse CSV: {exc}"
    if df.empty:
        return None, "The CSV file is empty."
    return df, None


def check_csv_features(df: pd.DataFrame,
                       expected: List[str]) -> Tuple[List[str], List[str]]:
    """Return ``(missing_features, extra_columns)`` vs the model schema."""
    cols = set(df.columns)
    id_like = {"id", "ID", "Id"}
    missing = [f for f in expected if f not in cols]
    extra = [c for c in df.columns
             if c not in set(expected) | id_like]
    return missing, extra


def find_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that appear more than once, in first-seen order.

    Note that ``pandas.read_csv`` de-duplicates repeated headers by suffixing
    them (``x``, ``x.1``), so raw duplicates surface here only for frames built
    programmatically; the suffixed twins are reported as unrecognized extra
    columns by :func:`check_csv_features` instead.
    """
    seen: set = set()
    dupes: List[str] = []
    for col in df.columns:
        if col in seen and col not in dupes:
            dupes.append(str(col))
        seen.add(col)
    return dupes


def non_numeric_feature_columns(df: pd.DataFrame,
                                expected: List[str]) -> List[str]:
    """Return expected feature columns whose values are not fully numeric.

    A column qualifies when every non-empty value is coercible to a float, so
    numeric strings (``"1.5"``) pass while text, booleans-as-words, and mixed
    columns fail. Missing columns are skipped — they are reported separately as
    missing required features.
    """
    bad: List[str] = []
    for feature in expected:
        if feature not in df.columns:
            continue
        col = df[feature]
        if isinstance(col, pd.DataFrame):  # duplicated header
            col = col.iloc[:, 0]
        if pd.api.types.is_bool_dtype(col) or \
                not pd.api.types.is_numeric_dtype(col):
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.isna().sum() > col.isna().sum():
                bad.append(str(feature))
    return bad


def df_to_instances(df: pd.DataFrame, expected: List[str]) -> List[dict]:
    """Convert a validated dataframe into /predict/batch instances."""
    id_col = next((c for c in ("id", "ID", "Id") if c in df.columns), None)
    instances = []
    for idx, row in df.iterrows():
        instances.append({
            "id": str(row[id_col]) if id_col else f"row-{idx}",
            "features": {f: float(row[f]) for f in expected},
        })
    return instances


def predictions_to_df(predictions: List[dict]) -> pd.DataFrame:
    """Flatten API prediction results into a display/download dataframe."""
    df = pd.DataFrame(predictions)
    ordered = ["id", "prediction", "risk_level", "risk_score", "probability",
               "confidence_score", "threshold", "model_version", "algorithm",
               "prediction_timestamp"]
    return df[[c for c in ordered if c in df.columns]]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def template_csv(expected: List[str]) -> bytes:
    """A one-row template CSV with all required feature columns."""
    df = pd.DataFrame([{"id": "company-001", **{f: 0.0 for f in expected}}])
    return df_to_csv_bytes(df)


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
