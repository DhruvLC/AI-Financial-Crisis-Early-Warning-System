"""CSV uploader component with schema validation feedback."""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from config import MAX_BATCH_ROWS, MAX_CSV_MB
from services.utils import (check_csv_features, find_duplicate_columns,
                            non_numeric_feature_columns, parse_csv,
                            template_csv)


def csv_uploader(expected_features: List[str]
                 ) -> Tuple[Optional[pd.DataFrame], bool]:
    """Upload + validate a CSV against the model schema.

    All guards run client-side BEFORE any backend call: file size, row
    count, required/duplicate columns, data types, and missing values.
    Returns ``(df, is_valid)``; renders all feedback inline.
    """
    st.download_button("⬇ Download CSV template",
                       data=template_csv(expected_features),
                       file_name="ews_batch_template.csv", mime="text/csv",
                       help="One row per company; all model feature columns "
                            "plus an optional 'id' column.")
    file = st.file_uploader(
        "Upload a CSV of companies to score", type=["csv"],
        accept_multiple_files=False,
        help=f"Limits: {MAX_BATCH_ROWS:,} rows, {MAX_CSV_MB} MB.")
    if file is None:
        return None, False

    size_mb = file.size / (1024 * 1024)
    if size_mb > MAX_CSV_MB:
        st.error(f"File is {size_mb:.1f} MB — the maximum is {MAX_CSV_MB} MB. "
                 "Please split the file into smaller batches.")
        return None, False

    df, err = parse_csv(file)
    if err:
        st.error(err)
        return None, False

    if len(df) > MAX_BATCH_ROWS:
        st.error(f"CSV has {len(df):,} rows — the maximum per batch is "
                 f"{MAX_BATCH_ROWS:,} (backend limit). Please split the "
                 "file and score it in batches.")
        return df, False

    dupes = find_duplicate_columns(df)
    if dupes:
        st.error(f"CSV contains duplicate column name(s): "
                 f"{', '.join(dupes)}. Column names must be unique.")
        return df, False

    missing, extra = check_csv_features(df, expected_features)
    if missing:
        st.error(f"CSV is missing {len(missing)} required feature "
                 f"column(s).")
        with st.expander("Missing columns"):
            st.write(missing)
        return df, False
    if extra:
        st.warning(f"{len(extra)} unrecognized column(s) will be ignored: "
                   f"{', '.join(extra[:8])}"
                   f"{'…' if len(extra) > 8 else ''}")

    non_numeric = non_numeric_feature_columns(df, expected_features)
    if non_numeric:
        st.error(f"{len(non_numeric)} feature column(s) contain non-numeric "
                 "values — all model features must be numbers.")
        with st.expander("Non-numeric columns"):
            st.write(non_numeric)
        return df, False

    nan_rows = int(df[expected_features].isna().any(axis=1).sum())
    if nan_rows:
        st.error(f"{nan_rows} row(s) contain missing values in feature "
                 "columns — please fill or drop them.")
        return df, False

    st.success(f"CSV validated: {len(df):,} rows × "
               f"{len(expected_features)} features.")
    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(20), use_container_width=True)
    return df, True
