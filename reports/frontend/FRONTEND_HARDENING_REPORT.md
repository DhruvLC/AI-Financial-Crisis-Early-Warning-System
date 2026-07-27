# Frontend Hardening Report

**Project:** Financial Crisis Early Warning System — Streamlit Frontend
**Date:** 2026-07-27
**Scope:** Fix of one blocking bug found by the verification audit; no other code changed.

---

## 1. The Bug

`frontend/components/uploader.py` imported two helpers from
`frontend/services/utils.py` that did not exist:

- `find_duplicate_columns` (used at `uploader.py:52`)
- `non_numeric_feature_columns` (used at `uploader.py:70`)

**Reproduced impact:**

```
ImportError: cannot import name 'find_duplicate_columns' from 'services.utils'
```

Because `pages/3_Batch_Prediction.py:11` imports `csv_uploader`, the entire
**Batch Prediction page failed to load**, and two of the five client-side CSV
checks (duplicate columns, unsupported data types) were unreachable.

## 2. The Fix

Two functions were added to `frontend/services/utils.py`. **No other file was
modified.**

### `find_duplicate_columns(df) -> List[str]`
Returns column names appearing more than once, in first-seen order, each
reported once. Documents the `pandas.read_csv` caveat: repeated CSV headers are
auto-suffixed (`x`, `x.1`) by pandas, so those twins surface as *unrecognized
extra columns* via `check_csv_features` instead; raw duplicates occur for
programmatically built frames.

### `non_numeric_feature_columns(df, expected) -> List[str]`
Returns expected feature columns whose values are not fully numeric:
- Numeric dtypes pass; bool dtype is rejected (not a valid model feature).
- Object columns are coerced with `pd.to_numeric(errors="coerce")` — numeric
  strings (`"1.5"`) pass; a column fails only if coercion introduces *new* NaNs
  (pre-existing NaNs are handled by the separate missing-values check).
- Missing columns are skipped (reported separately as missing features).
- Duplicated-header `DataFrame` slices are handled defensively.

## 3. Verification Performed

| Check | Result |
|---|---|
| `import components.uploader` from `frontend/` | ✅ succeeds |
| All 7 entry files (`app.py`, 6 pages) parse | ✅ AST-OK |
| Streamlit headless startup (`--server.port 8599`) | ✅ HTTP 200 (home and Batch Prediction routes) |
| Duplicate-column detection (`concat` twin column) | ✅ returns `['a']` |
| Non-numeric detection (text ✗, bool ✗ path, numeric string ✓) | ✅ returns exactly the text column |
| Template CSV round-trip → parse → schema check → instances | ✅ zero missing/extra, valid instances |
| Missing/extra column detection | ✅ correct lists |
| Empty CSV rejection | ✅ error message returned |
| Duplicate CSV headers (`f1,f1`) flagged via `f1.1` extra | ✅ |

All checks passed: **`ALL VALIDATION CHECKS PASS`**.

## 4. Regression Scope

- Only `frontend/services/utils.py` changed (two additive functions).
- Existing functions (`parse_csv`, `check_csv_features`, `df_to_instances`,
  `predictions_to_df`, `df_to_csv_bytes`, `template_csv`, `fmt_pct`) untouched.
- Backend (`src/`, `tests/`) untouched — confirmed via `git status`.

## 5. Status

**Fixed and verified.** The Batch Prediction page and all five client-side CSV
guards (max rows, max file size, duplicate columns, missing required columns,
unsupported data types) are now functional end-to-end.
