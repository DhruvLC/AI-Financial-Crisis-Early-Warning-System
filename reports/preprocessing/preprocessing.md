# Preprocessing Report

_Generated: 2026-07-12T08:26:59.694996+00:00_

- Target column: `Bankrupt?`
- Initial shape: 6819 rows × 96 cols
- Final (train) shape: 4773 rows × 96 cols
- Rows removed: 0  |  Feature cols added: 0
- Steps applied: 5  |  skipped: 1

## Splits

| Split | Rows | Cols |
|-------|------|------|
| train | 4773 | 96 |
| val | 1023 | 96 |
| test | 1023 | 96 |

## Transformation lineage

| # | Step | Status | Rows Δ | Cols Δ | Notes |
|---|------|--------|--------|--------|-------|
| 1 | cleaning | applied | +0 | +0 | removed 0 invalid record(s); standardized 0 categorical + 0 date column(s) |
| 2 | duplicates | applied | +0 | +0 | removed 0 duplicate row(s) |
| 3 | imputation | applied | +0 | +0 | imputed 0 cell(s) via 'median' (numeric) + most_frequent (categorical) |
| 4 | outliers | applied | +0 | +0 | clipped outliers via 'winsorize' on 95 numeric column(s) |
| 5 | encoding | skipped | +0 | +0 | no categorical feature columns |
| 6 | scaling | applied | +0 | +0 | scaled 95 numeric column(s) via 'standard' |
