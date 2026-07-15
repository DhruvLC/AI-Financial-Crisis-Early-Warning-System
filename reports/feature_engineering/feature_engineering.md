# Feature Engineering Report

_Generated: 2026-07-15T11:31:55.378390+00:00_

- Target column: `Bankrupt?`
- Initial (train) shape: 4773 rows × 96 cols
- Final (train) shape: 4773 rows × 23 cols
- Features generated: 124  |  removed: 197
- Steps applied: 5  |  skipped: 0
- EDA hints used: yes  |  Feature-store version: v001

## Splits

| Split | Rows | Cols |
|-------|------|------|
| train | 4773 | 23 |
| val | 1023 | 23 |
| test | 1023 | 23 |

## Transformation lineage

| # | Step | Status | Cols Δ | Generated | Removed | Notes |
|---|------|--------|--------|-----------|---------|-------|
| 1 | generation | applied | +117 | 117 | 0 | generated 117 new feature(s) |
| 2 | multicollinearity | applied | -105 | 0 | 105 | dropped 73 by correlation, 32 by VIF |
| 3 | selection | applied | -92 | 0 | 92 | selected 15 of 107 features |
| 4 | reduction | applied | +7 | 7 | 0 | pca: 7 components explain 95.9% of variance (append mode) |
| 5 | importance | applied | +0 | 0 | 0 | scored 22 features via random_forest, xgboost, shap |

## Top features by importance

| Feature | random_forest | xgboost | shap |
|---------|------|------|------|
| pca__1 | 0.1450 | 0.6086 | 0.1652 |
| diff__Borrowing dependency__minus__Retained Earnings to Total Assets | 0.1196 | 0.0857 | 0.1287 |
| Total debt/Total net worth | 0.0457 | 0.0171 | 0.0743 |
| Allocation rate per person | 0.0269 | 0.0223 | 0.0517 |
| Retained Earnings to Total Assets | 0.0371 | 0.0159 | 0.0449 |
| log__Net Value Growth Rate | 0.0332 | 0.0216 | 0.0399 |
| ROA(B) before interest and depreciation after tax | 0.0255 | 0.0158 | 0.0794 |
| Borrowing dependency | 0.0566 | 0.0143 | 0.0266 |
| Total income/Total expense | 0.0377 | 0.0132 | 0.0433 |
| diff__Net Income to Stockholder's Equity__minus__Total debt/Total net worth | 0.1047 | 0.0133 | 0.0311 |
| mul__Net Income to Stockholder's Equity__x__Borrowing dependency | 0.0350 | 0.0260 | 0.0163 |
| mul__Total debt/Total net worth__x__After-tax net Interest Rate | 0.0263 | 0.0213 | 0.0193 |
| Non-industry income and expenditure/revenue | 0.0257 | 0.0135 | 0.0469 |
| pca__7 | 0.0186 | 0.0138 | 0.0489 |
| pca__4 | 0.0193 | 0.0149 | 0.0395 |
