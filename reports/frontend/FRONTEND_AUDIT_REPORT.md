# Phase 10B — Streamlit Frontend AUDIT REPORT

**Date:** 2026-07-26
**Scope:** Read-only audit of `frontend/`, its backend integration, documentation, and verification evidence.
**Auditor action taken:** Inspection only — no code modified, no files regenerated, no fixes applied.

---

## 1. Executive Summary

The Phase 10B Streamlit frontend is complete, well-structured, and verified
end-to-end against the live FastAPI backend. All 6 required pages exist and
execute without exceptions; all 9 required component categories are present;
backend consumption is via HTTP only with zero duplication of business
logic. Git status confirms the only changes are additive (`frontend/`,
`reports/frontend/`, `requirements-frontend.txt`) — the ML pipeline and
FastAPI backend are untouched.

Three non-blocking defects were identified (sidebar URL override not
propagated to prediction calls; local-file feature-schema fallback breaks
frontend/backend separation on remote deployment; no client-side batch-size
chunking). None prevent correct operation in the documented local setup.

**Verdict: PASS WITH RECOMMENDATIONS** — Production readiness **8.5 / 10**.

## 2. Architecture Review

```
frontend/
├── app.py            entry point (Home)          39 lines
├── config.py         URL resolution + constants  40 lines
├── pages/            6 pages (1_ … 6_)           84/67/72/~110/~120/~90 lines
├── components/       sidebar, cards, charts, tables, uploader
├── services/         api_client (typed client), utils (CSV/dataframe)
└── assets/           placeholder (.gitkeep)
```

- ✅ Clean layered design: pages → components → services → HTTP.
- ✅ Multipage app follows Streamlit's `pages/` numbered-file convention.
- ✅ Zero business logic in the frontend — predictions, validation,
  thresholds, and risk banding all come from API responses.
- ✅ `setup_page()` enforces consistent page config; `render_sidebar()`
  shared across every page.
- ✅ Errors normalized in one place (`APIError`) and rendered via one
  component (`error_box`).
- ⚠️ Minor: `4_Model_Information.py` builds one Plotly figure inline
  (metric comparison) instead of via `components/charts.py`, and imports
  `plotly.express` mid-file — small consistency deviation, not a defect.

## 3. Integration Assessment

| Concern | Finding |
|---|---|
| Endpoints consumed | `/health`, `/version`, `/models`, `/metrics`, `/predict`, `/predict/batch` (+ `/validate` implemented in client, unused by pages) — all under `/api/v1`, matching `src/api/routes.py` and `src/api/health.py` exactly |
| Schema alignment | Request bodies (`{features, id}`, `{instances}`) and response fields (`prediction`, `probability`, `risk_score`, `risk_level`, `confidence_score`, `threshold`, `model_version`, `algorithm`, `prediction_timestamp`) match `src/api/schemas.py` |
| Logic duplication | None — client is transport + error normalization only |
| Caching | `st.cache_data` with sensible TTLs (health 15 s, models 60 s, schema 300 s); health cache keyed by base URL and cleared on URL change/refresh |
| **Defect (medium)** | Sidebar "Backend URL" override is stored in `st.session_state.api_url` and honored by `cached_health`/`cached_models` (URL passed explicitly), but `get_client()` constructs `APIClient()` from the env/secrets/default — so **Single/Batch Prediction and API Status probes ignore the sidebar override**. Runtime backend switching only partially works. |
| **Defect (low)** | `get_feature_schema()` reads `models/best_model.json` from the local filesystem. Correct locally, but if the frontend is deployed separately from the backend the schema (and therefore both prediction pages) fails, despite `/validate` existing server-side. |
| **Defect (low)** | Backend enforces `max_batch_size = 1000`; the batch page sends all rows in one request with no client-side chunking or pre-check — a >1000-row CSV yields a raw 400 error rather than guidance. |

## 4. UI Assessment

| Page | Assessment |
|---|---|
| 1 Dashboard | ✅ Status row, service metrics, best-model card + metrics chart, nav cards; graceful offline stop with start-command hint |
| 2 Single Prediction | ✅ Form with 22 schema-driven inputs, risk banner, gauge with threshold, metric grid, raw-response expander; spinner during scoring |
| 3 Batch Prediction | ✅ Template download, validated upload, session-persisted results, summary stats, CSV download, styled table capped at 100 preview rows, 4 charts |
| 4 Model Information | ✅ Best-model card, registry table with numeric formatting, cross-model metric comparison, feature schema table |
| 5 API Status | ✅ Health/version/metrics with raw expanders, endpoint probe latency table, docs link, manual re-probe |
| 6 About | ✅ Architecture diagram, phase table, output semantics, appropriate non-advice disclaimer |

Components — all 9 required categories present: sidebar ✅, metric cards ✅,
status badges ✅, CSV uploader (with schema/NaN validation) ✅, tables
(progress-styled columns) ✅, charts ✅, alerts (`error_box`, banners) ✅,
loading indicators (`st.spinner`) ✅, error messages ✅.

Charts — 7 interactive Plotly builders (donut, bar, histogram with threshold
line, strip, gauge, model-metrics bar, comparison bar). Good practice
observed: risk colors always paired with text labels (not color alone),
single sequential hue for magnitude, transparent backgrounds.

- ⚠️ Cosmetic: pervasive `use_container_width=True` — deprecated by
  Streamlit (removal announced after 2025-12-31); currently warns, works.
- ⚠️ Minor: Single Prediction defaults all 22 features to 0.0 — valid but
  unrepresentative inputs are easy to submit silently.

## 5. Backend Integration & Untouched-Pipeline Confirmation

- ✅ `git status` shows only untracked additions: `frontend/`,
  `reports/frontend/`, `requirements-frontend.txt`. **No tracked file in
  `src/` (ML, DL, transformer, SSL, API), `tests/`, `configs/`, or
  `models/` was modified.** Last commit ("Completed the Backend Part")
  predates all frontend work.
- ✅ FastAPI backend consumed strictly over REST; no imports from `src/`
  into the frontend. The only cross-boundary touch is the read-only
  `models/best_model.json` schema fallback noted in §3.

## 6. Documentation Review

- ✅ `frontend/README.md`: structure tree, installation, backend + frontend
  run commands, 3-tier URL configuration with deployed-backend example,
  per-page usage, screenshot placeholder table. Accurate against the code.
- ✅ `reports/frontend/FRONTEND_COMPLETION_REPORT.md`: present; created/reused
  inventories match the filesystem; verification claims match evidence
  reproduced in this audit.
- ⚠️ Screenshots are placeholders only; no `.streamlit/secrets.toml.example`.

## 7. Test Summary (evidence from the verification session, re-checked where possible)

| Check | Result |
|---|---|
| `py_compile` on all 14 frontend modules (re-run during audit) | ✅ Pass |
| Backend `/health` — model v004 (extra_trees) loaded | ✅ |
| `POST /predict` (22 features) → 200 with full result payload | ✅ |
| `POST /predict/batch` (5 instances) → 200, 5 predictions | ✅ |
| Malformed payload → 422, surfaced through `error_box` path | ✅ |
| Streamlit headless launch, `_stcore/health` → ok | ✅ |
| All 6 page routes → HTTP 200 | ✅ |
| `AppTest` execution of every page against the live backend — no exceptions | ✅ |
| Gaps | No persistent automated test suite (`tests/test_frontend.py` absent); no CI; offline-backend page behavior exercised only implicitly; downloads verified by wiring, not by driving the browser |

## 8. Risks

1. **Partial URL-override wiring (medium):** users switching backends via
   the sidebar will see health for the new backend but predict against the
   old one — silent inconsistency.
2. **Deployment coupling (low-medium):** local `best_model.json` dependency
   makes a split frontend/backend deployment fail on both prediction pages.
3. **Large-batch failure UX (low):** >1000-row CSVs hit the backend limit
   with a raw error.
4. **Deprecation debt (low):** `use_container_width` removal in future
   Streamlit versions would break rendering calls if dependencies are
   upgraded unpinned.
5. **No frontend test suite in `tests/` (low):** regressions detectable
   only manually; verification was session-scoped.
6. **Python 3.9 runtime (info):** system Python used; frontend code uses
   `X | None` unions only inside `from __future__ import annotations`
   scope, so it runs — but pinning a supported runtime is prudent.

## 9. Recommendations (NOT implemented — audit only)

1. Thread `st.session_state.api_url` into `get_client()` so all calls honor
   the sidebar override.
2. Fetch the feature schema from the backend (e.g., extend `/models` or use
   `/validate`) and keep `best_model.json` as fallback only.
3. Pre-check CSV row count against `max_batch_size` (or chunk requests)
   with a friendly message.
4. Replace `use_container_width` with `width="stretch"`; pin
   `streamlit<next-major` in `requirements-frontend.txt` meanwhile.
5. Add `tests/test_frontend.py` using `streamlit.testing.v1.AppTest`
   (entrypoint + `switch_page`) so page execution is CI-verifiable.
6. Capture the placeholder screenshots; add `.streamlit/secrets.toml.example`.
7. Move the inline metric-comparison chart in `4_Model_Information.py` into
   `components/charts.py` for consistency.

## 10. Production Readiness Score

**8.5 / 10**

| Dimension | Score | Rationale |
|---|---|---|
| Architecture | 9/10 | Clean layering; one minor consistency deviation |
| Backend integration | 8/10 | Correct and verified; URL-override + schema-fallback defects |
| UI completeness | 10/10 | 6/6 pages, 9/9 components, 7 chart types |
| Error handling | 9/10 | Normalized, user-friendly; batch-limit edge unguarded |
| Configuration | 8/10 | 3-tier resolution good; sidebar override partially wired |
| Documentation | 9/10 | Accurate and complete; screenshots pending |
| Testing | 7/10 | Strong one-off verification; no persistent suite/CI |

## 11. Final Verdict

## ✅ PASS WITH RECOMMENDATIONS

The frontend is functionally complete, verified against the live backend,
cleanly separated from the untouched ML pipeline and FastAPI service, and
suitable for local/staging use as documented. Address recommendations 1–3
before a split-deployment production rollout.
