# Frontend

Nothing is built here yet. The dashboard is a pure client of the HTTP API (`backend/prism/api`), so
the technology is a free choice that does not touch the backend.

## Decision needed: Streamlit or React

| | Streamlit + Plotly | React + TypeScript + Recharts/Plotly |
|---|---|---|
| Time to a working dashboard | hours (Python, already installed) | 1-2 days |
| Look and polish | good enough, limited layout control | full control |
| Live refresh | `st.autorefresh` / fragments | polling or SSE |
| Risk with a 7-day deadline | low | medium |

Recommendation: **Streamlit first** (it can live in `frontend/app.py` and call the API with `httpx`); move to React only
if the team has the capacity after the engine and evaluation are done. If React is chosen, enable CORS for the
dev origin via `CORS_ORIGINS` (localhost:5173 and :3000 are allowed by default).

## What the single dashboard needs

All of it is already served by the API:

| Panel | Endpoint |
|---|---|
| KPI strip: documents, signals, high-risk, avg impact | `GET /api/stats` |
| Live risk feed (entity, event, sentiment, impact bar, confidence) | `GET /api/risk-signals?limit=50` (poll every few seconds) |
| Why this score? (impact breakdown) | `explanation.impact_points` on each signal |
| Latest stress test: before / after / loss | `GET /api/stress-runs?limit=1` |
| Asset-level loss contribution | `by_asset` / `by_type` in the stress result |
| Portfolio and scenario matrix | `GET /api/portfolio`, `GET /api/scenarios` |
| Event distribution, risk over time | `signals_by_event` in `/api/stats`; `GET /api/risk-signals` grouped client-side |
| Pipeline health (per-source status, latency) | `GET /api/pipeline/runs`, `GET /api/pipeline/rejects` |
| "Try it": analyse pasted text | `POST /api/analyze` |
| Demo button: replay next batch / trigger a scenario | `POST /api/pipeline/run`, `POST /api/stress-test` |

Interactive API docs: `http://127.0.0.1:8000/docs` while the backend is running.
