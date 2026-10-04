# Dashboard (Streamlit)

One page, "AI Financial Risk Cockpit", on top of the PRISM API. It holds no business logic: everything
comes from the HTTP API, so it can be replaced without touching the backend.

```bash
# terminal 1 - the API (from backend/)
python -m uvicorn prism.api.app:create_app --factory

# terminal 2 - the dashboard (from frontend/)
pip install -r requirements.txt
streamlit run app.py          # http://localhost:8501 ; API URL via the sidebar or PRISM_API_URL
```

## What is on the page

| Section | Source | Notes |
|---|---|---|
| KPI strip | `GET /api/stats`, `/api/portfolio`, `/api/stress-runs` | "high-risk" = impact >= 7 **and** negative sentiment |
| Live risk signals (select a row) | `GET /api/risk-signals` | impact / confidence as bars, sentiment signed |
| Why this score? | `explanation.impact_points` of the selected signal | base 1.0 + five feature contributions = the score |
| Portfolio stress test | latest auto-triggered run, or the sidebar what-if | before / after / P&L tiles, waterfall by asset class, asset table |
| Risk over time, signals by event type | signals | each chart has a table view |
| Pipeline health | `GET /api/pipeline/runs`, `/api/pipeline/rejects` | per-source counts and errors |
| Try the Risk Engine | `POST /api/analyze` | paste headlines, one per line |

Demo controls (sidebar): ingest the next replay batch, auto-stream replay every N seconds, what-if stress test.

## Files

- `app.py`: the Streamlit page (layout only)
- `prism_client.py`: HTTP client + pure data shaping (unit-tested without Streamlit)
- `prism_charts.py`: Plotly figures; colors from the validated reference palette (single series blue;
  loss/negative red vs gain/positive blue; neutral gray), labels never clipped (`automargin`)
- `.streamlit/config.toml`: light theme matching the chart palette

Tests: `backend/tests/test_dashboard.py` (client, shaping, charts, and the page rendered headless with
`streamlit.testing`). Verified in a browser at 1440x900 and at the narrow pane width: no truncated KPI values,
no clipped chart labels.
