"""PRISM Risk Cockpit - one Streamlit page over the PRISM API.

    cd frontend
    streamlit run app.py            # expects the API at http://127.0.0.1:8000 (override: PRISM_API_URL)
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from prism_charts import events_figure, impact_breakdown_figure, risk_timeline_figure, waterfall_figure
from prism_client import (
    HIGH_IMPACT,
    ApiError,
    PrismClient,
    asset_table,
    impact_breakdown,
    money,
    signals_frame,
    waterfall_rows,
)

st.set_page_config(page_title="PRISM Risk Cockpit", layout="wide")

DEFAULT_API = os.environ.get("PRISM_API_URL", "http://127.0.0.1:8000")


def get_client() -> PrismClient:
    # tests inject a client through session state; otherwise build one for the configured URL
    injected = st.session_state.get("prism_client")
    if injected is not None:
        return injected
    url = st.session_state.get("api_url", DEFAULT_API)
    if st.session_state.get("_client_url") != url:
        st.session_state["_client"] = PrismClient(url)
        st.session_state["_client_url"] = url
    return st.session_state["_client"]


# ============================================================================ sidebar
with st.sidebar:
    st.title("PRISM")
    st.caption("Unstructured text → risk signals → portfolio stress")
    st.text_input("API URL", value=DEFAULT_API, key="api_url")
    client = get_client()
    try:
        health = client.health()
    except ApiError as exc:
        st.error(f"{exc}\n\nStart it with `python -m uvicorn prism.api.app:create_app --factory` in `backend/`.")
        st.stop()
    st.success(f"API online · sources: {', '.join(health['sources']) or 'none'}")
    st.caption(f"sentiment `{health['sentiment_model']}` · events `{health['event_classifier']}` · impact `{health['impact_model']}`")

    st.subheader("Demo controls")
    batch = st.slider("Records per replay batch", 1, 10, 3)
    if st.button("Ingest next replay batch", use_container_width=True):
        try:
            report = client.run_pipeline(["replay"], batch)
            st.toast(f"{report['signals_loaded']} new signals from {report['sources'].get('replay', {}).get('fetched', 0)} records")
        except ApiError as exc:
            st.error(str(exc))
    auto_stream = st.toggle("Auto-stream replay", value=False, help="Ingest one batch every refresh")
    refresh_seconds = st.slider("Refresh every (seconds)", 3, 30, 5, disabled=not auto_stream)

    st.subheader("What-if stress test")
    try:
        scenario_names = [s["event_type"] for s in client.scenarios()]
    except ApiError:
        scenario_names = []
    whatif_event = st.selectbox("Event type", scenario_names)
    whatif_impact = st.slider("Impact score", 1.0, 10.0, 9.0, 0.5)
    if st.button("Run what-if", use_container_width=True, disabled=not scenario_names):
        try:
            st.session_state["whatif"] = client.stress_test(whatif_event, whatif_impact, persist=False)
        except ApiError as exc:
            st.error(str(exc))
    if st.session_state.get("whatif") and st.button("Clear what-if", use_container_width=True):
        st.session_state.pop("whatif")

# ============================================================================ header + filters
st.title("AI Financial Risk Cockpit")
st.caption("News and social text are scored by the NLP Risk Engine; high-impact negative events trigger a stress test of the wholesale book.")

f1, f2, f3, f4 = st.columns([2, 3, 2, 2])
min_impact = f1.slider("Minimum impact", 1.0, 10.0, 1.0, 0.5)
event_filter = f2.multiselect("Event types", sorted(health.get("event_types", [])) or [], placeholder="All event types")
source_filter = f3.multiselect("Sources", health["sources"], placeholder="All sources")
negative_only = f4.toggle("Negative sentiment only", value=False)


def render_signal_detail(signal: pd.Series) -> None:
    st.markdown(f"**{signal['who']}** · {signal['event_type']} · {signal['time']:%d %b %H:%M} UTC")
    st.caption(signal["headline"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Impact", f"{signal['impact_score']:.2f}")
    c2.metric("Sentiment", f"{signal['sentiment_score']:+.2f}")
    c3.metric("Confidence", f"{signal['confidence']:.0%}")
    breakdown = impact_breakdown({"explanation": signal["explanation"]})
    st.plotly_chart(impact_breakdown_figure(breakdown), use_container_width=True, theme=None, key="breakdown_chart")
    explanation = signal["explanation"] or {}
    top = ", ".join(f"{e['event']} {e['p']:.0%}" for e in explanation.get("event_top3", []))
    st.caption(
        f"Event evidence: {top or 'n/a'} (method `{explanation.get('event_method', '?')}`) · "
        f"corroborated by {explanation.get('corroboration', 1)} outlet(s) · "
        f"models: {', '.join(f'{k}={v}' for k, v in explanation.get('models', {}).items())}"
    )


def render_stress(result: dict, label: str) -> None:
    st.markdown(f"**{label}: {result['scenario']}** ({result['event_type']}"
                + (f", impact {result['impact_score']:.2f}" if result.get("impact_score") else "") + ")")
    shock = result["shock"]
    st.caption(
        f"Shock applied: equities {shock['equity_pct']:+.0%} · rates {shock['rate_bps']:+.0f} bp · "
        f"credit spreads {shock['credit_spread_bps']:+.0f} bp"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Portfolio before", money(result["value_before"]))
    m2.metric("Portfolio after", money(result["value_after"]))
    m3.metric("Stress P&L", money(result["pnl"]), f"{result['pnl_pct']:+.2%}")
    st.plotly_chart(waterfall_figure(waterfall_rows(result)), use_container_width=True, theme=None, key=f"waterfall_{label}")
    with st.expander("Asset-level table"):
        table = asset_table(result)
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "name": "Asset",
                "asset_type": "Class",
                "value_before": st.column_config.NumberColumn("Before", format="$%.0f"),
                "pnl": st.column_config.NumberColumn("P&L", format="$%.0f"),
                "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.2f"),
                "value_after": st.column_config.NumberColumn("After", format="$%.0f"),
            },
        )


def live_view() -> None:
    if auto_stream:
        try:
            client.run_pipeline(["replay"], batch)
        except ApiError as exc:
            st.warning(f"Replay batch failed: {exc}")
    try:
        stats = client.stats()
        signals = signals_frame(client.signals(limit=500, min_impact=min_impact))
        stress_runs = client.stress_runs(limit=10)
        portfolio = client.portfolio()
    except ApiError as exc:
        st.error(str(exc))
        return

    if event_filter:
        signals = signals[signals["event_type"].isin(event_filter)]
    if source_filter:
        signals = signals[signals["source"].isin(source_filter)]
    if negative_only:
        signals = signals[signals["direction"] == "negative"]

    # ------------------------------------------------------------------ KPI strip
    total_value = sum(a["value"] for a in portfolio["assets"])
    latest = stress_runs[0] if stress_runs else None
    k = st.columns(6)
    k[0].metric("Articles processed", f"{stats['documents']:,}")
    k[1].metric("Risk signals", f"{stats['signals']:,}")
    k[2].metric("High-risk signals", f"{stats['high_risk_signals']:,}",
                help=f"impact ≥ {HIGH_IMPACT:g} and negative sentiment")
    k[3].metric("Average impact", f"{stats['avg_impact']:.2f}" if stats["avg_impact"] is not None else "–")
    k[4].metric("Portfolio value", money(total_value, 1))
    k[5].metric("Latest stress P&L", money(latest["pnl"], 1) if latest else "–",
                f"{latest['pnl_pct']:+.2%}" if latest else None)

    # ------------------------------------------------------------------ feed + detail
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Live risk signals")
        if signals.empty:
            st.info("No signals yet. Use **Ingest next replay batch** or turn on auto-stream.")
            selected = None
        else:
            view = signals[["time", "who", "event_type", "sentiment_score", "impact_score", "confidence", "headline"]]
            state = st.dataframe(
                view,
                hide_index=True,
                use_container_width=True,
                height=420,
                on_select="rerun",
                selection_mode="single-row",
                key="feed",
                column_config={
                    "time": st.column_config.DatetimeColumn("Time (UTC)", format="HH:mm:ss"),
                    "who": "Entity",
                    "event_type": "Event",
                    "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
                    "impact_score": st.column_config.ProgressColumn("Impact", min_value=1, max_value=10, format="%.1f"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                    "headline": st.column_config.TextColumn("Headline", width="large"),
                },
            )
            rows = state.selection.rows if state else []
            selected = signals.iloc[rows[0]] if rows else signals.sort_values("impact_score", ascending=False).iloc[0]
    with right:
        st.subheader("Why this score?")
        if selected is None:
            st.caption("Select a signal to see its impact breakdown.")
        else:
            render_signal_detail(selected)

    # ------------------------------------------------------------------ stress test
    st.subheader("Portfolio stress test")
    whatif = st.session_state.get("whatif")
    if whatif:
        render_stress(whatif, "What-if")
    elif latest:
        trigger = next((s for _, s in signals.iterrows() if s["id"] == latest.get("trigger_signal_id")), None)
        render_stress(latest, "Auto-triggered")
        if trigger is not None:
            st.caption(f"Triggered by: “{trigger['headline']}” ({trigger['who']}, impact {trigger['impact_score']:.2f})")
        if len(stress_runs) > 1:
            with st.expander(f"Earlier stress runs ({len(stress_runs) - 1})"):
                st.dataframe(
                    pd.DataFrame(stress_runs[1:])[["created_at", "event_type", "impact_score", "pnl", "pnl_pct"]],
                    hide_index=True,
                    use_container_width=True,
                )
    else:
        st.info("No stress test yet. One runs automatically when a high-impact negative event arrives, or use What-if.")

    # ------------------------------------------------------------------ charts
    if not signals.empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Risk over time")
            st.plotly_chart(risk_timeline_figure(signals), use_container_width=True, theme=None, key="timeline_chart")
            with st.expander("Table view"):
                st.dataframe(signals[["time", "who", "event_type", "sentiment_score", "impact_score"]], hide_index=True)
        with c2:
            st.subheader("Signals by event type")
            st.plotly_chart(events_figure(signals), use_container_width=True, theme=None, key="events_chart")
            with st.expander("Table view"):
                counts = signals["event_type"].value_counts().rename_axis("event_type").reset_index(name="signals")
                st.dataframe(counts, hide_index=True)

    # ------------------------------------------------------------------ pipeline health
    with st.expander("Pipeline health"):
        try:
            runs = client.runs(limit=10)
            rejects = client.rejects(limit=20)
        except ApiError as exc:
            st.error(str(exc))
            return
        if runs:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "started": r["started_at"],
                            "sources": ", ".join(f"{n}: {s['fetched']}" + (f" ({s['error']})" if s["error"] else "")
                                                 for n, s in r["sources"].items()),
                            "duplicates": r["duplicates"],
                            "signals": r["signals_loaded"],
                            "seconds": round(r["duration_seconds"], 3),
                        }
                        for r in runs
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        if rejects:
            st.caption("Rejected records")
            st.dataframe(pd.DataFrame(rejects)[["source", "reason", "snippet"]], hide_index=True, use_container_width=True)


st.fragment(live_view, run_every=refresh_seconds if auto_stream else None)()

# ============================================================================ try the engine
with st.expander("Try the Risk Engine on your own text"):
    text = st.text_area("One headline or post per line", placeholder="Rating agency downgrades a major lender to junk")
    if st.button("Analyze") and text.strip():
        try:
            result = client.analyze([line for line in text.splitlines() if line.strip()])
            if result["signals"]:
                st.dataframe(
                    signals_frame(result["signals"])[["who", "event_type", "sentiment_score", "impact_score", "confidence", "headline"]],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No tracked entity and no recognisable event: the engine produced no signal.")
            for reject in result["rejected"]:
                st.warning(f"Rejected ({reject['reason']}): {reject['snippet']}")
        except ApiError as exc:
            st.error(str(exc))
