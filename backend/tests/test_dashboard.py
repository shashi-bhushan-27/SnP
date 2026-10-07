"""Dashboard (frontend/) tests: API client, data shaping, chart builders and a headless Streamlit run.

Skipped when the frontend requirements (streamlit, plotly, pandas) are not installed; CI installs them
in a separate job.
"""

from __future__ import annotations

import sys

import httpx
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("pandas")

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from prism.api.app import create_app  # noqa: E402
from prism.bootstrap import build_container  # noqa: E402
from prism.config import REPO_ROOT  # noqa: E402
from prism.storage import SqlStore  # noqa: E402

FRONTEND = REPO_ROOT / "frontend"
sys.path.insert(0, str(FRONTEND))

import prism_charts  # noqa: E402
from prism_client import (  # noqa: E402
    ApiError,
    PrismClient,
    asset_table,
    direction,
    evidence_frame,
    impact_breakdown,
    money,
    outcomes_frame,
    signals_frame,
    waterfall_rows,
)


@pytest.fixture
def api(settings, engine):
    container = build_container(settings, store=SqlStore("sqlite://"), engine=engine)
    with TestClient(create_app(container)) as test_client:
        yield test_client


@pytest.fixture
def client(api) -> PrismClient:
    return PrismClient("http://testserver", client=api)


@pytest.fixture
def loaded(client) -> PrismClient:
    client.run_pipeline(["replay"], 100)
    return client


def unreachable() -> PrismClient:
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    return PrismClient("http://127.0.0.1:9", client=httpx.Client(transport=httpx.MockTransport(refuse)))


# ---------------------------------------------------------------- client
def test_client_reads_and_actions(loaded) -> None:
    assert loaded.health()["status"] == "ok"
    assert "Geopolitical" in loaded.health()["event_types"]
    assert loaded.stats()["signals"] > 0
    assert loaded.signals(min_impact=7, event_type=None)  # None filters are dropped, not sent
    assert loaded.stress_runs()
    assert loaded.runs()[0]["signals_loaded"] > 0
    assert loaded.rejects()[0]["reason"] == "non_english"
    assert loaded.analyze(["Tesla shares rise as quarterly deliveries beat analyst estimates"])["signals"]


def test_client_turns_http_errors_into_api_errors(client) -> None:
    with pytest.raises(ApiError, match="404"):
        client.stress_test("Product Launch")


def test_client_reports_an_unreachable_api() -> None:
    with pytest.raises(ApiError, match="Cannot reach"):
        unreachable().health()


# ---------------------------------------------------------------- shaping
def test_signals_frame(loaded) -> None:
    df = signals_frame(loaded.signals(limit=500))
    assert {"time", "who", "direction", "high_risk"} <= set(df.columns)
    assert df["time"].is_monotonic_decreasing
    assert set(df["direction"]) <= {"negative", "neutral", "positive"}
    assert (df.loc[df["high_risk"], "impact_score"] >= 7).all()
    assert "MARKET" in set(df["who"]) and "TSLA" in set(df["who"])


def test_signals_frame_empty() -> None:
    assert signals_frame([]).empty


@pytest.mark.parametrize("score,expected", [(-0.5, "negative"), (-0.2, "negative"), (0.0, "neutral"), (0.2, "positive")])
def test_direction_bands(score, expected) -> None:
    assert direction(score) == expected


def test_impact_breakdown_sums_to_the_score(loaded) -> None:
    signal = loaded.signals(limit=1)[0]
    assert impact_breakdown(signal)["points"].sum() == pytest.approx(signal["impact_score"], abs=0.01)


def test_waterfall_steps_reconcile(client) -> None:
    result = client.stress_test("Geopolitical", 9)
    rows = waterfall_rows(result)
    assert list(rows["step"]) == ["Before", "Loans", "Bonds", "Equities", "Derivatives", "After"]
    deltas = rows.loc[rows["measure"] == "relative", "value"].sum()
    assert rows["value"].iloc[0] + deltas == pytest.approx(rows["value"].iloc[-1])
    assert asset_table(result)["pnl"].is_monotonic_increasing  # biggest loss first


@pytest.mark.parametrize("value,text", [(60_000_000, "$60.00M"), (-5_200_000, "-$5.20M"), (2_500, "$2K"), (1.5e9, "$1.50B")])
def test_money(value, text) -> None:
    assert money(value) == text


# ---------------------------------------------------------------- charts
def test_chart_builders_use_the_validated_palette(loaded) -> None:
    signals = signals_frame(loaded.signals(limit=500))
    timeline = prism_charts.risk_timeline_figure(signals)
    colors = {trace.name: trace.marker.color for trace in timeline.data}
    assert colors["negative sentiment"] == prism_charts.RED and colors["positive sentiment"] == prism_charts.BLUE

    waterfall = prism_charts.waterfall_figure(waterfall_rows(loaded.stress_test("Geopolitical", 9)))
    assert waterfall.data[0].type == "waterfall"
    assert waterfall.data[0].decreasing.marker.color == prism_charts.RED

    events = prism_charts.events_figure(signals)
    assert events.data[0].marker.color == prism_charts.BLUE  # one series, one color
    breakdown = prism_charts.impact_breakdown_figure(impact_breakdown(loaded.signals(limit=1)[0]))
    assert list(breakdown.data[0].y)[0] == "Base"


def test_history_views(client) -> None:
    result = client.stress_test("Bankruptcy", 9, headline="Lender collapses after a deposit run", quantile=0.1)
    history = result["history"]
    dist = outcomes_frame(history["distribution"])
    assert dist["P&L today"].is_monotonic_increasing and history["basis"]["id"] in set(dist["id"])
    fig = prism_charts.distribution_figure(dist, history["basis"]["id"], history["expected_pnl"])
    colors = list(fig.data[0].marker.color)
    assert colors.count(prism_charts.RED) == 1  # exactly one highlighted basis event
    assert len(outcomes_frame(history["analogs"])) == 5

    evidence = evidence_frame(client.history_summary())
    bankruptcy = evidence.set_index("event type").loc["Bankruptcy"]
    assert bankruptcy["10y fell"] == 100 and bankruptcy["hand-written matrix"].startswith("-9%")

    analogs = client.analogs("Bank collapses after a run on deposits", k=2)
    assert len(analogs) == 2


# ---------------------------------------------------------------- the Streamlit page, headless
def run_app(client: PrismClient) -> AppTest:
    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=60)
    app.session_state["prism_client"] = client
    return app.run()


def test_dashboard_renders_with_data(loaded) -> None:
    app = run_app(loaded)
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Articles processed", "Risk signals", "High-risk signals", "Portfolio before", "Stress P&L"} <= set(labels)
    assert any("Auto-triggered" in md.value for md in app.markdown)


def test_dashboard_renders_empty_state(client) -> None:
    app = run_app(client)
    assert not app.exception
    assert any("No signals yet" in info.value for info in app.info)


def test_dashboard_explains_an_unreachable_api() -> None:
    app = run_app(unreachable())
    assert not app.exception
    assert any("Cannot reach" in err.value for err in app.error)


def test_dashboard_what_if_button(loaded) -> None:
    app = run_app(loaded)
    next(b for b in app.sidebar.button if b.label == "Run what-if").click().run()
    assert not app.exception
    assert any("What-if" in md.value for md in app.markdown)
    assert any("Shock taken from history" in info.value for info in app.info)
    assert "Average past outcome" in [m.label for m in app.metric]


def test_dashboard_what_if_with_the_matrix(loaded) -> None:
    app = run_app(loaded)
    app.sidebar.radio[0].set_value("matrix")
    next(b for b in app.sidebar.button if b.label == "Run what-if").click().run()
    assert not app.exception
    assert not any("Shock taken from history" in info.value for info in app.info)
