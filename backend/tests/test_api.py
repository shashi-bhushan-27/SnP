from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prism.api.app import create_app
from prism.bootstrap import build_container
from prism.storage import SqlStore


@pytest.fixture
def client(settings, engine):
    container = build_container(settings, store=SqlStore("sqlite://"), engine=engine)
    with TestClient(create_app(container)) as client:
        yield client


def run_pipeline(client: TestClient) -> dict:
    response = client.post("/api/pipeline/run", json={"limit": 100})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_reports_active_components(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert (body["sentiment_model"], body["event_classifier"], body["impact_model"]) == ("lexicon", "rules", "weighted-v1")
    assert body["sources"] == ["replay"]


def test_analyze_returns_structured_signals_and_rejections(client) -> None:
    response = client.post("/api/analyze", json={"texts": ["Tesla shares rise as quarterly deliveries beat analyst estimates", "tiny"]})
    body = response.json()
    (signal,) = body["signals"]
    assert signal["ticker"] == "TSLA" and signal["event_type"] == "Earnings"
    assert -1 <= signal["sentiment_score"] <= 1 and 1 <= signal["impact_score"] <= 10 and 0 <= signal["confidence"] <= 1
    assert [r["reason"] for r in body["rejected"]] == ["too_short"]
    assert client.get("/api/risk-signals").json() == []  # not persisted unless asked


def test_analyze_can_persist(client) -> None:
    client.post("/api/analyze", json={"texts": ["Tesla shares rise as quarterly deliveries beat analyst estimates"], "persist": True})
    assert len(client.get("/api/risk-signals").json()) == 1


def test_analyze_validates_the_request(client) -> None:
    assert client.post("/api/analyze", json={"texts": []}).status_code == 422


def test_pipeline_then_signal_queries(client) -> None:
    report = run_pipeline(client)
    assert report["signals_loaded"] > 0 and report["errors"] == []

    signals = client.get("/api/risk-signals", params={"limit": 500}).json()
    assert len(signals) == report["signals_loaded"]
    assert all(s["impact_score"] >= 1 for s in signals)

    tsla = client.get("/api/risk-signals", params={"symbol": "tsla"}).json()
    assert tsla and {s["ticker"] for s in tsla} == {"TSLA"}

    high = client.get("/api/risk-signals", params={"min_impact": 7}).json()
    assert high and all(s["impact_score"] >= 7 for s in high)

    credit = client.get("/api/risk-signals", params={"event_type": "Credit Event"}).json()
    assert credit and {s["event_type"] for s in credit} == {"Credit Event"}

    first = signals[0]
    assert client.get(f"/api/risk-signals/{first['id']}").json()["id"] == first["id"]
    assert client.get("/api/risk-signals/does-not-exist").status_code == 404


def test_latest_signal_for_a_symbol(client) -> None:
    assert client.get("/api/risk-signals/latest").status_code == 404
    run_pipeline(client)
    assert client.get("/api/risk-signals/latest", params={"symbol": "TSLA"}).json()["ticker"] == "TSLA"
    assert client.get("/api/risk-signals/latest", params={"symbol": "ZZZZ"}).status_code == 404


def test_stats_and_news(client) -> None:
    run_pipeline(client)
    stats = client.get("/api/stats").json()
    assert stats["signals"] > 0 and stats["documents"] >= stats["signals"] - 2
    assert stats["high_risk_signals"] > 0 and 1 <= stats["avg_impact"] <= 10
    # positive high-impact news (e.g. record earnings) counts as impact, not as risk
    assert stats["high_risk_signals"] < stats["high_impact_signals"]
    assert "Geopolitical" in stats["signals_by_event"]
    assert len(client.get("/api/news", params={"limit": 5}).json()) == 5


def test_pipeline_run_history(client) -> None:
    run_pipeline(client)
    runs = client.get("/api/pipeline/runs").json()
    assert len(runs) == 1 and runs[0]["sources"]["replay"]["fetched"] > 0


def test_pipeline_triggers_stress_tests_automatically(client) -> None:
    run_pipeline(client)
    runs = client.get("/api/stress-runs").json()
    assert {"Geopolitical", "Market Shock"} <= {r["event_type"] for r in runs}
    assert all(r["pnl"] < 0 for r in runs)


def test_portfolio_and_scenarios(client) -> None:
    portfolio = client.get("/api/portfolio").json()
    assert sum(a["value"] for a in portfolio["assets"]) == 60_000_000
    assert {s["event_type"] for s in client.get("/api/scenarios").json()} >= {"Geopolitical", "Credit Event"}


def test_manual_stress_test(client) -> None:
    response = client.post("/api/stress-test", json={"event_type": "Geopolitical", "impact_score": 9})
    assert response.status_code == 200
    body = response.json()
    assert body["pnl"] == pytest.approx(-5_200_000) and body["impact_score"] == 9
    assert len(body["by_asset"]) == 7
    assert len(client.get("/api/stress-runs").json()) == 1  # persisted


def test_manual_stress_test_with_shock_override_and_no_persist(client) -> None:
    body = client.post(
        "/api/stress-test", json={"event_type": "Geopolitical", "shock": {"equity_pct": -0.2}, "persist": False}
    ).json()
    assert body["shock"]["equity_pct"] == -0.2 and body["shock"]["rate_bps"] == 0
    assert client.get("/api/stress-runs").json() == []


def test_stress_test_errors(client) -> None:
    assert client.post("/api/stress-test", json={"event_type": "Product Launch"}).status_code == 404
    assert client.post("/api/stress-test", json={"event_type": "Nonsense"}).status_code == 422
