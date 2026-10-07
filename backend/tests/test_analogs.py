"""Historical-analog library and history-calibrated scenarios (bag-of-words encoder: no model download)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from conftest import make_signal
from prism.config import BACKEND_ROOT
from prism.core.taxonomy import EventType
from prism.modules.stress import (
    AnalogLibrary,
    HistoricalScenarioBuilder,
    Portfolio,
    ScenarioBook,
    Shock,
    StressEngine,
    StressTrigger,
)
from prism.nlp.embeddings import HashingEncoder

LIBRARY = BACKEND_ROOT / "config" / "analog_library.json"


def event(i, title, event_type, eq, rate, spread, day):
    moves = {h: {"equity_pct": eq, "rate_bps": rate, "credit_spread_bps": spread} for h in ("1d", "5d", "trough")}
    return {"id": f"e{i}", "date": day, "reaction_date": day, "title": title, "event_type": event_type, "moves": moves}


@pytest.fixture
def tiny(tmp_path):
    events = [
        event(1, "Large bank collapses and is seized by regulators", "Bankruptcy", -0.08, -30, 40, "2010-01-04"),
        event(2, "Bank failure triggers deposit run fears", "Bankruptcy", -0.02, -10, 15, "2012-01-03"),
        event(3, "Investment bank files for bankruptcy", "Bankruptcy", -0.05, -20, 30, "2014-01-02"),
        event(4, "Inflation report comes in hotter than expected", "Macroeconomic", -0.03, 25, 2, "2016-01-04"),
        event(5, "Central bank cuts interest rates", "Macroeconomic", 0.01, -15, -3, "2018-01-02"),
        event(6, "Military conflict breaks out between two countries", "Geopolitical", -0.02, -8, 5, "2020-01-02"),
    ]
    path = tmp_path / "lib.json"
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return path


def lib(path, **kw):
    kw.setdefault("min_similarity", 0.0)
    return AnalogLibrary.from_json(path, HashingEncoder(), **kw)


@pytest.fixture
def portfolio(config_dir):
    return Portfolio.from_yaml(config_dir / "portfolio.yaml")


# ---------------------------------------------------------------- retrieval
def test_nearest_ranks_by_meaning_and_weights_sum_to_one(tiny) -> None:
    matches = lib(tiny).nearest("Regional bank collapses after a deposit run", EventType.BANKRUPTCY, k=3)
    assert matches[0].event_type is EventType.BANKRUPTCY
    assert matches[0].similarity >= matches[-1].similarity
    assert sum(m.weight for m in matches) == pytest.approx(1.0, abs=1e-3)


def test_scenario_is_the_weighted_realised_reaction(tiny) -> None:
    scenario = lib(tiny, k=2).scenario("Bank collapses and is seized by regulators", EventType.BANKRUPTCY)
    assert scenario.method == "analog"
    expected = sum(m.weight * m.shock.rate_bps for m in scenario.matches)
    assert scenario.shock.rate_bps == pytest.approx(expected)


def test_dissimilar_headline_falls_back_to_the_type_average(tiny) -> None:
    scenario = lib(tiny, min_similarity=0.99).scenario("Completely unrelated sports result", EventType.BANKRUPTCY)
    assert scenario.method == "type_mean" and scenario.matches == []
    assert scenario.shock.equity_pct == pytest.approx((-0.08 - 0.02 - 0.05) / 3)
    unknown = lib(tiny, min_similarity=0.99).scenario("Completely unrelated sports result", EventType.LEGAL)
    assert unknown.method == "global_mean"


def test_exclude_and_before_hide_events(tiny) -> None:
    library = lib(tiny)
    assert all(m.id != "e1" for m in library.nearest("Large bank collapses", exclude={"e1"}))
    assert all(m.date < date(2015, 1, 1) for m in library.nearest("bank", before=date(2015, 1, 1)))


def test_pool_uses_same_type_only_when_there_are_enough(tiny) -> None:
    library = lib(tiny)
    same, used = library.pool(EventType.BANKRUPTCY, min_pool=3)
    assert used and {e.event_type for e in same} == {EventType.BANKRUPTCY}
    everything, used = library.pool(EventType.GEOPOLITICAL, min_pool=3)
    assert not used and len(everything) == 6


# ---------------------------------------------------------------- calibrated scenarios
def test_stress_is_a_real_event_at_the_chosen_quantile(tiny, portfolio) -> None:
    builder = HistoricalScenarioBuilder(lib(tiny), portfolio, min_pool=3)
    worst = builder.build("bank collapse", EventType.BANKRUPTCY, quantile=0.01)
    median = builder.build("bank collapse", EventType.BANKRUPTCY, quantile=0.5)
    pnls = [o.pnl for o in worst.distribution]
    assert pnls == sorted(pnls)  # worst first
    assert worst.basis.pnl == pnls[0] and worst.shock == worst.basis.shock
    assert median.basis.pnl == pnls[1]
    assert worst.pool_size == 3 and worst.pool_same_type
    assert worst.expected_pnl > worst.basis.pnl


def test_invalid_quantile(tiny, portfolio) -> None:
    with pytest.raises(ValueError):
        HistoricalScenarioBuilder(lib(tiny), portfolio, quantile=1.5)


def test_engine_records_the_shock_source(tiny, portfolio, config_dir) -> None:
    scenario = ScenarioBook.from_yaml(config_dir / "scenarios.yaml").for_event(EventType.BANKRUPTCY)
    engine = StressEngine()
    history = HistoricalScenarioBuilder(lib(tiny), portfolio, min_pool=3).build("bank", EventType.BANKRUPTCY)
    assert engine.run(portfolio, scenario).source == "matrix"
    assert engine.run(portfolio, scenario, shock=Shock(equity_pct=-0.1)).source == "manual"
    result = engine.run(portfolio, scenario, history=history)
    assert result.source == "history" and result.shock == history.shock and result.scenario == history.name


def test_trigger_uses_history_when_wired(tiny, portfolio, config_dir) -> None:
    results = []
    builder = HistoricalScenarioBuilder(lib(tiny), portfolio, min_pool=3)
    book = ScenarioBook.from_yaml(config_dir / "scenarios.yaml")
    StressTrigger(portfolio, book, StressEngine(), results.append, history=builder).on_signals(
        [make_signal(event_type=EventType.BANKRUPTCY, impact=9.0)]
    )
    (result,) = results
    assert result.source == "history" and result.history.basis.event_type is EventType.BANKRUPTCY


# ---------------------------------------------------------------- the committed library
def test_committed_library_is_complete_and_sane() -> None:
    events = json.loads(LIBRARY.read_text(encoding="utf-8"))["events"]
    assert len(events) >= 80 and len({e["id"] for e in events}) == len(events)
    for e in events:
        EventType.parse(e["event_type"])
        assert set(e["moves"]) == {"1d", "5d", "trough"}
        assert e["reaction_date"] >= e["date"]
    lehman = next(e for e in events if e["date"] == "2008-09-15")
    assert lehman["moves"]["1d"]["equity_pct"] < -0.04  # S&P 500 fell 4.7% that day
    assert lehman["moves"]["trough"]["rate_bps"] < 0  # flight to safety


def test_committed_titles_do_not_leak_the_outcome() -> None:
    titles = " | ".join(e["title"].lower() for e in json.loads(LIBRARY.read_text(encoding="utf-8"))["events"])
    for leak in ("stocks plunge", "stocks surge", "stocks soar", "stocks tumble", "and stocks", "dow drops", "dow falls"):
        assert leak not in titles
