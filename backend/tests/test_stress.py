from __future__ import annotations

import pytest

from conftest import make_signal
from prism.core.taxonomy import EventType
from prism.modules.stress import Asset, Portfolio, Scenario, ScenarioBook, Shock, StressEngine, StressTrigger
from prism.modules.stress.engine import asset_pnl


@pytest.fixture
def portfolio(config_dir) -> Portfolio:
    return Portfolio.from_yaml(config_dir / "portfolio.yaml")


@pytest.fixture
def book(config_dir) -> ScenarioBook:
    return ScenarioBook.from_yaml(config_dir / "scenarios.yaml")


# ---------------------------------------------------------------- valuation maths
def test_default_portfolio_is_60m_with_all_four_asset_types(portfolio) -> None:
    assert portfolio.total_value == 60_000_000
    assert {a.asset_type for a in portfolio.assets} == {"loan", "bond", "equity", "derivative"}


def test_geopolitical_scenario_matches_hand_computation(portfolio, book) -> None:
    """equity -10%, rates +200bp, spreads +150bp, valued asset by asset (see engine docstring)."""
    result = StressEngine().run(portfolio, book.for_event(EventType.GEOPOLITICAL))
    by_asset = {i.asset_id: i.pnl for i in result.by_asset}
    assert by_asset["LN-CORP-01"] == pytest.approx(-400_000)
    assert by_asset["BD-GOV-01"] == pytest.approx(-1_920_000)  # 15m * (-7*0.02 + 0.5*60*0.02^2)
    assert by_asset["BD-CORP-01"] == pytest.approx(-1_680_000)
    assert by_asset["EQ-01"] == pytest.approx(-1_100_000)  # 10m * 1.1 * -10%
    assert by_asset["DR-IRS-01"] == pytest.approx(500_000)  # 2500 per bp * 200bp: the hedge pays out
    assert result.pnl == pytest.approx(-5_200_000)
    assert result.value_before == 60_000_000
    assert result.value_after == pytest.approx(54_800_000)
    assert result.pnl_pct == pytest.approx(-5.2 / 60)


def test_pnl_is_broken_down_by_asset_type(portfolio, book) -> None:
    result = StressEngine().run(portfolio, book.for_event(EventType.GEOPOLITICAL))
    assert result.by_type == pytest.approx({"loan": -1_000_000, "bond": -3_600_000, "equity": -1_100_000, "derivative": 500_000})
    assert sum(result.by_type.values()) == pytest.approx(result.pnl)


def test_zero_shock_changes_nothing(portfolio, book) -> None:
    result = StressEngine().run(portfolio, book.for_event(EventType.GEOPOLITICAL), shock=Shock())
    assert result.pnl == 0 and result.value_after == result.value_before


def test_shock_override_replaces_the_scenario_shock(portfolio, book) -> None:
    result = StressEngine().run(portfolio, book.for_event(EventType.GEOPOLITICAL), shock=Shock(equity_pct=-0.5))
    assert {i.asset_id: i.pnl for i in result.by_asset}["EQ-01"] == pytest.approx(-5_500_000)


def test_an_asset_cannot_lose_more_than_its_value() -> None:
    equity = Asset(id="e", name="e", asset_type="equity", value=1_000_000)
    assert asset_pnl(equity, Shock(equity_pct=-5.0)) == -1_000_000


def test_equity_beta_defaults_to_one_only_for_equities() -> None:
    assert Asset(id="e", name="e", asset_type="equity", value=1).effective_beta == 1.0
    assert Asset(id="b", name="b", asset_type="bond", value=1).effective_beta == 0.0


def test_rate_decline_helps_a_long_duration_bond() -> None:
    bond = Asset(id="b", name="b", asset_type="bond", value=1_000_000, duration=7.0, convexity=60)
    assert asset_pnl(bond, Shock(rate_bps=-100)) > 0


# ---------------------------------------------------------------- trigger rules
def test_every_scenario_in_the_book_is_valid(book) -> None:
    scenarios = book.all()
    assert len(scenarios) >= 5
    assert len({s.event_type for s in scenarios}) == len(scenarios)
    assert all(s.shock.equity_pct < 0 for s in scenarios)


def test_signal_fires_a_scenario_only_above_the_impact_threshold(book) -> None:
    assert book.match(make_signal(impact=7.01)) is not None
    assert book.match(make_signal(impact=7.0)) is None  # strictly greater than
    assert book.match(make_signal(impact=6.0)) is None


def test_low_confidence_signals_do_not_trigger(book) -> None:
    assert book.match(make_signal(confidence=0.49)) is None
    assert book.match(make_signal(confidence=0.5)) is not None


def test_unconfirmed_single_source_stories_do_not_trigger(book) -> None:
    # one article on an unknown site (reliability 0.5) is not enough...
    assert book.match(make_signal(corroboration=1, reliability=0.5)) is None
    # ...two independent outlets are, and so is one trusted outlet
    assert book.match(make_signal(corroboration=2, reliability=0.5)) is not None
    assert book.match(make_signal(corroboration=1, reliability=0.95)) is not None


def test_positive_news_does_not_stress_the_portfolio(book) -> None:
    assert book.match(make_signal(sentiment=0.6)) is None
    assert book.match(make_signal(sentiment=-0.1)) is None


def test_event_types_without_a_scenario_never_trigger(book) -> None:
    assert book.for_event(EventType.PRODUCT_LAUNCH) is None
    assert book.match(make_signal(event_type=EventType.PRODUCT_LAUNCH, impact=10)) is None


def test_trigger_runs_one_stress_test_per_event_type_using_the_strongest_signal(portfolio, book) -> None:
    results = []
    trigger = StressTrigger(portfolio, book, StressEngine(), on_result=results.append)
    trigger.on_signals(
        [
            make_signal(impact=7.5, signal_id="weak"),
            make_signal(impact=9.1, signal_id="strong"),
            make_signal(impact=8.0, signal_id="mid"),
            make_signal(event_type=EventType.MARKET_SHOCK, impact=8.2, signal_id="crash"),
            make_signal(event_type=EventType.PRODUCT_LAUNCH, impact=9.9, signal_id="ignored"),
            make_signal(impact=5.0, sentiment=-0.9, signal_id="too-weak"),
        ]
    )
    assert sorted((r.event_type.value, r.trigger_signal_id) for r in results) == [("Geopolitical", "strong"), ("Market Shock", "crash")]
    assert next(r for r in results if r.trigger_signal_id == "strong").impact_score == 9.1


def test_trigger_with_no_matching_signal_does_nothing(portfolio, book) -> None:
    results = []
    StressTrigger(portfolio, book, StressEngine(), on_result=results.append).on_signals([make_signal(impact=3.0)])
    assert results == []


def test_scenario_model_round_trips() -> None:
    scenario = Scenario.model_validate({"event_type": "Credit Event", "name": "x", "shock": {"credit_spread_bps": 100}})
    assert scenario.event_type is EventType.CREDIT_EVENT and scenario.trigger.min_impact == 7.0
