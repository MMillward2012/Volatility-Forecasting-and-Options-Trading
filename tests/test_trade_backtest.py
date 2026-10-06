import numpy as np
import pandas as pd
import pytest

from src.pricing import black_scholes_greeks
from src.trade_backtest import account_trade, check_exit_coverage, trade_schedule
from src.trade_selection import select_rr_spread
from src.trading_feasibility import run_development_feasibility
from tests.test_trade_selection import synthetic_chain, synthetic_prices


@pytest.fixture
def trade():
    dates = pd.bdate_range("2023-01-03", periods=10)
    prices = synthetic_prices(dates)
    prices["close"] = 100 + np.arange(len(dates))
    schedule = trade_schedule(dates[0], prices)
    legs = select_rr_spread(synthetic_chain(dates[1]), 1)["legs"]
    marks = []
    for index, date in enumerate(dates[1:]):
        rows = legs.drop(columns=["quantity", "gross_vega", "net_vega", "position_delta"]).copy()
        rows["quote_date"] = date
        rows["time_to_expiry"] = (rows.expiry_date - date).dt.days / 365
        rows["forward"] = 100 + index
        rows["best_bid"] += .03 * index
        rows["best_ask"] += .03 * index
        marks.append(rows)
    return legs, pd.concat(marks, ignore_index=True), prices, schedule


def test_execution_uses_correct_sides_and_fixed_quantities(trade):
    legs, marks, prices, schedule = trade
    result = account_trade(legs, marks, prices, schedule)
    assert result["status"] == "evaluated"
    q = legs.quantity.to_numpy()
    entry = marks.loc[marks.quote_date.eq(schedule["entry_date"])]
    exit_rows = marks.loc[marks.quote_date.eq(schedule["scheduled_exit"])]
    expected = 100 * np.sum(q * (np.where(q > 0, exit_rows.best_bid, exit_rows.best_ask)
                                - np.where(q > 0, entry.best_ask, entry.best_bid)))
    assert result["option_pnl"] == pytest.approx(expected)
    assert result["execution_drag"] == pytest.approx(-100 * .02 * abs(q).sum())
    assert result["total_pnl"] == pytest.approx(result["option_pnl"] + result["hedge_pnl"])
    changed = marks.copy()
    changed["quantity"] = 999  # Future marks cannot rebalance original option quantities.
    assert account_trade(legs, changed, prices, schedule)["option_pnl"] == pytest.approx(expected)


def test_exit_requires_same_contracts_and_at_most_one_fallback(trade):
    legs, marks, prices, schedule = trade
    marks = marks.loc[~((marks.quote_date == schedule["scheduled_exit"]) & (marks.option_id == legs.option_id.iloc[0]))]
    assert check_exit_coverage(legs, marks, schedule)["status"] == "fallback"
    result = account_trade(legs, marks, prices, schedule)
    # A missing scheduled quote also prevents the mandatory delayed-exit hedge rebalance.
    assert result["reason"] == "missing_daily_hedge_inputs"
    marks = marks.loc[marks.quote_date != schedule["fallback_date"]]
    assert check_exit_coverage(legs, marks, schedule)["status"] == "unevaluable"
    assert account_trade(legs, marks, prices, schedule)["reason"] == "missing_exit_quotes"
    replacement = marks.loc[marks.quote_date == schedule["scheduled_exit"]].iloc[:1].copy()
    replacement["option_id"] = legs.option_id.iloc[0]
    replacement["strike"] += 5
    marks = pd.concat([marks, replacement], ignore_index=True)
    assert check_exit_coverage(legs, marks, schedule)["status"] == "unevaluable"


def test_one_session_fallback_can_be_accounted_with_valid_hedge_inputs(trade):
    legs, marks, prices, schedule = trade
    # Zero bid prevents execution under the protocol but permits a midpoint Greek.
    marks.loc[marks.quote_date == schedule["scheduled_exit"], "best_bid"] = 0
    result = account_trade(legs, marks, prices, schedule)
    assert result["exit_coverage"]["status"] == "fallback"
    assert result["status"] == "evaluated"
    assert result["hedges"].quote_date.iloc[-1] == schedule["fallback_date"]


def test_hedge_uses_previous_eod_delta_and_unwinds_at_exit(trade):
    legs, marks, prices, schedule = trade
    result = account_trade(legs, marks, prices, schedule)
    history = result["hedges"]
    entry = marks.loc[marks.quote_date == schedule["entry_date"]]
    spot = prices.set_index("date").close
    greek = black_scholes_greeks(entry.forward, entry.strike, entry.discount_factor,
                                entry.time_to_expiry, entry.mid_iv, entry.option_type,
                                spot.loc[schedule["entry_date"]])
    hedge = -100 * np.dot(legs.quantity, greek["spot_delta"])
    assert history.hedge_position.iloc[0] == pytest.approx(hedge)
    assert history.hedge_pnl.iloc[1] == pytest.approx(hedge * (spot.iloc[2] - spot.iloc[1]))
    assert history.hedge_pnl.iloc[0] == 0
    assert history.hedge_position.iloc[-1] == 0
    np.testing.assert_allclose(history.previous_hedge.iloc[1:], history.hedge_position.iloc[:-1])
    assert result["hedge_turnover"] == pytest.approx(history.hedge_turnover.sum())


def test_today_delta_only_changes_following_hedge(trade):
    legs, marks, prices, schedule = trade
    before = account_trade(legs, marks, prices, schedule)["hedges"]
    day = prices.date.iloc[3]
    changed = marks.copy()
    changed.loc[changed.quote_date >= day, "mid_iv"] = .5
    after = account_trade(legs, changed, prices, schedule)["hedges"]
    np.testing.assert_allclose(before.loc[before.quote_date <= day, "hedge_pnl"],
                               after.loc[after.quote_date <= day, "hedge_pnl"])
    assert before.loc[before.quote_date == day, "hedge_position"].iloc[0] != pytest.approx(
        after.loc[after.quote_date == day, "hedge_position"].iloc[0])


def test_missing_spot_or_intermediate_marks_never_bridge(trade):
    legs, marks, prices, schedule = trade
    missing = prices.copy()
    missing.loc[2, "close"] = np.nan
    assert account_trade(legs, marks, missing, schedule)["reason"] == "missing_spx_close"
    missing_marks = marks.loc[marks.quote_date != prices.date.iloc[2]]
    assert account_trade(legs, missing_marks, prices, schedule)["reason"] == "missing_daily_hedge_inputs"


def test_missing_future_exit_does_not_affect_entry_selection(trade):
    legs, marks, prices, schedule = trade
    before = select_rr_spread(synthetic_chain(schedule["entry_date"]), 1)["legs"]
    marks = marks.loc[marks.quote_date < schedule["scheduled_exit"]]
    assert check_exit_coverage(legs, marks, schedule)["status"] == "unevaluable"
    after = select_rr_spread(synthetic_chain(schedule["entry_date"]), 1)["legs"]
    pd.testing.assert_frame_equal(before, after)


def test_marks_after_completed_exit_cannot_affect_accounting(trade):
    legs, marks, prices, schedule = trade
    expected = account_trade(legs, marks, prices, schedule)
    duplicate_future = marks.loc[marks.quote_date == schedule["fallback_date"]].iloc[:1]
    changed = pd.concat([marks, duplicate_future], ignore_index=True)
    actual = account_trade(legs, changed, prices, schedule)
    assert actual["total_pnl"] == pytest.approx(expected["total_pnl"])


def test_exit_is_whole_basket_on_one_common_date(trade):
    legs, marks, _, schedule = trade
    first, second = legs.option_id.iloc[:2]
    marks.loc[(marks.quote_date == schedule["scheduled_exit"]) & (marks.option_id == first), "best_bid"] = 0
    marks.loc[(marks.quote_date == schedule["fallback_date"]) & (marks.option_id == second), "best_bid"] = 0
    assert check_exit_coverage(legs, marks, schedule)["status"] == "unevaluable"


def test_session_offsets_use_original_calendar_even_if_options_absent():
    dates = pd.bdate_range("2023-01-03", periods=20)
    prices = synthetic_prices(dates)
    for index in (0, 4, 5, 9):
        schedule = trade_schedule(dates[index], prices)
        assert schedule["session_index"] == index
        assert schedule["offset"] == index % 5
        assert schedule["entry_date"] == dates[index + 1]
        assert schedule["scheduled_exit"] == dates[index + 5]
        assert schedule["fallback_date"] == dates[index + 6]


def test_duplicate_identity_raises(trade):
    legs, marks, _, schedule = trade
    with pytest.raises(ValueError, match="unique"):
        check_exit_coverage(legs, pd.concat([marks, marks.loc[marks.quote_date == schedule["scheduled_exit"]].iloc[:1]]), schedule)


def test_feasibility_rejects_2025_before_reading_any_files():
    prices = synthetic_prices(pd.bdate_range("2023-01-03", periods=20))
    with pytest.raises(ValueError, match="2023 and 2024"):
        run_development_feasibility(["spx_option_prices_2023.csv", "spx_option_prices_2025.csv"], prices)
    prices.loc[len(prices)] = [pd.Timestamp("2025-01-02"), 100, "SPX", 108105]
    with pytest.raises(ValueError, match="development-only"):
        run_development_feasibility(["spx_option_prices_2023.csv", "spx_option_prices_2024.csv"], prices)
