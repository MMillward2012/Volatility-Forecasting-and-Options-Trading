import copy

import numpy as np
import pandas as pd
import pytest

import src.surface_factor_trade as diagnostic
from src.market_state import spx_trading_dates
from src.surface_factor_trade import (
    FROZEN_RR_SIGNAL, build_synthetic_rr, cumulative_results, direct_factor_outcomes,
    evaluate_surface_trades, frozen_signals, offset_summary, reprice_held_basket,
    roll_step, run_surface_factor_2025, solve_25_delta, start_position, surface_inputs,
    development_sanity,
)


def prices(dates, closes=None):
    return pd.DataFrame({"date": dates, "close": 100. if closes is None else closes,
                         "ticker": "SPX", "secid": 108105})


def snapshot(date, spot=100., iv=.2, constant_variance=False):
    k = np.linspace(-.5, .25, 151)
    tau = np.array([14., 21., 35., 49., 63., 91., 180.]) / 365
    w = np.full((len(tau), len(k)), .004) if constant_variance else tau[:, None] * np.full((len(tau), len(k)), iv**2)
    return {"quote_date": pd.Timestamp(date),
            "grid": {"time_to_expiry": tau, "log_moneyness": k,
                     "sampled_support_min": np.full(len(tau), k[0]),
                     "sampled_support_max": np.full(len(tau), k[-1]),
                     "raw_total_variance": w.copy(), "repaired_total_variance": w.copy()},
            "forwards": pd.DataFrame({"time_to_expiry": tau, "forward": spot, "discount_factor": 1.}),
            "metrics": pd.DataFrame({"target_days": [30, 60, 90], "rr25_valid": True})}


@pytest.fixture
def factor_data():
    dates = pd.bdate_range("2023-01-03", periods=20)
    states = pd.DataFrame({"quote_date": dates, "rr25_30_60": np.arange(20.) / 100})
    signals = pd.DataFrame({"quote_date": dates, "session_index": np.arange(20),
                            "forecast": np.where(np.arange(20) % 2, -1., 1.)})
    return dates, states, signals, prices(dates)


def test_factor_timing_sign_reconciliation_and_final_origins(factor_data):
    dates, states, signals, spx = factor_data
    actual = direct_factor_outcomes(signals, states, spx)
    assert actual.entry_date.iloc[0] == dates[1]
    assert actual.exit_date.iloc[0] == dates[5]
    assert actual.factor_pnl.iloc[0] == pytest.approx(.04)
    assert actual.factor_pnl.iloc[1] == pytest.approx(-.04)
    np.testing.assert_allclose(actual.full_horizon_factor_pnl.dropna(),
                               (actual.missed_first_session + actual.factor_pnl).dropna(), atol=1e-16)
    assert not actual.factor_available.iloc[-5:].any()
    assert actual.factor_pnl.iloc[-5:].isna().all()
    np.testing.assert_array_equal(actual.session_index, np.arange(20))


def test_factor_missing_endpoint_or_entire_date_never_fills(factor_data):
    dates, states, signals, spx = factor_data
    states = states.loc[states.quote_date != dates[1]].copy()
    states.loc[states.quote_date == dates[6], "rr25_30_60"] = np.nan
    actual = direct_factor_outcomes(signals, states, spx)
    assert actual.factor_pnl.iloc[:2].isna().all()
    assert np.isfinite(actual.full_horizon_factor_pnl.iloc[0])
    assert actual.factor_reason.iloc[0] == "missing_qc_rr_endpoint"
    np.testing.assert_array_equal(actual.offset, np.arange(20) % 5)


@pytest.mark.parametrize("forecast, signs", [(1., [1, -1, -1, 1]), (-1., [-1, 1, 1, -1])])
def test_delta_roots_exact_tenors_and_unit_vega(forecast, signs):
    state = snapshot("2023-01-03")
    legs = build_synthetic_rr(state, 100., forecast)
    np.testing.assert_allclose(legs.forward_delta, [-.25, .25, -.25, .25], atol=1e-10)
    np.testing.assert_allclose(legs.time_to_expiry, np.array([30, 30, 60, 60]) / 365, atol=0)
    np.testing.assert_array_equal(np.sign(legs.quantity), signs)
    np.testing.assert_allclose(legs.gross_vega, .25, atol=1e-14)
    assert legs.gross_vega.sum() == pytest.approx(1.)
    assert legs.net_vega.sum() == pytest.approx(0., abs=1e-14)
    assert not {"option_id", "bid", "ask"}.intersection(legs)


def test_surface_inputs_same_bracket_and_log_discount():
    state = snapshot("2023-01-03")
    tau = state["grid"]["time_to_expiry"]
    state["forwards"]["forward"] = 100 + 10 * tau
    state["forwards"]["discount_factor"] = np.exp(-.04 * tau)
    inputs = surface_inputs(state, 30 / 365)
    assert inputs["forward"] == pytest.approx(100 + 10 * 30 / 365)
    assert inputs["discount_factor"] == pytest.approx(np.exp(-.04 * 30 / 365))


def test_no_extrapolation_large_gap_or_invalid_rr():
    state = snapshot("2023-01-03")
    with pytest.raises(ValueError, match="outside_maturity_range"):
        surface_inputs(state, 10 / 365)
    narrow = copy.deepcopy(state)
    narrow["grid"]["sampled_support_max"][:] = 0.
    with pytest.raises(ValueError, match="25_delta_points_unsupported"):
        solve_25_delta(narrow, 30, "call")
    state["metrics"].loc[state["metrics"].target_days == 30, "rr25_valid"] = False
    with pytest.raises(ValueError, match="invalid_30d_rr25"):
        build_synthetic_rr(state, 100., 1.)
    with pytest.raises(ValueError, match="excessive_maturity_gap"):
        surface_inputs(snapshot("2023-01-03"), 120 / 365)


def test_old_basket_ages_over_weekend_new_resets_delta_tenor_and_vega():
    friday = snapshot("2023-01-06")
    position = start_position(friday, 100., 1.)
    monday = snapshot("2023-01-09", spot=102., iv=.24)
    step = roll_step(position["legs"], position["spot"], position["hedge"], position["cash"], monday, 102., 1.)
    np.testing.assert_allclose(step["aged_maturities"] * 365, [27, 27, 57, 57], atol=1e-12)
    np.testing.assert_allclose(step["legs"].time_to_expiry * 365, [30, 30, 60, 60])
    np.testing.assert_allclose(step["legs"].forward_delta, [-.25, .25, -.25, .25], atol=1e-10)
    assert step["legs"].gross_vega.sum() == pytest.approx(1.)
    assert step["legs"].quote_date.eq(pd.Timestamp("2023-01-09")).all()


def test_roll_is_self_financing_previous_hedge_and_attribution_reconcile():
    position = start_position(snapshot("2023-01-03"), 100., 1.)
    # An arbitrary previous hedge makes the interval ordering observable even on flat smiles.
    old_hedge = .123
    old_cash = -np.dot(position["legs"].quantity, position["legs"].model_price) - old_hedge * 100.
    today = snapshot("2023-01-04", spot=103., iv=.25)
    step = roll_step(position["legs"], 100., old_hedge, old_cash, today, 103., 1.)
    assert step["hedge_pnl"] == pytest.approx(.123 * 3)
    assert step["hedge"] == pytest.approx(-step["legs"].position_delta.sum())
    assert step["cash_equity"] == pytest.approx(step["delta_hedged_pnl"], abs=1e-12)
    assert step["accounting_residual"] == pytest.approx(0., abs=1e-12)
    assert step["model_option_pnl"] == pytest.approx(
        step["age_repricing"] + step["forward_discount_repricing"] + step["held_iv_repricing"], abs=1e-12)
    changed = copy.deepcopy(today)
    changed["metrics"]["rr25_downside"] = [-1e6, 1e6, 0]
    repeated = roll_step(position["legs"], 100., old_hedge, old_cash, changed, 103., 1.)
    assert repeated["hedge_pnl"] == step["hedge_pnl"]


def test_zero_price_movement_has_no_spurious_roll_profit_or_label_cash():
    first = snapshot("2023-01-03", constant_variance=True)
    position = start_position(first, 100., 1.)
    legs = position["legs"].copy()
    legs["strike_label"] = "completely different label"
    legs["option_id"] = np.arange(4) + 999
    step = roll_step(legs, 100., position["hedge"], position["cash"],
                     snapshot("2023-01-04", constant_variance=True), 100., 1.)
    assert step["model_option_pnl"] == pytest.approx(0., abs=1e-12)
    assert step["hedge_pnl"] == 0
    assert step["cash_equity"] == pytest.approx(0., abs=1e-12)
    # Under a fixed-IV surface, genuine time decay may remain; only the roll itself is free.
    repriced = reprice_held_basket(legs, snapshot("2023-01-04"), 100.)
    assert np.isfinite(repriced["value"])


def test_future_and_outcome_fields_do_not_affect_current_basket():
    state = snapshot("2023-01-03")
    before = build_synthetic_rr(state, 100., 1.)
    changed = copy.deepcopy(state)
    changed["future_surface"] = snapshot("2023-01-04", iv=10.)
    changed["actual_outcome"] = -100.
    changed["forecast"] = -999.
    pd.testing.assert_frame_equal(before, build_synthetic_rr(changed, 100., 1.))


def test_today_delta_changes_only_the_following_hedge_interval():
    initial = start_position(snapshot("2023-01-03"), 100., 1.)
    fixed = roll_step(initial["legs"], 100., initial["hedge"], initial["cash"],
                      snapshot("2023-01-04", spot=101., iv=.2), 101., 1.)
    changed = roll_step(initial["legs"], 100., initial["hedge"], initial["cash"],
                        snapshot("2023-01-04", spot=101., iv=.4), 101., 1.)
    assert fixed["hedge_pnl"] == changed["hedge_pnl"]
    # Exact forward-25-delta/unit-vega wings have the same reset spot hedge
    # at a given spot, even when the smile's volatility level changes.
    assert fixed["hedge"] == pytest.approx(changed["hedge"], abs=1e-12)
    assert fixed["hedge"] != pytest.approx(initial["hedge"], abs=1e-12)
    for current in (fixed, changed):
        following = roll_step(current["legs"], 101., current["hedge"], current["cash"],
                              snapshot("2023-01-05", spot=103.), 103., 1., close=True)
        assert following["hedge_pnl"] == pytest.approx(current["hedge"] * 2)


def test_production_snapshot_optional_return_keeps_default_api():
    from tests.test_historical_pipeline import raw_options
    from src.historical_pipeline import process_quote_date
    raw = raw_options("2023-01-03")
    ordinary = process_quote_date(raw, "2023-01-03")
    retained = process_quote_date(raw, "2023-01-03", retain_surface=True)
    assert len(ordinary) == 2 and len(retained) == 3
    pd.testing.assert_frame_equal(ordinary[0], retained[0])
    assert retained[2]["grid"] is not None
    assert np.isfinite(retained[2]["forwards"].forward).all()


def test_sanity_rejects_non_development_data_before_reading():
    dates = pd.DatetimeIndex(["2023-01-03", "2024-12-31"])
    daily = pd.DataFrame({"quote_date": dates, "rr25_30": 0., "rr25_60": 0.})
    with pytest.raises(ValueError, match="2023 and 2024"):
        development_sanity(["spx_option_prices_2023.csv", "spx_option_prices_2025.csv"], daily, prices(dates))
    daily.loc[2] = [pd.Timestamp("2025-01-02"), 0., 0.]
    with pytest.raises(ValueError, match="development-only"):
        development_sanity(["spx_option_prices_2023.csv", "spx_option_prices_2024.csv"], daily, prices(dates))


def test_entire_lifecycle_exact_four_intervals_frozen_sign_and_cash():
    dates = pd.bdate_range("2023-01-03", periods=8)
    spx = prices(dates, 100 + np.arange(8.))
    signals = pd.DataFrame({"quote_date": dates, "session_index": np.arange(8), "forecast": 1.})
    stream = [(d, snapshot(d, spot=100 + i, iv=.2 + .001 * i)) for i, d in enumerate(dates)]
    result = evaluate_surface_trades(signals, spx, stream)
    trades = result["synthetic_trades"]
    assert trades.status.eq("evaluated").sum() == 3
    assert trades.reason.iloc[-5:].eq("no_full_exit_horizon").all()
    intervals = result["intervals"]
    for origin, rows in intervals.groupby("origin_date"):
        assert len(rows) == 4
        trade = trades.loc[trades.origin_date == origin].iloc[0]
        assert trade.cash_equity == pytest.approx(rows.delta_hedged_pnl.sum(), abs=1e-12)
        assert rows.next_hedge.iloc[-1] == 0
        np.testing.assert_allclose(rows.previous_hedge.iloc[1:], rows.next_hedge.iloc[:-1])
    np.testing.assert_array_equal(trades.offset, np.arange(8) % 5)


def test_missing_future_surface_keeps_entry_then_marks_unevaluable():
    dates = pd.bdate_range("2023-01-03", periods=8)
    signals = pd.DataFrame({"quote_date": dates[:1], "session_index": [0], "forecast": [1.]})
    stream = [(d, None if i == 3 else snapshot(d)) for i, d in enumerate(dates)]
    result = evaluate_surface_trades(signals, prices(dates), stream)
    row = result["synthetic_trades"].iloc[0]
    assert row.entry_available
    assert row.status == "unevaluable"
    assert "missing_production_surface" in row.reason
    assert np.isnan(row.delta_hedged_pnl)
    assert result["baskets"].quote_date.min() == dates[1]


def test_missing_intermediate_session_is_not_compressed():
    dates = pd.bdate_range("2023-01-03", periods=8)
    signals = pd.DataFrame({"quote_date": dates[:1], "session_index": [0], "forecast": [1.]})
    stream = [(d, snapshot(d)) for i, d in enumerate(dates) if i != 3]
    with pytest.raises(ValueError, match="cannot bridge"):
        evaluate_surface_trades(signals, prices(dates), stream)


def test_future_surface_does_not_change_earlier_baskets_or_intervals():
    dates = pd.bdate_range("2023-01-03", periods=8)
    signals = pd.DataFrame({"quote_date": dates[:1], "session_index": [0], "forecast": [1.]})
    stream = [(d, snapshot(d, spot=100 + i)) for i, d in enumerate(dates)]
    original = evaluate_surface_trades(signals, prices(dates, 100 + np.arange(8.)), stream)
    future = [(d, snapshot(d, spot=100 + i, iv=1.) if i >= 4 else s) for i, (d, s) in enumerate(stream)]
    changed = evaluate_surface_trades(signals, prices(dates, 100 + np.arange(8.)), future)
    for key in ("baskets", "intervals"):
        date_column = "quote_date"
        pd.testing.assert_frame_equal(original[key].loc[original[key][date_column] < dates[4]],
                                      changed[key].loc[changed[key][date_column] < dates[4]])


def test_frozen_signal_never_fits_and_zscore_is_causal(monkeypatch):
    dates = pd.bdate_range("2023-01-03", "2025-01-31")
    daily = pd.DataFrame({"quote_date": dates, "rr25_30_60": .01 * np.sin(np.arange(len(dates)) / 7)})
    spx = prices(dates)
    import src.trade_selection as selection
    monkeypatch.setattr(selection, "fit_rr_spread_signal", lambda *args: pytest.fail("No fitting allowed"))
    original = frozen_signals(daily, spx)
    changed = daily.copy()
    changed["future_target"] = -1e12
    changed.loc[changed.quote_date >= "2025-01-20", "rr25_30_60"] = 100
    after = frozen_signals(changed, spx)
    pd.testing.assert_frame_equal(original.loc[original.quote_date < "2025-01-20"],
                                  after.loc[after.quote_date < "2025-01-20"])
    assert FROZEN_RR_SIGNAL == {"intercept": .0006537436032636668, "slope": -.0017253059632106394}


def test_offset_summary_preserves_gaps_and_cumulative_realisation(factor_data):
    _, states, signals, spx = factor_data
    result = direct_factor_outcomes(signals, states, spx)
    result.loc[result.session_index == 5, "factor_pnl"] = np.nan
    offsets = offset_summary(result, ["factor_pnl"])
    assert offsets.loc[offsets.offset == 0, "n"].iloc[0] == 2
    curve = cumulative_results(result, "factor_pnl")
    assert curve.exit_date.is_monotonic_increasing
    assert curve.cumulative_pnl.iloc[-1] == pytest.approx(result.factor_pnl.sum())


def test_real_run_guard_is_exclusive_and_never_refits(tmp_path, monkeypatch):
    dates = pd.DatetimeIndex(["2023-01-03", "2024-12-31", "2025-01-02", "2025-01-03",
                              "2025-01-06", "2025-01-07", "2025-01-08", "2025-08-29"])
    spx = prices(dates)
    daily = pd.DataFrame({"quote_date": dates, "rr25_30_60": .01 * np.arange(len(dates))})
    signals = pd.DataFrame({"quote_date": dates[2:], "session_index": np.arange(2, 8), "forecast": 1.})
    monkeypatch.setattr(diagnostic, "frozen_signals", lambda *args: signals)
    monkeypatch.setattr(diagnostic, "production_snapshots", lambda *args: ((d, snapshot(d)) for d in dates[2:]))
    result = run_surface_factor_2025("spx_option_prices_2025.csv", daily, spx, "a" * 40, tmp_path)
    assert len(result["direct_factors"]) == 6
    with pytest.raises(FileExistsError):
        run_surface_factor_2025("spx_option_prices_2025.csv", daily, spx, "a" * 40, tmp_path)
