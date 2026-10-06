import json

import numpy as np
import pandas as pd
import pytest

from src.pricing import black_scholes_call_price, black_scholes_put_price, black_scholes_greeks
from src.trade_selection import select_rr_spread
from src.trading_locked import (
    PNL_COLUMNS, build_locked_rr_signals, control_positions, control_statistics,
    cumulative_pnl, evaluate_locked_trades, offset_statistics, primary_statistics,
    run_locked_2025, trading_coverage,
)
from tests.test_trade_selection import synthetic_chain, synthetic_prices


@pytest.fixture
def locked_inputs(monkeypatch):
    development = pd.bdate_range("2023-01-03", "2024-12-31")
    confirmation = pd.bdate_range("2025-01-02", periods=10)
    prices = synthetic_prices(development.append(confirmation))
    prices.loc[prices.date.isin(confirmation), "close"] = 100 + .2 * np.arange(10)
    signals = pd.DataFrame({"quote_date": confirmation,
                            "session_index": np.arange(len(development), len(prices)),
                            "forecast": [.1, -.1, .2, -.2, .1, .1, .1, .1, .1, .1]})
    base = synthetic_chain(confirmation[1])
    days = []
    for date in confirmation:
        chain = base.copy()
        chain["quote_date"] = date
        chain["days_to_expiry"] = (chain.expiry_date - date).dt.days
        chain["time_to_expiry"] = chain.days_to_expiry / 365
        spot = float(prices.loc[prices.date == date, "close"].iloc[0])
        greek = black_scholes_greeks(chain.forward, chain.strike, chain.discount_factor,
                                    chain.time_to_expiry, chain.mid_iv, chain.option_type, spot)
        chain["forward_delta"] = greek["forward_delta"]
        chain["option_delta"] = 100 * greek["spot_delta"]
        chain["vega"] = 100 * greek["vega"]
        call = chain.option_type == "call"
        mid = np.where(call,
                       black_scholes_call_price(chain.forward, chain.strike, chain.discount_factor,
                                                chain.time_to_expiry, chain.mid_iv),
                       black_scholes_put_price(chain.forward, chain.strike, chain.discount_factor,
                                               chain.time_to_expiry, chain.mid_iv))
        chain["mid_price"] = mid
        chain["best_bid"] = mid - .01
        chain["best_ask"] = mid + .01
        chain["date"] = date
        days.append((None, date, chain))
    def prepare(raw, spot, held_option_ids=()):
        chain = raw.drop(columns="date").copy()
        quotes = chain.drop(columns=["forward", "discount_factor", "mid_iv",
                                     "forward_delta", "option_delta", "vega", "is_otm"])
        return {"chain": chain, "quotes": quotes}
    monkeypatch.setattr("src.trading_locked.prepare_executable_chain", prepare)
    return signals, prices, days


def test_timing_end_boundary_entry_only_selection_and_offsets(locked_inputs, monkeypatch):
    signals, prices, days = locked_inputs
    selected_dates = []
    def select(chain, forecast):
        selected_dates.append(chain.quote_date.iloc[0])
        return select_rr_spread(chain, forecast)
    monkeypatch.setattr("src.trading_locked.select_rr_spread", select)
    result = evaluate_locked_trades(signals, prices, iter(days))
    trades = result["trades"]
    assert trades.status.eq("evaluated").sum() == 5
    assert trades.reason.eq("no_full_exit_horizon").sum() == 5
    assert selected_dates == signals.quote_date.iloc[1:6].to_list()
    for index in range(5):
        row = trades.iloc[index]
        assert row.entry_date == signals.quote_date.iloc[index + 1]
        assert row.scheduled_exit == signals.quote_date.iloc[index + 5]
        assert row.actual_exit == row.scheduled_exit
        assert row.session_index == signals.session_index.iloc[index]
        assert row.offset == row.session_index % 5
    assert trading_coverage(trades)["scheduled_evaluated_exits"] == 5


@pytest.mark.parametrize("forecast", [.1, -.1])
def test_control_quantities_and_identities(forecast):
    primary = select_rr_spread(synthetic_chain("2025-01-03"), forecast)["legs"]
    positions = control_positions(primary, forecast)
    for name, position in positions.items():
        np.testing.assert_array_equal(position.option_id, primary.option_id)
        np.testing.assert_allclose(position.gross_vega, .25)
        assert position.gross_vega.sum() == pytest.approx(1)
    np.testing.assert_allclose(positions["reversed_sign"].quantity, -primary.quantity)
    np.testing.assert_array_equal(np.sign(positions["constant_positive"].quantity), [1, -1, -1, 1])
    np.testing.assert_array_equal(np.sign(primary.quantity), np.sign(forecast) * np.array([1, -1, -1, 1]))


def test_controls_recompute_hedges_and_midpoints_are_sign_symmetric(locked_inputs):
    result = evaluate_locked_trades(*locked_inputs)
    for origin, rows in result["hedges"].groupby("origin_date"):
        primary = rows.loc[rows.variant == "primary"].reset_index(drop=True)
        reverse = rows.loc[rows.variant == "reversed_sign"].reset_index(drop=True)
        positive = rows.loc[rows.variant == "constant_positive"].reset_index(drop=True)
        sign = result["trades"].loc[result["trades"].origin_date == origin, "sign"].iloc[0]
        np.testing.assert_allclose(reverse.hedge_position, -primary.hedge_position)
        np.testing.assert_allclose(reverse.hedge_pnl, -primary.hedge_pnl)
        np.testing.assert_allclose(positive.hedge_position, sign * primary.hedge_position)
        np.testing.assert_allclose(positive.hedge_pnl, sign * primary.hedge_pnl)
    primary = result["trades"].loc[result["trades"].status == "evaluated"]
    reverse = result["controls"].loc[result["controls"].variant == "reversed_sign"]
    np.testing.assert_allclose(reverse.midpoint_total_pnl, -primary.midpoint_total_pnl)
    assert primary.execution_drag.le(1e-10).all()
    assert result["controls"].execution_drag.le(1e-10).all()
    np.testing.assert_allclose(primary.executable_total_pnl,
                               primary.midpoint_total_pnl + primary.execution_drag)


def test_no_outcomes_or_model_fitting_in_evaluation(locked_inputs, monkeypatch):
    signals, prices, days = locked_inputs
    def forbidden(*args, **kwargs):
        raise AssertionError("Locked execution must not fit a model or inspect target outcomes.")
    monkeypatch.setattr("src.trading_locked.fit_rr_spread_signal", forbidden)
    monkeypatch.setattr("src.trading_locked.forecast_rr_spread_signal", forbidden)
    first = evaluate_locked_trades(signals, prices, days)
    changed = signals.assign(y_rr25_spread_5d=999, future_pnl=-999)
    second = evaluate_locked_trades(changed, prices, days)
    pd.testing.assert_frame_equal(first["trades"], second["trades"])


def test_future_prices_can_change_pnl_but_not_entry_eligibility(locked_inputs):
    signals, prices, days = locked_inputs
    first = evaluate_locked_trades(signals, prices, days)["trades"]
    changed = []
    for path, date, raw in days:
        raw = raw.copy()
        if date >= signals.quote_date.iloc[5]:
            raw["best_bid"] += 20
            raw["best_ask"] += 20
        changed.append((path, date, raw))
    second = evaluate_locked_trades(signals, prices, changed)["trades"]
    columns = ["origin_date", "option_ids", "quantities", "four_wings_available", "status"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    # No trade is removed for a positive or negative realised accounting result.
    assert second.status.eq("evaluated").sum() == 5


def test_missing_origin_signal_does_not_compress_indices(locked_inputs):
    signals, prices, days = locked_inputs
    signals = signals.copy()
    signals.loc[1, "forecast"] = np.nan
    result = evaluate_locked_trades(signals, prices, days)
    assert result["trades"].iloc[1].reason == "no_signal"
    assert result["trades"].iloc[2].session_index == signals.session_index.iloc[2]
    assert result["trades"].iloc[2].offset == signals.session_index.iloc[2] % 5
    assert len(result["trades"]) == len(signals)


def test_missing_entry_and_intermediate_chain_are_not_filled(locked_inputs):
    signals, prices, days = locked_inputs
    missing = [day for index, day in enumerate(days) if index != 2]
    result = evaluate_locked_trades(signals, prices, missing)
    assert result["trades"].iloc[1].reason == "missing_entry_chain"
    assert result["trades"].iloc[0].status == "unevaluable"
    assert result["trades"].iloc[0].reason == "missing_daily_hedge_inputs"
    assert result["trades"].iloc[2].offset == signals.session_index.iloc[2] % 5


def test_scheduled_final_day_valid_and_no_outside_window_fallback(locked_inputs):
    signals, prices, days = locked_inputs
    result = evaluate_locked_trades(signals, prices, days)
    assert result["trades"].iloc[4].actual_exit == signals.quote_date.iloc[-1]
    changed = [(path, date, raw.copy()) for path, date, raw in days]
    changed[-1][2]["best_bid"] = 0
    result = evaluate_locked_trades(signals, prices, changed)
    assert result["trades"].iloc[4].reason == "missing_exit_quotes"
    assert pd.isna(result["trades"].iloc[4].fallback_date)


def test_one_session_fallback_and_unevaluable_quote_path(locked_inputs):
    signals, prices, days = locked_inputs
    changed = [(path, date, raw.copy()) for path, date, raw in days]
    changed[5][2]["best_bid"] = 0
    result = evaluate_locked_trades(signals, prices, changed)
    first = result["trades"].iloc[0]
    assert first.status == "evaluated"
    assert first.exit_status == "fallback"
    assert first.actual_exit == signals.quote_date.iloc[6]
    changed[6][2]["best_bid"] = 0
    result = evaluate_locked_trades(signals, prices, changed)
    assert result["trades"].iloc[0].reason == "missing_exit_quotes"


def test_signal_builder_fits_only_development_and_ignores_outcomes(monkeypatch):
    dates = pd.bdate_range("2023-01-03", "2025-08-29")
    prices = synthetic_prices(dates)
    daily = pd.DataFrame({"quote_date": dates, "rr25_30_60": .02 + .005 * np.sin(np.arange(len(dates)) / 8)})
    fitted, before = build_locked_rr_signals(daily, prices)
    changed = daily.assign(y_rr25_spread_5d=1000)
    changed.loc[changed.quote_date > "2025-03-01", "rr25_30_60"] *= 10
    refitted, after = build_locked_rr_signals(changed, prices)
    assert refitted == fitted
    np.testing.assert_allclose(before.loc[before.quote_date < "2025-03-01", "forecast"],
                               after.loc[after.quote_date < "2025-03-01", "forecast"], equal_nan=True)
    assert fitted["fit_end"] == pd.Timestamp("2024-12-31")
    assert fitted["last_matured_origin"] < fitted["fit_end"]


def test_calendar_validation_rejects_compressed_or_absent_origins(locked_inputs):
    signals, prices, days = locked_inputs
    with pytest.raises(ValueError, match="every confirmation"):
        evaluate_locked_trades(signals.drop(index=1), prices, days)
    compressed = signals.copy()
    compressed["session_index"] -= 1
    with pytest.raises(ValueError, match="original SPX calendar"):
        evaluate_locked_trades(compressed, prices, days)


def test_audit_and_controls_have_exact_common_origin_set(locked_inputs):
    result = evaluate_locked_trades(*locked_inputs)
    trades = result["trades"].loc[result["trades"].status == "evaluated"]
    summary = control_statistics(result["trades"], result["controls"])
    assert summary.n.eq(len(trades)).all()
    assert result["legs"].groupby("origin_date").size().eq(4).all()
    for row in trades.itertuples():
        assert len(json.loads(row.option_ids)) == 4
        assert len(json.loads(row.entry_vegas)) == 4
        assert len(json.loads(row.exit_bid)) == 4
        assert row.gross_vega == pytest.approx(1)
    with pytest.raises(ValueError, match="exactly the primary"):
        control_statistics(result["trades"], result["controls"].iloc[1:])


def test_cumulative_drawdown_and_pooled_labelling():
    dates = pd.bdate_range("2025-01-02", periods=3)
    trades = pd.DataFrame({"origin_date": dates, "actual_exit": dates,
                           "session_index": [502, 507, 512], "offset": [2, 2, 2],
                           "status": "evaluated"})
    for column in PNL_COLUMNS:
        trades[column] = [-2., 1., -3.]
    curve = cumulative_pnl(trades)
    np.testing.assert_allclose(curve.cumulative_pnl, [-2, -1, -4])
    np.testing.assert_allclose(curve.drawdown, [2, 1, 4])
    assert curve.scope.eq("pooled_overlapping_descriptive").all()
    assert cumulative_pnl(trades, offset=2).scope.eq("fixed_offset_nonoverlapping").all()
    stats = primary_statistics(trades)
    assert stats["final_cumulative_executable_total_pnl"] == -4
    assert stats["max_drawdown_executable_total_pnl"] == 4
    offsets = offset_statistics(trades)
    assert offsets.index.to_list() == [0, 1, 2, 3, 4]
    assert offsets.loc[2, "n"] == 3
    assert offsets.loc[0, "n"] == 0


def test_real_run_manifest_forbids_automatic_retry(tmp_path, monkeypatch):
    dates = pd.bdate_range("2023-01-03", "2025-08-29")
    prices = synthetic_prices(dates).loc[lambda rows: rows.date != "2025-01-01"]
    daily = pd.DataFrame({"quote_date": prices.date.to_numpy(),
                         "rr25_30_60": .02 + .005 * np.sin(np.arange(len(prices)) / 9)})
    reads = []
    monkeypatch.setattr("src.trading_locked.iter_quote_dates", lambda paths: reads.append(paths) or [])
    monkeypatch.setattr("src.trading_locked.evaluate_locked_trades", lambda *args: {"trades": pd.DataFrame()})
    raw = tmp_path / "spx_option_prices_2025.csv"  # No real file is opened.
    run_locked_2025(raw, daily, prices, "f" * 40, tmp_path)
    assert len(reads) == 1
    manifest = json.loads((tmp_path / "trading_locked_2025_run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["evaluator_commit"] == "f" * 40
    with pytest.raises(FileExistsError):
        run_locked_2025(raw, daily, prices, "f" * 40, tmp_path)
    assert len(reads) == 1


def test_unexpected_failure_is_logged_and_cannot_be_retried(tmp_path, monkeypatch):
    dates = pd.bdate_range("2023-01-03", "2025-08-29")
    prices = synthetic_prices(dates).loc[lambda rows: rows.date != "2025-01-01"]
    daily = pd.DataFrame({"quote_date": prices.date.to_numpy(),
                         "rr25_30_60": .02 + .005 * np.sin(np.arange(len(prices)) / 9)})
    monkeypatch.setattr("src.trading_locked.iter_quote_dates", lambda paths: [])
    def fail(*args):
        raise RuntimeError("synthetic technical failure")
    monkeypatch.setattr("src.trading_locked.evaluate_locked_trades", fail)
    raw = tmp_path / "spx_option_prices_2025.csv"
    with pytest.raises(RuntimeError, match="synthetic technical failure"):
        run_locked_2025(raw, daily, prices, "f" * 40, tmp_path)
    manifest = json.loads((tmp_path / "trading_locked_2025_run.json").read_text())
    assert manifest["status"] == "failed"
    with pytest.raises(FileExistsError):
        run_locked_2025(raw, daily, prices, "f" * 40, tmp_path)
