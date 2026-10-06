import numpy as np
import pandas as pd
import pytest

from src.pricing import black_scholes_call_price, black_scholes_put_price, black_scholes_greeks
from src.trade_attribution import COMPONENTS, attribute_saved_trades, direction_summary


def example(vol_multiplier=1., spot_multiplier=1., roll=False):
    entry_date, exit_date = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-07")
    types = np.array(["put", "call", "put", "call"])
    strike = np.array([90., 110., 90., 110.])
    v0 = np.array([.25, .20, .27, .22])
    tau = np.array([30., 30., 60., 60.]) / 365
    q = np.array([1., -1., -1., 1.]) / (400 * black_scholes_greeks(101, strike, .99, tau, v0, types, 100)["vega"])
    legs = pd.DataFrame({"origin_date": pd.Timestamp("2024-02-29"), "quote_date": entry_date,
                         "option_id": range(4), "option_type": types, "strike": strike,
                         "expiry_date": entry_date + pd.to_timedelta(tau * 365, unit="D"),
                         "target_days": [30, 30, 60, 60], "forward": 101., "discount_factor": .99,
                         "time_to_expiry": tau, "mid_iv": v0, "quantity": q})
    end = legs.drop(columns=["origin_date", "quantity", "target_days"]).copy()
    end["quote_date"] = exit_date
    end["forward"] *= spot_multiplier
    end["mid_iv"] *= vol_multiplier
    if roll:
        end["time_to_expiry"] -= 6 / 365
        end["discount_factor"] += .001

    def mids(rows):
        return np.where(types == "put",
            black_scholes_put_price(rows.forward, strike, rows.discount_factor, rows.time_to_expiry, rows.mid_iv),
            black_scholes_call_price(rows.forward, strike, rows.discount_factor, rows.time_to_expiry, rows.mid_iv))

    m0, m1 = mids(legs), mids(end)
    end["best_bid"], end["best_ask"] = m1 - .01, m1 + .01
    legs["exit_bid"], legs["exit_ask"] = end.best_bid, end.best_ask
    mid_pnl = 100 * np.dot(q, m1 - m0)
    drag = -2 * 100 * .01 * abs(q).sum()
    trades = pd.DataFrame([{"origin_date": pd.Timestamp("2024-02-29"), "entry_date": entry_date,
                            "actual_exit": exit_date, "status": "evaluated", "offset": 2,
                            "midpoint_option_pnl": mid_pnl, "hedge_pnl": .003,
                            "midpoint_total_pnl": mid_pnl + .003,
                            "execution_drag": drag, "executable_total_pnl": mid_pnl + .003 + drag}])
    prices = pd.DataFrame({"date": [entry_date, exit_date], "close": [100, 100 * spot_multiplier]})
    return trades, legs, end, prices


@pytest.mark.parametrize("factor", ["iv_level", "held_wing_shape", "spot_repricing", "time_discount"])
def test_single_factor_isolates_correct_component(factor):
    inputs = example(
        vol_multiplier=1.2 if factor == "iv_level" else np.array([1.1, 1/1.1, 1.1, 1/1.1]) if factor == "held_wing_shape" else 1.,
        spot_multiplier=1.1 if factor == "spot_repricing" else 1.,
        roll=factor == "time_discount",
    )
    result = attribute_saved_trades(*inputs).iloc[0]
    assert result[factor] == pytest.approx(inputs[0].midpoint_option_pnl.iloc[0], abs=1e-12)
    for column in COMPONENTS[:5]:
        if column != factor:
            assert result[column] == pytest.approx(0, abs=1e-12)
    assert result.reconstructed_total == pytest.approx(inputs[0].executable_total_pnl.iloc[0])


def test_joint_changes_reconcile_without_mutating_or_reordering_contracts():
    inputs = example(vol_multiplier=np.array([1.1, .8, 1.3, 1.2]), spot_multiplier=.9, roll=True)
    trades, legs, end, prices = inputs
    snapshots = [frame.copy(deep=True) for frame in inputs]
    result = attribute_saved_trades(trades, legs.sample(frac=1, random_state=3), end.sample(frac=1, random_state=5), prices)
    assert result.reconstructed_total.iloc[0] == pytest.approx(trades.executable_total_pnl.iloc[0], abs=1e-12)
    assert result.pricing_residual.abs().max() < 1e-12
    for actual, original in zip(inputs, snapshots):
        pd.testing.assert_frame_equal(actual, original)


def test_missing_or_changed_contracts_raise_instead_of_dropping_trades():
    trades, legs, end, prices = example()
    with pytest.raises(ValueError, match="identities"):
        attribute_saved_trades(trades, legs, end.iloc[:-1], prices)
    end.loc[0, "strike"] += 1
    with pytest.raises(ValueError, match="identities"):
        attribute_saved_trades(trades, legs, end, prices)


def test_inconsistent_saved_pnl_cannot_be_hidden_in_residual():
    trades, legs, end, prices = example()
    trades["midpoint_option_pnl"] += .01
    with pytest.raises(ValueError, match="reproduce saved"):
        attribute_saved_trades(trades, legs, end, prices)


def test_direction_counts_keep_rejection_denominators_separate():
    trades = pd.DataFrame({"forecast": [.1, -.1, .2, np.nan],
                           "finite_nonzero_signal": [True, True, True, False],
                           "entry_reached": [True, True, False, False],
                           "status": ["evaluated", "skipped", "skipped", "skipped"]})
    counts = direction_summary(trades)
    assert counts.loc["finite signals", ["n", "positive", "negative"]].tolist() == [3, 2, 1]
    assert counts.loc["reachable entries", ["n", "positive", "negative"]].tolist() == [2, 1, 1]
    assert counts.loc["evaluated trades", "positive_fraction"] == 1
