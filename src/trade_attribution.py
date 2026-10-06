"""Post-result Black repricing attribution of saved trades; never selects trades."""

import numpy as np
import pandas as pd

from src.data_cleaning import clean_option_data
from src.forward_inference import infer_expiry_forwards
from src.historical_pipeline import iter_quote_dates
from src.iv_panel import build_iv_panel
from src.option_matching import match_calls_and_puts
from src.pricing import black_scholes_call_price, black_scholes_put_price
from src.trade_selection import CONTRACT_MULTIPLIER


COMPONENTS = (
    "time_discount", "spot_repricing", "forward_basis", "iv_level",
    "held_wing_shape", "hedge_pnl", "execution_drag", "pricing_residual",
)


def direction_summary(trades):
    """Count directions on explicitly different denominators, without refitting."""
    rows = []
    for name, mask in (
        ("finite signals", trades.finite_nonzero_signal),
        ("reachable entries", trades.entry_reached),
        ("evaluated trades", trades.status.eq("evaluated")),
    ):
        values = trades.loc[mask, "forecast"]
        if not np.isfinite(values).all() or values.eq(0).any():
            raise ValueError("Direction counts require finite nonzero forecasts.")
        rows.append({"sample": name, "n": len(values), "positive": int(values.gt(0).sum()),
                     "negative": int(values.lt(0).sum()),
                     "positive_fraction": float(values.gt(0).mean())})
    return pd.DataFrame(rows).set_index("sample")


def load_saved_exit_marks(raw_file, trades, legs):
    """Recover only the existing contracts' exit inputs from the licensed file."""
    evaluated = trades.loc[trades.status.eq("evaluated")]
    wanted = legs[["origin_date", "option_id", "expiry_date"]].merge(
        evaluated[["origin_date", "actual_exit"]], on="origin_date", validate="many_to_one"
    )
    wanted["actual_exit"] = pd.to_datetime(wanted.actual_exit)
    wanted["expiry_date"] = pd.to_datetime(wanted.expiry_date)
    by_date = {date: rows for date, rows in wanted.groupby("actual_exit")}
    marks = []
    for _, date, raw in iter_quote_dates([raw_file]):
        if date not in by_date:
            continue
        request = by_date[date]
        bid = pd.to_numeric(raw.best_bid, errors="coerce")
        ask = pd.to_numeric(raw.best_offer, errors="coerce")
        eligible = raw.am_settlement.eq(0) & np.isfinite(bid) & np.isfinite(ask) & bid.ge(0) & ask.ge(bid)
        cleaned = clean_option_data(raw.loc[eligible])
        cleaned = cleaned.loc[cleaned.expiry_date.isin(request.expiry_date)]
        forwards = infer_expiry_forwards(match_calls_and_puts(cleaned))
        selected = cleaned.loc[cleaned.option_id.isin(request.option_id)]
        panel = build_iv_panel(selected, forwards)
        marks.append(panel[["quote_date", "option_id", "expiry_date", "strike", "option_type",
                            "best_bid", "best_ask", "forward", "discount_factor", "time_to_expiry", "mid_iv"]])
    if not marks:
        raise ValueError("No saved-contract exit marks were found.")
    result = pd.concat(marks, ignore_index=True)
    expected = wanted[["actual_exit", "option_id"]].drop_duplicates().rename(columns={"actual_exit": "quote_date"})
    checked = expected.merge(result, on=["quote_date", "option_id"], how="left", validate="one_to_one")
    if not np.isfinite(checked.mid_iv).all() or checked.mid_iv.le(0).any():
        raise ValueError("Attribution needs a positive finite exit IV for every saved contract.")
    return checked


def attribute_saved_trades(trades, legs, exit_marks, spx_prices):
    """Reconcile saved P&L through a declared endpoint repricing path.

    Order: time/discount, spot, forward basis, then the volatility block.
    Log-IV level and wing shape share their interaction equally by averaging
    the two orders inside that block. This is not a causal Greek attribution.
    """
    evaluated = trades.loc[trades.status.eq("evaluated")].copy()
    if evaluated.origin_date.duplicated().any():
        raise ValueError("Evaluated origins must be unique.")
    prices = spx_prices.copy()
    prices["date"] = pd.to_datetime(prices.date)
    if prices.date.duplicated().any():
        raise ValueError("SPX dates must be unique.")
    spot = prices.set_index("date").close
    exits = exit_marks.copy()
    exits["quote_date"] = pd.to_datetime(exits.quote_date)
    if exits.duplicated(["quote_date", "option_id"]).any():
        raise ValueError("Exit marks must be unique by date and option ID.")
    outputs = []
    for trade in evaluated.itertuples():
        entry = legs.loc[pd.to_datetime(legs.origin_date).eq(pd.Timestamp(trade.origin_date))].copy()
        entry = entry.sort_values(["target_days", "option_type"], ascending=[True, False])
        if len(entry) != 4 or entry.option_id.duplicated().any():
            raise ValueError("Attribution requires the four original legs.")
        if list(zip(entry.target_days, entry.option_type)) != [(30, "put"), (30, "call"), (60, "put"), (60, "call")]:
            raise ValueError("Each tenor must have one put and one call.")
        end = entry[["option_id"]].merge(
            exits.loc[exits.quote_date.eq(pd.Timestamp(trade.actual_exit))],
            on="option_id", how="left", validate="one_to_one",
        )
        for column in ("strike", "option_type"):
            if not np.array_equal(entry[column].to_numpy(), end[column].to_numpy()):
                raise ValueError("Exit marks must retain the saved contract identities.")
        if not np.array_equal(pd.to_datetime(entry.expiry_date), pd.to_datetime(end.expiry_date)):
            raise ValueError("Exit expiries must match the original contracts.")
        np.testing.assert_allclose(end[["best_bid", "best_ask"]], entry[["exit_bid", "exit_ask"]], rtol=0, atol=1e-10)
        q = entry.quantity.to_numpy()
        s0, s1 = float(spot.loc[pd.Timestamp(trade.entry_date)]), float(spot.loc[pd.Timestamp(trade.actual_exit)])
        if not np.isfinite([s0, s1]).all() or min(s0, s1) <= 0:
            raise ValueError("Both endpoint SPX closes must be positive and finite.")
        f0, f1 = entry.forward.to_numpy(), end.forward.to_numpy()
        d0, d1 = entry.discount_factor.to_numpy(), end.discount_factor.to_numpy()
        t0, t1 = entry.time_to_expiry.to_numpy(), end.time_to_expiry.to_numpy()
        v0, v1 = entry.mid_iv.to_numpy(), end.mid_iv.to_numpy()
        strike = entry.strike.to_numpy()
        put = entry.option_type.eq("put").to_numpy()

        def value(forward, discount, tau, volatility):
            prices = np.where(put,
                black_scholes_put_price(forward, strike, discount, tau, volatility),
                black_scholes_call_price(forward, strike, discount, tau, volatility))
            return float(CONTRACT_MULTIPLIER * np.dot(q, prices))

        initial = value(f0, d0, t0, v0)
        rolled = value(f0, d1, t1, v0)
        spot_changed = value(f0 * s1 / s0, d1, t1, v0)
        basis_changed = value(f1, d1, t1, v0)
        final = value(f1, d1, t1, v1)
        # Geometric mean and put/call IV ratio remain positive in counterfactuals.
        log_change = np.log(v1).reshape(2, 2) - np.log(v0).reshape(2, 2)
        level_change = np.repeat(log_change.mean(axis=1), 2)
        shape_change = log_change.ravel() - level_change
        level_only = value(f1, d1, t1, v0 * np.exp(level_change))
        shape_only = value(f1, d1, t1, v0 * np.exp(shape_change))
        level = .5 * ((level_only - basis_changed) + (final - shape_only))
        shape = .5 * ((shape_only - basis_changed) + (final - level_only))
        row = {"origin_date": trade.origin_date, "offset": trade.offset,
               "time_discount": rolled - initial,
               "spot_repricing": spot_changed - rolled,
               "forward_basis": basis_changed - spot_changed,
               "iv_level": level, "held_wing_shape": shape,
               "hedge_pnl": trade.hedge_pnl, "execution_drag": trade.execution_drag,
               "pricing_residual": trade.midpoint_option_pnl - (final - initial),
               "midpoint_total_pnl": trade.midpoint_total_pnl,
               "executable_total_pnl": trade.executable_total_pnl}
        if not np.isfinite([row[column] for column in COMPONENTS]).all():
            raise ValueError("Attribution components must be finite.")
        row["reconstructed_total"] = sum(row[column] for column in COMPONENTS)
        if abs(row["pricing_residual"]) > 1e-8:
            raise ValueError("Model endpoint prices do not reproduce saved midpoint P&L.")
        if not np.isclose(row["reconstructed_total"], trade.executable_total_pnl, rtol=0, atol=1e-10):
            raise ValueError("Attribution does not reconcile to the saved result.")
        outputs.append(row)
    return pd.DataFrame(outputs)
