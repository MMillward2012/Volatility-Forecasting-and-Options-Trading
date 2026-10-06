"""Post-result factor and ideal daily-reset RR diagnostics; no model fitting."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting import DEV_START, DEV_END, HOLDOUT_END
from src.historical_pipeline import (
    MAX_BRACKET_GAP_DAYS, iter_quote_dates, process_quote_date,
)
from src.market_state import spx_trading_dates
from src.pricing import (
    black_scholes_call_price, black_scholes_put_price, black_scholes_greeks,
)
from src.skew_metrics import _find_delta_strike
from src.trade_selection import CONTRACT_MULTIPLIER, forecast_rr_spread_signal
from src.vol_surface import evaluate_surface


# Existing development-fitted RR25 M2, retained at its previously audited precision.
# These coefficients are model inputs, not estimates from this experiment.
FROZEN_RR_SIGNAL = {
    "intercept": 0.0006537436032636668,
    "slope": -0.0017253059632106394,
}
TARGET_DAYS = (30, 60)
PNL_COLUMNS = ("model_option_pnl", "hedge_pnl", "delta_hedged_pnl")


def frozen_signals(daily, spx_prices):
    """Use the existing causal feature algorithm with saved coefficients only."""
    return forecast_rr_spread_signal(FROZEN_RR_SIGNAL, daily, spx_prices)


def _calendar(spx_prices):
    dates = spx_trading_dates(spx_prices)
    return dates[(dates >= DEV_START) & (dates <= HOLDOUT_END)]


def direct_factor_outcomes(signals, daily, spx_prices):
    """Preserve all origins; factor outcomes need valid delayed-entry/exit states."""
    calendar = _calendar(spx_prices)
    if signals.quote_date.duplicated().any() or daily.quote_date.duplicated().any():
        raise ValueError("Signal and state dates must be unique.")
    states = daily[["quote_date", "rr25_30_60"]].copy()
    states["quote_date"] = pd.to_datetime(states.quote_date, errors="raise")
    if not states.quote_date.isin(calendar).all():
        raise ValueError("States must lie on the original SPX session calendar.")
    values = pd.to_numeric(states.rr25_30_60, errors="coerce")
    states["rr25_30_60"] = values.where(np.isfinite(values))
    state = states.set_index("quote_date").rr25_30_60.reindex(calendar)
    rows = []
    for signal in signals[["quote_date", "session_index", "forecast"]].itertuples(index=False):
        origin = pd.Timestamp(signal.quote_date)
        index = calendar.get_indexer([origin])[0]
        if index < 0 or index != signal.session_index:
            raise ValueError("Original SPX session indices must be preserved.")
        row = {"origin_date": origin, "session_index": index, "offset": index % 5,
               "forecast": signal.forecast, "entry_date": pd.NaT, "exit_date": pd.NaT,
               "factor_pnl": np.nan, "full_horizon_factor_pnl": np.nan,
               "missed_first_session": np.nan, "factor_available": False,
               "finite_nonzero_signal": bool(np.isfinite(signal.forecast) and signal.forecast != 0),
               "full_exit_horizon": index + 5 < len(calendar), "factor_reason": ""}
        if not row["full_exit_horizon"]:
            row["factor_reason"] = "no_full_exit_horizon"
        elif not row["finite_nonzero_signal"]:
            row["factor_reason"] = "no_signal"
        else:
            row.update(entry_date=calendar[index + 1], exit_date=calendar[index + 5])
            sign = np.sign(signal.forecast)
            initial, entry, exit_state = state.iloc[[index, index + 1, index + 5]]
            if np.isfinite([initial, exit_state]).all():
                row["full_horizon_factor_pnl"] = sign * (exit_state - initial)
            if np.isfinite([initial, entry]).all():
                row["missed_first_session"] = sign * (entry - initial)
            if np.isfinite([entry, exit_state]).all():
                row["factor_pnl"] = sign * (exit_state - entry)
                row["factor_available"] = True
            else:
                row["factor_reason"] = "missing_qc_rr_endpoint"
        rows.append(row)
    return pd.DataFrame(rows)


def surface_inputs(snapshot, tau):
    """Same accepted maturity bracket: linear forward and log-linear discount."""
    grid = snapshot["grid"]
    if grid is None:
        raise ValueError("missing_production_surface")
    maturities = np.asarray(grid["time_to_expiry"], dtype=float)
    if not np.isfinite(tau) or tau <= 0 or not maturities[0] <= tau <= maturities[-1]:
        raise ValueError("outside_maturity_range")
    right = np.searchsorted(maturities, tau)
    left = right if maturities[right] == tau else right - 1
    if (maturities[right] - maturities[left]) * 365 > MAX_BRACKET_GAP_DAYS + 1e-10:
        raise ValueError("excessive_maturity_gap")
    lower = float(np.max(grid["sampled_support_min"][left:right + 1]))
    upper = float(np.min(grid["sampled_support_max"][left:right + 1]))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("insufficient_shared_support")
    forwards = snapshot["forwards"].sort_values("time_to_expiry")
    selected = []
    for maturity in maturities[left:right + 1]:
        rows = forwards.loc[np.isclose(forwards.time_to_expiry, maturity, rtol=0, atol=1e-12)]
        if len(rows) != 1:
            raise ValueError("missing_unique_forward_bracket")
        selected.append(rows.iloc[0])
    f = np.array([row.forward for row in selected], dtype=float)
    d = np.array([row.discount_factor for row in selected], dtype=float)
    if not np.isfinite(np.r_[f, d]).all() or (f <= 0).any() or (d <= 0).any():
        raise ValueError("invalid_forward_discount")
    weight = 0.0 if left == right else (tau - maturities[left]) / (maturities[right] - maturities[left])
    forward = float((1 - weight) * f[0] + weight * f[-1])
    discount = float(np.exp((1 - weight) * np.log(d[0]) + weight * np.log(d[-1])))
    atm = evaluate_surface(0., tau, grid)["implied_volatility"]
    raw_grid = dict(grid, repaired_total_variance=grid.get("raw_total_variance", grid["repaired_total_variance"]))
    raw_atm = evaluate_surface(0., tau, raw_grid)["implied_volatility"]
    if not np.isfinite([atm, raw_atm]).all() or abs(atm - raw_atm) > .05:
        raise ValueError("excessive_calendar_repair")
    return {"forward": forward, "discount_factor": discount, "lower_k": lower, "upper_k": upper}


def solve_25_delta(snapshot, days, option_type):
    """Reuse the unique supported OTM forward-delta root used by RR25."""
    tau = days / 365
    inputs = surface_inputs(snapshot, tau)
    lower, upper = inputs["lower_k"], inputs["upper_k"]
    lower, upper = (lower, min(upper, 0.)) if option_type == "put" else (max(lower, 0.), upper)
    if lower >= upper:
        raise ValueError("25_delta_points_unsupported")
    k = _find_delta_strike(snapshot["grid"], tau, .25, option_type, lower, upper)
    return k, inputs


def _price_and_greeks(forward, strike, discount, tau, iv, option_type, spot):
    price_function = black_scholes_put_price if option_type == "put" else black_scholes_call_price
    price = CONTRACT_MULTIPLIER * float(price_function(forward, strike, discount, tau, iv))
    greeks = black_scholes_greeks(forward, strike, discount, tau, iv, option_type, spot)
    return {"model_price": price, "forward_delta": float(greeks["forward_delta"]),
            "option_delta": CONTRACT_MULTIPLIER * float(greeks["spot_delta"]),
            "vega": CONTRACT_MULTIPLIER * float(greeks["vega"])}


def build_synthetic_rr(snapshot, spot, forecast):
    """Open exact 30D/60D wings; each has absolute vega .25 at this close."""
    if not np.isfinite(forecast) or forecast == 0:
        raise ValueError("no_signal")
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("invalid_spx_close")
    metrics = snapshot["metrics"].set_index("target_days")
    for days in TARGET_DAYS:
        if days not in metrics.index or not bool(metrics.loc[days, "rr25_valid"]):
            raise ValueError(f"invalid_{days}d_rr25")
    rows = []
    date = pd.Timestamp(snapshot["quote_date"])
    signs = np.sign(forecast) * np.array([1., -1., -1., 1.])
    for sign, (days, option_type) in zip(signs, [(d, kind) for d in TARGET_DAYS for kind in ("put", "call")]):
        k, inputs = solve_25_delta(snapshot, days, option_type)
        tau = days / 365
        strike = inputs["forward"] * np.exp(k)
        iv = evaluate_surface(k, tau, snapshot["grid"])["implied_volatility"]
        greeks = _price_and_greeks(inputs["forward"], strike, inputs["discount_factor"], tau, iv, option_type, spot)
        if not np.isfinite(greeks["vega"]) or greeks["vega"] <= 0:
            raise ValueError("invalid_vega")
        quantity = sign / (4 * greeks["vega"])
        rows.append({"quote_date": date, "target_days": days, "option_type": option_type,
                     "expiry_date": date + pd.Timedelta(days=days), "time_to_expiry": tau,
                     "strike": strike, "log_moneyness": k, "mid_iv": iv,
                     "forward": inputs["forward"], "discount_factor": inputs["discount_factor"],
                     **greeks, "quantity": quantity, "gross_vega": abs(quantity * greeks["vega"]),
                     "net_vega": quantity * greeks["vega"],
                     "position_delta": quantity * greeks["option_delta"]})
    return pd.DataFrame(rows)


def reprice_held_basket(legs, snapshot, spot):
    """Hold yesterday's strikes/quantities; age by elapsed ACT/365 calendar days."""
    today = pd.Timestamp(snapshot["quote_date"])
    prices, deltas, remaining, aged_prices, unchanged_iv_prices, iv_changes = [], [], [], [], [], []
    for leg in legs.itertuples(index=False):
        tau = (pd.Timestamp(leg.expiry_date) - today).days / 365
        inputs = surface_inputs(snapshot, tau)
        k = np.log(leg.strike / inputs["forward"])
        iv = evaluate_surface(k, tau, snapshot["grid"])["implied_volatility"]
        greeks = _price_and_greeks(inputs["forward"], leg.strike, inputs["discount_factor"], tau, iv, leg.option_type, spot)
        price_function = black_scholes_put_price if leg.option_type == "put" else black_scholes_call_price
        aged_prices.append(CONTRACT_MULTIPLIER * float(price_function(
            leg.forward, leg.strike, leg.discount_factor, tau, leg.mid_iv)))
        unchanged_iv_prices.append(CONTRACT_MULTIPLIER * float(price_function(
            inputs["forward"], leg.strike, inputs["discount_factor"], tau, leg.mid_iv)))
        iv_changes.append(iv - leg.mid_iv)
        prices.append(greeks["model_price"])
        deltas.append(greeks["option_delta"])
        remaining.append(tau)
    old_value = float(np.dot(legs.quantity, legs.model_price))
    aged_value = float(np.dot(legs.quantity, aged_prices))
    unchanged_iv_value = float(np.dot(legs.quantity, unchanged_iv_prices))
    current_value = float(np.dot(legs.quantity, prices))
    return {"value": current_value,
            "portfolio_delta": float(np.dot(legs.quantity, deltas)),
            "remaining_maturities": np.asarray(remaining),
            "age_repricing": aged_value - old_value,
            "forward_discount_repricing": unchanged_iv_value - aged_value,
            "held_iv_repricing": current_value - unchanged_iv_value,
            "linear_held_iv_pnl": float(np.dot(legs.quantity * legs.vega, iv_changes))}


def roll_step(legs, previous_spot, previous_hedge, cash, snapshot, spot, forecast, close=False):
    """Close the aged basket, reset at mid, and reconcile the self-financing ledger."""
    old_value = float(np.dot(legs.quantity, legs.model_price))
    old_equity = cash + old_value + previous_hedge * previous_spot
    held = reprice_held_basket(legs, snapshot, spot)
    option_pnl = held["value"] - old_value
    hedge_pnl = previous_hedge * (spot - previous_spot)
    new_legs = pd.DataFrame() if close else build_synthetic_rr(snapshot, spot, forecast)
    new_value = 0. if close else float(np.dot(new_legs.quantity, new_legs.model_price))
    new_hedge = 0. if close else -float(new_legs.position_delta.sum())
    # Sell old options, buy replacements, then exchange the hedge difference at today's spot.
    new_cash = cash + held["value"] - new_value - (new_hedge - previous_hedge) * spot
    equity = new_cash + new_value + new_hedge * spot
    residual = equity - old_equity - option_pnl - hedge_pnl
    if not np.isclose(residual, 0., atol=1e-10, rtol=0):
        raise RuntimeError("Self-financing roll identity failed.")
    return {"legs": new_legs, "cash": new_cash, "hedge": new_hedge,
            "spot": spot, "model_option_pnl": option_pnl, "hedge_pnl": hedge_pnl,
            "delta_hedged_pnl": option_pnl + hedge_pnl, "cash_equity": equity,
            "accounting_residual": residual, "aged_maturities": held["remaining_maturities"],
            **{key: held[key] for key in ("age_repricing", "forward_discount_repricing",
                                         "held_iv_repricing", "linear_held_iv_pnl")},
            "hedge_turnover": abs(new_hedge - previous_hedge)}


def start_position(snapshot, spot, forecast):
    legs = build_synthetic_rr(snapshot, spot, forecast)
    hedge = -float(legs.position_delta.sum())
    value = float(np.dot(legs.quantity, legs.model_price))
    return {"legs": legs, "hedge": hedge, "spot": spot, "cash": -value - hedge * spot}


def evaluate_surface_trades(signals, spx_prices, snapshots, progress=False):
    """Chronological surface stream; never condition entry on future availability."""
    calendar = _calendar(spx_prices)
    prices = spx_prices.assign(date=pd.to_datetime(spx_prices.date)).set_index("date").close
    records, entries = [], {}
    for signal in signals[["quote_date", "session_index", "forecast"]].itertuples(index=False):
        date = pd.Timestamp(signal.quote_date)
        index = calendar.get_indexer([date])[0]
        if index < 0 or index != signal.session_index:
            raise ValueError("Original SPX session indices must be preserved.")
        row = {"origin_date": date, "session_index": index, "offset": index % 5,
               "forecast": signal.forecast, "entry_date": pd.NaT, "exit_date": pd.NaT,
               "entry_available": False, "status": "skipped", "reason": "",
               **{column: np.nan for column in PNL_COLUMNS}, "cash_equity": np.nan,
               "max_accounting_residual": np.nan}
        records.append(row)
        if index + 5 >= len(calendar):
            row["reason"] = "no_full_exit_horizon"
        elif not np.isfinite(signal.forecast) or signal.forecast == 0:
            row["reason"] = "no_signal"
        else:
            row.update(entry_date=calendar[index + 1], exit_date=calendar[index + 5])
            if row["entry_date"] in entries:
                raise ValueError("Forecast origins must be unique.")
            entries[row["entry_date"]] = row
    intervals, basket_audit, daily_audit, active = [], [], [], []
    seen = set()
    previous_date = None
    for date, snapshot in snapshots:
        date = pd.Timestamp(date)
        if date not in calendar or (previous_date is not None and date <= previous_date):
            raise ValueError("Surface stream must follow distinct SPX sessions.")
        if previous_date is not None and calendar.get_loc(date) != calendar.get_loc(previous_date) + 1:
            raise ValueError("Surface stream cannot bridge missing sessions; emit an unavailable snapshot.")
        previous_date = date
        seen.add(date)
        spot = float(prices.loc[date])
        if snapshot is not None and pd.Timestamp(snapshot["quote_date"]) != date:
            raise ValueError("Surface date mismatch.")
        daily_audit.append({"quote_date": date, "surface_available": snapshot is not None and snapshot["grid"] is not None})
        still_active = []
        for item in active:
            row = item["record"]
            try:
                if snapshot is None or snapshot["grid"] is None:
                    raise ValueError("missing_production_surface")
                if not np.isfinite(spot) or spot <= 0:
                    raise ValueError("invalid_spx_close")
                final = date == row["exit_date"]
                step = roll_step(item["legs"], item["spot"], item["hedge"], item["cash"],
                                 snapshot, spot, row["forecast"], close=final)
                for column in PNL_COLUMNS:
                    item["totals"][column] += step[column]
                item["max_residual"] = max(item["max_residual"], abs(step["accounting_residual"]))
                intervals.append({"origin_date": row["origin_date"], "quote_date": date,
                                  "previous_hedge": item["hedge"], "next_hedge": step["hedge"],
                                  "aged_30d": step["aged_maturities"][0] * 365,
                                  "aged_60d": step["aged_maturities"][2] * 365,
                                  **{key: step[key] for key in (*PNL_COLUMNS, "cash_equity", "accounting_residual", "hedge_turnover",
                                                               "age_repricing", "forward_discount_repricing",
                                                               "held_iv_repricing", "linear_held_iv_pnl")}})
                if final:
                    if not np.isclose(step["cash_equity"], item["totals"]["delta_hedged_pnl"], atol=1e-10, rtol=0):
                        raise RuntimeError("Final cash and accumulated P&L disagree.")
                    row.update(status="evaluated", reason="", **item["totals"],
                               cash_equity=step["cash_equity"], max_accounting_residual=item["max_residual"])
                else:
                    item.update({key: step[key] for key in ("legs", "cash", "hedge", "spot")})
                    basket_audit.append(step["legs"].assign(origin_date=row["origin_date"]))
                    still_active.append(item)
            except ValueError as error:
                row.update(status="unevaluable", reason=f"{date.date()}: {error}")
        active = still_active
        if date in entries:
            row = entries[date]
            try:
                if snapshot is None or snapshot["grid"] is None:
                    raise ValueError("missing_production_surface")
                position = start_position(snapshot, spot, row["forecast"])
                row.update(entry_available=True, status="pending")
                active.append({**position, "record": row,
                               "totals": dict.fromkeys(PNL_COLUMNS, 0.), "max_residual": 0.})
                basket_audit.append(position["legs"].assign(origin_date=row["origin_date"]))
            except ValueError as error:
                row["reason"] = f"entry: {error}"
        if progress and len(seen) % 10 == 0:
            print(f"{date.date()}: {len(seen)} surface sessions processed", flush=True)
    if active or any(date not in seen for date in entries):
        raise ValueError("Surface stream does not cover every required entry/exit session.")
    return {"synthetic_trades": pd.DataFrame(records), "intervals": pd.DataFrame(intervals),
            "baskets": pd.concat(basket_audit, ignore_index=True) if basket_audit else pd.DataFrame(),
            "surface_daily": pd.DataFrame(daily_audit)}


def summarize_pnl(rows, columns, offset=None):
    """Per-column valid coverage, in each column's units; pooled trades overlap."""
    if offset is not None:
        rows = rows.loc[rows.offset.eq(offset)]
    records = []
    for column in columns:
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        records.append({"measure": column, "n": len(values), "total": values.sum() if len(values) else np.nan,
                        "mean": values.mean(), "median": values.median(),
                        "positive_fraction": values.gt(0).mean() if len(values) else np.nan})
    return pd.DataFrame(records)


def offset_summary(rows, columns):
    return pd.concat([summarize_pnl(rows, columns, offset).assign(offset=offset)
                      for offset in range(5)], ignore_index=True)


def cumulative_results(rows, column):
    """Realise at original exit date; pooled all-origin series is descriptive."""
    valid = rows.loc[np.isfinite(rows[column])].sort_values(["exit_date", "origin_date"])
    return pd.DataFrame({"exit_date": valid.exit_date, "origin_date": valid.origin_date,
                         "cumulative_pnl": valid[column].cumsum()}).reset_index(drop=True)


def production_snapshots(raw_files, spx_prices, session_dates, diagnostics=None):
    """Reconstruct once per required session using the unchanged production fit."""
    session_dates = pd.DatetimeIndex(session_dates)
    stream = iter(iter_quote_dates(raw_files))
    current = next(stream, None)
    for date in session_dates:
        while current is not None and current[1] < date:
            current = next(stream, None)
        snapshot = None
        if current is not None and current[1] == date:
            metrics, info, snapshot = process_quote_date(current[2], date, retain_surface=True)
            if diagnostics is not None:
                diagnostics.append(info)
            current = next(stream, None)
        yield date, snapshot
    # No later dates are fitted or used; annual files may contain dates beyond the sample.


def development_sanity(raw_files, daily, spx_prices):
    """Coverage and first-valid-quarter mechanics only; return no realised P&L."""
    paths = tuple(Path(path) for path in raw_files)
    if {p.name for p in paths} != {"spx_option_prices_2023.csv", "spx_option_prices_2024.csv"} or len(paths) != 2:
        raise ValueError("Development sanity accepts only the 2023 and 2024 option files.")
    if pd.to_datetime(daily.quote_date).gt(DEV_END).any() or pd.to_datetime(spx_prices.date).gt(DEV_END).any():
        raise ValueError("Supply development-only states and SPX prices.")
    calendar = _calendar(spx_prices)
    aligned = daily.assign(quote_date=pd.to_datetime(daily.quote_date)).set_index("quote_date").reindex(calendar)
    valid = np.isfinite(aligned[["rr25_30", "rr25_60"]]).all(axis=1)
    candidates = calendar[valid]
    chosen = pd.Series(candidates, index=candidates.to_period("Q")).groupby(level=0).first().tolist()
    chosen = [d for d in chosen if calendar.get_loc(d) + 1 < len(calendar)]
    requested = sorted({date for d in chosen for date in (d, calendar[calendar.get_loc(d) + 1])})
    diagnostics = []
    snapshots = dict(production_snapshots(paths, spx_prices, requested, diagnostics))
    spot = spx_prices.assign(date=pd.to_datetime(spx_prices.date)).set_index("date").close
    rows = []
    for date in chosen:
        next_date = calendar[calendar.get_loc(date) + 1]
        row = {"quote_date": date, "next_session": next_date, "basket_available": False,
               "accounting_available": False, "reason": "", "max_rr_state_difference": np.nan}
        try:
            current, following = snapshots[date], snapshots[next_date]
            if current is None or current["grid"] is None:
                raise ValueError("missing_production_surface")
            position = start_position(current, float(spot.loc[date]), 1.)
            legs = position["legs"]
            recalculated = current["metrics"].set_index("target_days")
            expected = aligned.loc[date, ["rr25_30", "rr25_60"]].to_numpy(dtype=float)
            actual = recalculated.loc[[30, 60], "rr25_downside"].to_numpy(dtype=float)
            row["max_rr_state_difference"] = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, atol=1e-7, rtol=1e-5):
                raise RuntimeError("Production reconstruction differs from saved development RR25 states.")
            row.update(basket_available=True, gross_vega=float(legs.gross_vega.sum()),
                       net_vega=float(legs.net_vega.sum()),
                       max_delta_error=float(np.max(np.abs(legs.forward_delta.to_numpy() - [-.25, .25, -.25, .25]))),
                       min_vega=float(legs.vega.min()), max_abs_quantity=float(legs.quantity.abs().max()))
            if following is None or following["grid"] is None:
                raise ValueError("missing_next_session_surface")
            step = roll_step(legs, position["spot"], position["hedge"], position["cash"],
                             following, float(spot.loc[next_date]), 1., close=True)
            row.update(accounting_available=True, accounting_residual=step["accounting_residual"],
                       liquidation_residual=step["cash_equity"] - step["delta_hedged_pnl"],
                       min_aged_days=float(step["aged_maturities"].min() * 365))
        except ValueError as error:
            row["reason"] = str(error)
        rows.append(row)
    return {"coverage": pd.Series({"development_sessions": len(calendar),
                                    "valid_30d_rr_states": int(np.isfinite(aligned.rr25_30).sum()),
                                    "valid_60d_rr_states": int(np.isfinite(aligned.rr25_60).sum()),
                                    "both_rr_states": int(valid.sum()),
                                    "missing_state_fraction": float((~valid).mean())}, name="count_or_fraction"),
            "mechanics": pd.DataFrame(rows), "diagnostics": pd.DataFrame(diagnostics)}


def run_surface_factor_2025(raw_file, daily, spx_prices, evaluator_commit, output_directory, progress=False):
    """Exclusive post-result run; neither model selection nor coefficient fitting."""
    raw_file = Path(raw_file)
    if raw_file.name != "spx_option_prices_2025.csv":
        raise ValueError("Only the frozen 2025 option file is permitted.")
    if len(evaluator_commit) != 40 or any(c not in "0123456789abcdef" for c in evaluator_commit):
        raise ValueError("Record the full Phase-A commit hash.")
    calendar = _calendar(spx_prices)
    if calendar[0] != DEV_START or calendar[-1] != HOLDOUT_END:
        raise ValueError("SPX dates must cover the frozen sample.")
    signals = frozen_signals(daily, spx_prices)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "surface_factor_run.json"
    manifest = {"evaluator_commit": evaluator_commit, "status": "running",
                "scientific_status": "blinded post-result re-analysis",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "signal_coefficients": FROZEN_RR_SIGNAL}
    with manifest_path.open("x") as file:
        json.dump(manifest, file, indent=2)
    try:
        factors = direct_factor_outcomes(signals, daily, spx_prices)
        factors.to_csv(output_directory / "direct_factors.csv", index=False)
        diagnostics = []
        result = evaluate_surface_trades(
            signals, spx_prices,
            production_snapshots([raw_file], spx_prices, calendar[calendar > DEV_END], diagnostics), progress,
        )
        for name, frame in result.items():
            frame.to_csv(output_directory / f"{name}.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(output_directory / "production_diagnostics.csv", index=False)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status="failed", failure=f"{type(error).__name__}: {error}")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"direct_factors": factors, **result}
