"""Fixed-contract execution and previous-close SPX hedge accounting."""

import numpy as np
import pandas as pd

from src.forecasting import DEV_START
from src.market_state import spx_trading_dates
from src.pricing import black_scholes_greeks
from src.trade_selection import CONTRACT_MULTIPLIER, executable_quote_mask


IDENTITY_COLUMNS = ["security_id", "option_id", "expiry_date", "strike", "option_type"]


def trade_schedule(origin_date, spx_prices):
    calendar = spx_trading_dates(spx_prices)
    calendar = calendar[calendar >= DEV_START]
    origin = pd.Timestamp(origin_date)
    if origin not in calendar:
        raise ValueError("Forecast origin must be an SPX session.")
    index = calendar.get_loc(origin)
    if index + 5 >= len(calendar):
        raise ValueError("Calendar does not cover the scheduled exit.")
    return {
        "origin_date": origin, "entry_date": calendar[index + 1],
        "scheduled_exit": calendar[index + 5],
        "fallback_date": calendar[index + 6] if index + 6 < len(calendar) else pd.NaT,
        "session_index": index, "offset": index % 5,
    }


def fixed_contract_quotes(legs, quotes, date):
    """Match identities, never substitute a different strike/expiry contract."""
    day = quotes.loc[pd.to_datetime(quotes.quote_date).eq(pd.Timestamp(date))].copy()
    if day.option_id.duplicated().any():
        raise ValueError("Daily option IDs must be unique.")
    if legs.option_id.duplicated().any():
        raise ValueError("Position option IDs must be unique.")
    identity = legs[IDENTITY_COLUMNS].copy()
    identity["expiry_date"] = pd.to_datetime(identity.expiry_date)
    day["expiry_date"] = pd.to_datetime(day.expiry_date)
    return identity.merge(day, on=IDENTITY_COLUMNS, how="left", validate="one_to_one")


def check_exit_coverage(legs, quotes, schedule):
    """Inspect quote availability only; never calculate a price change or P&L."""
    if legs.empty:
        raise ValueError("Exit coverage requires a nonempty fixed position.")
    scheduled = executable_quote_mask(fixed_contract_quotes(legs, quotes, schedule["scheduled_exit"]))
    fallback = pd.Series(False, index=scheduled.index)
    if not scheduled.all() and pd.notna(schedule["fallback_date"]):
        fallback = executable_quote_mask(fixed_contract_quotes(legs, quotes, schedule["fallback_date"]))
    if scheduled.all():
        status, date = "scheduled", schedule["scheduled_exit"]
    elif fallback.all():
        status, date = "fallback", schedule["fallback_date"]
    else:
        status, date = "unevaluable", pd.NaT
    return {"status": status, "exit_date": date,
            "scheduled_contracts": int(scheduled.sum()), "fallback_contracts": int(fallback.sum()),
            "n_contracts": len(legs)}


def account_trade(legs, marks, spx_prices, schedule):
    """Account one fixed basket. Real-data use is reserved for a later release."""
    result = {"status": "unevaluable", "reason": "", "option_pnl": np.nan,
              "midpoint_option_pnl": np.nan, "execution_drag": np.nan,
              "hedge_pnl": np.nan, "total_pnl": np.nan, "hedges": pd.DataFrame()}
    if len(legs) != 4 or not np.isfinite(legs.quantity).all() or (legs.quantity == 0).any():
        raise ValueError("Accounting requires four fixed nonzero option quantities.")
    if not pd.to_datetime(legs.quote_date).eq(schedule["entry_date"]).all():
        raise ValueError("Entry legs must match the scheduled entry date.")
    coverage = check_exit_coverage(legs, marks, schedule)
    if coverage["status"] == "unevaluable":
        return {**result, "reason": "missing_exit_quotes", "exit_coverage": coverage}
    exit_date = coverage["exit_date"]
    calendar = spx_trading_dates(spx_prices)
    dates = calendar[(calendar >= schedule["entry_date"]) & (calendar <= exit_date)]
    closes = spx_prices.assign(date=pd.to_datetime(spx_prices.date)).set_index("date").close
    spot = closes.reindex(dates)
    if not np.isfinite(spot).all() or spot.le(0).any():
        return {**result, "reason": "missing_spx_close", "exit_coverage": coverage}
    quantity = legs.quantity.to_numpy(dtype=float)
    hedge, previous_spot = 0., None
    history = []
    for date in dates:
        today = float(spot.loc[date])
        interval_pnl = 0. if previous_spot is None else hedge * (today - previous_spot)
        previous_hedge = hedge
        rows = fixed_contract_quotes(legs, marks, date)
        if date == exit_date:
            hedge = 0.
        else:
            try:
                bid, ask = rows.best_bid, rows.best_ask
                valid_marks = np.isfinite(bid) & np.isfinite(ask) & bid.ge(0) & ask.ge(bid)
                if not valid_marks.all() or (date == schedule["entry_date"] and not executable_quote_mask(rows).all()):
                    raise ValueError("Missing daily quotes.")
                greeks = black_scholes_greeks(
                    rows.forward, rows.strike, rows.discount_factor, rows.time_to_expiry,
                    rows.mid_iv, rows.option_type, today,
                )
                delta = CONTRACT_MULTIPLIER * greeks["spot_delta"]
                hedge = -float(np.dot(quantity, delta))
            except (ValueError, AttributeError):
                return {**result, "reason": "missing_daily_hedge_inputs", "exit_coverage": coverage}
        history.append({"quote_date": date, "previous_hedge": previous_hedge,
                        "hedge_position": hedge, "hedge_turnover": abs(hedge - previous_hedge),
                        "hedge_pnl": interval_pnl})
        previous_spot = today
    entry = fixed_contract_quotes(legs, marks, schedule["entry_date"])
    exit_rows = fixed_contract_quotes(legs, marks, exit_date)
    entry_price = np.where(quantity > 0, entry.best_ask, entry.best_bid)
    exit_price = np.where(quantity > 0, exit_rows.best_bid, exit_rows.best_ask)
    option_pnl = CONTRACT_MULTIPLIER * float(np.dot(quantity, exit_price - entry_price))
    entry_mid = (entry.best_bid + entry.best_ask) / 2
    exit_mid = (exit_rows.best_bid + exit_rows.best_ask) / 2
    midpoint_pnl = CONTRACT_MULTIPLIER * float(np.dot(quantity, exit_mid - entry_mid))
    history = pd.DataFrame(history)
    hedge_pnl = float(history.hedge_pnl.sum())
    return {**result, "status": "evaluated", "exit_coverage": coverage,
            "option_pnl": option_pnl, "midpoint_option_pnl": midpoint_pnl,
            "execution_drag": option_pnl - midpoint_pnl, "hedge_pnl": hedge_pnl,
            "total_pnl": option_pnl + hedge_pnl, "hedges": history,
            "hedge_turnover": float(history.hedge_turnover.sum())}
