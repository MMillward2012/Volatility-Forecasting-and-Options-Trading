"""Frozen listed-option translation evaluator; no target outcomes or tuning."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting import DEV_END, DEV_START, HOLDOUT_END
from src.historical_pipeline import iter_quote_dates
from src.market_state import spx_trading_dates
from src.trade_backtest import (
    IDENTITY_COLUMNS, account_trade, check_exit_coverage, fixed_contract_quotes, trade_schedule,
)
from src.trade_selection import (
    CHAIN_COLUMNS, fit_rr_spread_signal, forecast_rr_spread_signal,
    prepare_executable_chain, select_rr_spread,
)


VARIANTS = ("primary", "reversed_sign", "constant_positive")
PNL_COLUMNS = (
    "option_pnl", "midpoint_option_pnl", "execution_drag", "hedge_pnl",
    "executable_total_pnl", "midpoint_total_pnl",
)
DRAG_TOLERANCE = 1e-10


def build_locked_rr_signals(daily_surface, spx_prices):
    states = daily_surface[["quote_date", "rr25_30_60"]].copy()
    dates = pd.to_datetime(states.quote_date)
    states = states.loc[dates.between(DEV_START, HOLDOUT_END)].copy()
    development = states.loc[pd.to_datetime(states.quote_date) <= DEV_END]
    fitted = fit_rr_spread_signal(development, spx_prices)
    return fitted, forecast_rr_spread_signal(fitted, states, spx_prices)


def control_positions(legs, forecast):
    """Controls never perform a second contract-selection exercise."""
    if not np.isfinite(forecast) or forecast == 0:
        raise ValueError("Controls require a finite nonzero primary forecast.")
    positions = {}
    for name, multiplier in zip(VARIANTS, (1., -1., np.sign(forecast))):
        position = legs.copy()
        position["quantity"] *= multiplier
        position["gross_vega"] = (position.quantity * position.vega).abs()
        position["net_vega"] = position.quantity * position.vega
        position["position_delta"] = position.quantity * position.option_delta
        positions[name] = position
    return positions


def _daily_marks(prepared, option_ids):
    keys = IDENTITY_COLUMNS + ["quote_date"]
    quotes = prepared["quotes"].loc[prepared["quotes"].option_id.isin(option_ids)].copy()
    greeks = prepared["chain"][keys + ["forward", "discount_factor", "mid_iv"]]
    marks = quotes.merge(greeks, on=keys, how="left", validate="one_to_one")
    for column in ("best_bid", "best_ask", "time_to_expiry", "forward", "discount_factor", "mid_iv"):
        marks[column] = pd.to_numeric(marks[column], errors="coerce").astype(float)
    return marks


def _empty_day():
    return {
        "chain": pd.DataFrame(columns=CHAIN_COLUMNS),
        "quotes": pd.DataFrame(columns=IDENTITY_COLUMNS + [
            "quote_date", "best_bid", "best_ask", "time_to_expiry",
        ]),
    }


def _record_entry(record, legs):
    record["expiry_30"] = legs.expiry_date.iloc[0]
    record["expiry_60"] = legs.expiry_date.iloc[2]
    fields = {
        "option_ids": "option_id", "strikes": "strike", "entry_dtes": "days_to_expiry",
        "entry_deltas": "forward_delta", "entry_spot_deltas": "option_delta",
        "entry_vegas": "vega", "quantities": "quantity", "entry_bid": "best_bid",
        "entry_ask": "best_ask", "entry_iv": "mid_iv",
    }
    record.update({name: json.dumps(legs[column].to_list()) for name, column in fields.items()})
    record["gross_vega"] = float(legs.gross_vega.sum())
    record["net_vega"] = float(legs.net_vega.sum())
    record["entry_portfolio_delta"] = float(legs.position_delta.sum())


def _finish_trade(item, spx_prices, control_records, leg_records, hedge_records):
    record, legs, schedule = item["record"], item["legs"], item["schedule"]
    marks = pd.concat(item["marks"], ignore_index=True)
    positions = control_positions(legs, record["forecast"])
    results = {name: account_trade(position, marks, spx_prices, schedule)
               for name, position in positions.items()}
    primary = results["primary"]
    if any(result["status"] != primary["status"] for result in results.values()):
        raise RuntimeError("Controls and primary must share evaluation availability.")
    coverage = primary["exit_coverage"]
    record.update(status=primary["status"], reason=primary["reason"],
                  actual_exit=coverage["exit_date"], exit_status=coverage["status"])
    for name, result in results.items():
        pnl = {column: result[column] for column in PNL_COLUMNS[:4]}
        pnl["executable_total_pnl"] = result["total_pnl"]
        pnl["midpoint_total_pnl"] = result["midpoint_option_pnl"] + result["hedge_pnl"]
        pnl["hedge_turnover"] = result.get("hedge_turnover", np.nan)
        if result["status"] == "evaluated":
            if not np.isfinite(list(pnl.values())).all():
                raise RuntimeError("Evaluated P&L must be finite.")
            if pnl["execution_drag"] > DRAG_TOLERANCE:
                raise RuntimeError("Execution drag cannot be positive with valid quotes.")
        if name == "primary":
            record.update(pnl)
        else:
            control_records.append({"origin_date": record["origin_date"], "variant": name,
                                    "status": result["status"], "reason": result["reason"], **pnl})
        if not result["hedges"].empty:
            hedge_records.append(result["hedges"].assign(origin_date=record["origin_date"], variant=name))
    exit_rows = fixed_contract_quotes(legs, marks, coverage["exit_date"])
    record["exit_bid"] = json.dumps(exit_rows.best_bid.to_list())
    record["exit_ask"] = json.dumps(exit_rows.best_ask.to_list())
    audit = legs.copy()
    audit["origin_date"] = record["origin_date"]
    audit["session_index"] = record["session_index"]
    audit["offset"] = record["offset"]
    audit["actual_exit"] = coverage["exit_date"]
    audit["status"] = primary["status"]
    audit["exit_bid"] = exit_rows.best_bid.to_numpy()
    audit["exit_ask"] = exit_rows.best_ask.to_numpy()
    audit["reversed_quantity"] = positions["reversed_sign"].quantity.to_numpy()
    audit["constant_positive_quantity"] = positions["constant_positive"].quantity.to_numpy()
    leg_records.append(audit)


def evaluate_locked_trades(signals, spx_prices, quote_dates, progress=False):
    """One chronological pass; quote_dates is a stream, including in synthetic tests."""
    prices = spx_prices.loc[pd.to_datetime(spx_prices.date) <= HOLDOUT_END].copy()
    calendar = spx_trading_dates(prices)
    calendar = calendar[calendar >= DEV_START]
    confirmation = calendar[calendar > DEV_END]
    signals = signals[["quote_date", "session_index", "forecast"]].copy()
    signals["quote_date"] = pd.to_datetime(signals.quote_date)
    if not pd.DatetimeIndex(signals.quote_date).equals(confirmation):
        raise ValueError("Signals must retain every confirmation SPX session.")
    indices = calendar.get_indexer(confirmation)
    if not np.array_equal(signals.session_index, indices):
        raise ValueError("Signal indices must preserve the original SPX calendar.")
    closes = prices.assign(date=pd.to_datetime(prices.date)).set_index("date").close
    records, entries = [], {}
    for row in signals.itertuples(index=False):
        finite = bool(np.isfinite(row.forecast) and row.forecast != 0)
        full_horizon = row.session_index + 5 < len(calendar)
        record = {"origin_date": row.quote_date, "session_index": int(row.session_index),
                  "offset": int(row.session_index % 5), "forecast": row.forecast,
                  "sign": np.sign(row.forecast) if finite else np.nan,
                  "finite_nonzero_signal": finite, "full_exit_horizon": full_horizon,
                  "entry_reached": False, "entry_chain_available": False,
                  "expiry_pair_available": False, "four_wings_available": False,
                  "status": "skipped", "reason": "", "exit_status": "", "actual_exit": pd.NaT,
                  **{column: np.nan for column in PNL_COLUMNS}}
        records.append(record)
        if not full_horizon:
            record["reason"] = "no_full_exit_horizon"
        elif not finite:
            record["reason"] = "no_signal"
        else:
            schedule = trade_schedule(row.quote_date, prices)
            record.update({key: schedule[key] for key in ("entry_date", "scheduled_exit", "fallback_date")})
            entries[schedule["entry_date"]] = (record, schedule)
    stream = iter(quote_dates)
    current = next(stream, None)
    active, controls, legs_audit, hedges_audit, daily_audit = [], [], [], [], []
    for day_number, date in enumerate(confirmation, start=1):
        if current is not None and pd.Timestamp(current[1]) < date:
            raise ValueError("Option stream contains a date outside the locked calendar.")
        raw = None
        if current is not None and pd.Timestamp(current[1]) == date:
            raw = current[2]
            if not pd.to_datetime(raw.date).eq(date).all():
                raise ValueError("Daily option rows must match their locked session.")
        prepared = _empty_day()
        if raw is not None and (active or date in entries):
            held_ids = [option_id for item in active for option_id in item["legs"].option_id]
            prepared = prepare_executable_chain(raw, float(closes.loc[date]), held_ids)
        daily_audit.append({"quote_date": date, "raw_rows": len(raw) if raw is not None else 0,
                            **prepared.get("diagnostics", {})})
        for item in active:
            item["marks"].append(_daily_marks(prepared, item["legs"].option_id))
        finished = []
        for item in active:
            schedule = item["schedule"]
            if date == schedule["scheduled_exit"]:
                coverage = check_exit_coverage(item["legs"], pd.concat(item["marks"], ignore_index=True), schedule)
                if coverage["status"] == "scheduled" or pd.isna(schedule["fallback_date"]):
                    finished.append(item)
            elif date == schedule["fallback_date"]:
                finished.append(item)
        for item in finished:
            _finish_trade(item, prices, controls, legs_audit, hedges_audit)
        finished_ids = {id(item) for item in finished}
        active = [item for item in active if id(item) not in finished_ids]
        if date in entries:
            record, schedule = entries[date]
            record["entry_reached"] = True
            record["entry_chain_available"] = raw is not None
            selection = select_rr_spread(prepared["chain"], record["forecast"])
            record["expiry_pair_available"] = "expiry_pair" in selection
            record["four_wings_available"] = selection["status"] == "available"
            record["reason"] = selection["reason"] if raw is not None else "missing_entry_chain"
            if selection["status"] == "available":
                legs = selection["legs"]
                _record_entry(record, legs)
                record["status"] = "pending"
                active.append({"record": record, "schedule": schedule, "legs": legs,
                               "marks": [_daily_marks(prepared, legs.option_id)]})
        if current is not None and pd.Timestamp(current[1]) == date:
            current = next(stream, None)
        if progress and day_number % 20 == 0:
            print(f"{date.date()}: {day_number}/{len(confirmation)} locked sessions processed", flush=True)
    if current is not None or active:
        raise RuntimeError("Unconsumed option dates or unresolved trades outside locked window.")
    return {"trades": pd.DataFrame(records), "controls": pd.DataFrame(controls, columns=[
                "origin_date", "variant", "status", "reason", *PNL_COLUMNS, "hedge_turnover"]),
            "legs": pd.concat(legs_audit, ignore_index=True) if legs_audit else pd.DataFrame(),
            "hedges": pd.concat(hedges_audit, ignore_index=True) if hedges_audit else pd.DataFrame(),
            "daily_diagnostics": pd.DataFrame(daily_audit)}


def trading_coverage(trades):
    counts = {"confirmation_sessions": len(trades)}
    for column in ("full_exit_horizon", "finite_nonzero_signal", "entry_reached",
                   "entry_chain_available", "expiry_pair_available", "four_wings_available"):
        counts[column] = int(trades[column].sum())
    for status in ("evaluated", "unevaluable", "skipped"):
        counts[status] = int(trades.status.eq(status).sum())
    for status in ("scheduled", "fallback"):
        counts[f"{status}_evaluated_exits"] = int((trades.status.eq("evaluated") & trades.exit_status.eq(status)).sum())
    return pd.Series(counts, name="count")


def cumulative_pnl(trades, column="executable_total_pnl", offset=None):
    evaluated = trades.loc[trades.status.eq("evaluated")].copy()
    if offset is not None:
        if offset not in range(5):
            raise ValueError("Offset must be one of the five frozen offsets.")
        evaluated = evaluated.loc[evaluated.offset.eq(offset)]
    evaluated = evaluated.sort_values(["actual_exit", "origin_date"])
    values = evaluated[column].to_numpy(dtype=float) if len(evaluated) else np.array([])
    cumulative = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0., cumulative])[1:]
    result = evaluated[["actual_exit", "origin_date", "session_index", "offset"]].copy()
    result["cumulative_pnl"] = cumulative
    result["drawdown"] = peaks - cumulative
    result["scope"] = "pooled_overlapping_descriptive" if offset is None else "fixed_offset_nonoverlapping"
    return result.reset_index(drop=True)


def primary_statistics(trades):
    evaluated = trades.loc[trades.status.eq("evaluated")]
    row = {"n": len(evaluated)}
    for column in PNL_COLUMNS:
        values = evaluated[column] if len(evaluated) else pd.Series(dtype=float)
        row.update({f"total_{column}": float(values.sum()), f"mean_{column}": float(values.mean()),
                    f"median_{column}": float(values.median()), f"std_{column}": float(values.std(ddof=1)),
                    f"positive_fraction_{column}": float(values.gt(0).mean()) if len(values) else np.nan})
    for column in ("executable_total_pnl", "midpoint_total_pnl"):
        curve = cumulative_pnl(trades, column)
        row[f"final_cumulative_{column}"] = float(curve.cumulative_pnl.iloc[-1]) if len(curve) else 0.
        row[f"max_drawdown_{column}"] = float(curve.drawdown.max()) if len(curve) else 0.
    row["scope"] = "pooled_overlapping_descriptive"
    return pd.Series(row)


def offset_statistics(trades):
    columns = ["n", "total_executable_total_pnl", "mean_executable_total_pnl",
               "median_executable_total_pnl", "positive_fraction_executable_total_pnl",
               "total_midpoint_total_pnl", "total_execution_drag", "total_hedge_pnl",
               "max_drawdown_executable_total_pnl"]
    return pd.DataFrame([
        {"offset": offset, **primary_statistics(trades.loc[trades.offset.eq(offset)])[columns].to_dict()}
        for offset in range(5)
    ]).set_index("offset")


def control_statistics(trades, controls):
    primary = trades.loc[trades.status.eq("evaluated")]
    columns = ["executable_total_pnl", "midpoint_total_pnl", "execution_drag", "hedge_pnl"]
    rows = []
    for variant in VARIANTS:
        data = primary if variant == "primary" else controls.loc[
            controls.variant.eq(variant) & controls.status.eq("evaluated")
        ]
        if data.origin_date.duplicated().any() or set(data.origin_date) != set(primary.origin_date):
            raise ValueError("Control statistics must use exactly the primary evaluated origins.")
        rows.append({"variant": variant, "n": len(data),
                     **{f"total_{column}": float(data[column].sum()) for column in columns},
                     "mean_executable_total_pnl": float(data.executable_total_pnl.mean()),
                     "positive_fraction": float(data.executable_total_pnl.gt(0).mean()) if len(data) else np.nan})
    return pd.DataFrame(rows).set_index("variant")


def run_locked_2025(raw_file, daily_surface, spx_prices, evaluator_commit, output_directory, progress=False):
    """Exclusive real-data run; an existing manifest prohibits automatic retry."""
    raw_file = Path(raw_file)
    if raw_file.name != "spx_option_prices_2025.csv":
        raise ValueError("Locked evaluation accepts only the 2025 annual option file.")
    if len(evaluator_commit) != 40 or any(char not in "0123456789abcdef" for char in evaluator_commit):
        raise ValueError("Record the full Phase-A evaluator commit hash.")
    prices = spx_prices.loc[pd.to_datetime(spx_prices.date) <= HOLDOUT_END].copy()
    calendar = spx_trading_dates(prices)
    calendar = calendar[calendar >= DEV_START]
    if calendar.empty or calendar[0] != DEV_START or calendar[-1] != HOLDOUT_END:
        raise ValueError("SPX calendar must cover the frozen start and end.")
    if calendar[calendar > DEV_END][0] != pd.Timestamp("2025-01-02"):
        raise ValueError("Confirmation must start on the frozen 2025-01-02 session.")
    fitted, signals = build_locked_rr_signals(daily_surface, prices)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "trading_locked_2025_run.json"
    manifest = {"evaluator_commit": evaluator_commit, "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "signal_fit": {key: str(value) if isinstance(value, pd.Timestamp) else value
                               for key, value in fitted.items()}}
    with manifest_path.open("x") as file:
        json.dump(manifest, file, indent=2)
    try:
        result = evaluate_locked_trades(signals, prices, iter_quote_dates([raw_file]), progress)
        for name, table in result.items():
            table.to_csv(output_directory / f"trading_locked_2025_{name}.csv", index=False)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status="failed", failure=f"{type(error).__name__}: {error}")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {**result, "signal_fit": fitted, "evaluator_commit": evaluator_commit}
