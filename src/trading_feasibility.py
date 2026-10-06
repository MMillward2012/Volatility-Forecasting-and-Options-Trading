"""Development-only coverage checks, without real-data P&L calculation."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting import DEV_END, DEV_START
from src.historical_pipeline import iter_quote_dates
from src.market_state import spx_trading_dates
from src.trade_backtest import check_exit_coverage, fixed_contract_quotes, trade_schedule
from src.trade_selection import CHAIN_COLUMNS, executable_quote_mask, prepare_executable_chain, select_rr_spread


def run_development_feasibility(raw_files, spx_prices, progress=False):
    """Stream only the two development files; retain selected rows, not full chains."""
    paths = tuple(Path(path) for path in raw_files)
    expected = {"spx_option_prices_2023.csv", "spx_option_prices_2024.csv"}
    if {path.name for path in paths} != expected or len(paths) != 2:
        raise ValueError("Feasibility requires exactly the 2023 and 2024 option files.")
    if pd.to_datetime(spx_prices.date).gt(DEV_END).any():
        raise ValueError("Supply development-only SPX prices for feasibility.")
    calendar = spx_trading_dates(spx_prices)
    calendar = calendar[(calendar >= DEV_START) & (calendar <= DEV_END)]
    closes = spx_prices.assign(date=pd.to_datetime(spx_prices.date)).set_index("date").close
    stream = iter(iter_quote_dates(paths))
    current = next(stream, None)
    daily, positions, wings, lifecycles, pending = [], [], [], [], []
    for index, date in enumerate(calendar):
        held = [option_id for item in pending for option_id in item["legs"].option_id]
        if current is not None and pd.Timestamp(current[1]) < date:
            raise ValueError("Raw date is outside the authoritative development calendar.")
        if current is not None and pd.Timestamp(current[1]) == date:
            _, _, raw = current
            if pd.to_datetime(raw.date).gt(DEV_END).any():
                raise ValueError("Non-development option rows are forbidden.")
            prepared = prepare_executable_chain(raw, float(closes.loc[date]), held)
            current = next(stream, None)
            chain, quotes, diagnostics = prepared["chain"], prepared["quotes"], prepared["diagnostics"]
        else:
            chain = pd.DataFrame(columns=CHAIN_COLUMNS)
            quotes = pd.DataFrame(columns=["security_id", "quote_date", "expiry_date", "option_id",
                                          "strike", "option_type", "best_bid", "best_ask"])
            diagnostics = {"raw_rows": 0, "pm_rows": 0, "zero_bid_rows": 0,
                           "invalid_quote_rows": 0, "forward_failures": [], "iv_failures": 0}
        for item in pending:
            rows = fixed_contract_quotes(item["legs"], quotes, date)
            item["quotes"].append(rows)
            quote_good = executable_quote_mask(rows)
            item["contract_checks"].append(quote_good.to_numpy())
            if date < item["schedule"]["scheduled_exit"] or (
                date == item["schedule"]["scheduled_exit"] and not quote_good.all()
            ):
                greek_ids = set(chain.option_id) if not chain.empty else set()
                item["hedge_checks"].append(set(item["legs"].option_id).issubset(greek_ids))
        finished = [item for item in pending if item["schedule"]["fallback_date"] == date]
        for item in finished:
            coverage = check_exit_coverage(item["legs"], pd.concat(item["quotes"], ignore_index=True), item["schedule"])
            checks = np.vstack(item["contract_checks"])
            # First four subsequent dates cover entry+1 through scheduled exit.
            through_scheduled = checks[:4].all(axis=0)
            lifecycles.append({"origin_date": item["schedule"]["origin_date"],
                               "entry_date": item["schedule"]["entry_date"],
                               "session_index": item["schedule"]["session_index"],
                               "offset": item["schedule"]["offset"], **coverage,
                               "contracts_valid_through_scheduled": int(through_scheduled.sum()),
                               "complete_quote_path": bool(through_scheduled.all()),
                               "complete_hedge_inputs": bool(all(item["hedge_checks"]))})
        finished_ids = {id(item) for item in finished}
        pending = [item for item in pending if id(item) not in finished_ids]
        selection = select_rr_spread(chain, 1.)
        pair = selection.get("expiry_pair")
        row = {"quote_date": date, **{key: diagnostics[key] for key in (
            "raw_rows", "pm_rows", "zero_bid_rows", "invalid_quote_rows", "iv_failures")},
            "failed_forward_expiries": len(diagnostics["forward_failures"]),
            "expiry_pair_available": pair is not None, "four_wings_available": selection["status"] == "available",
            "reason": selection["reason"], "lifecycle_boundary": False}
        if pair is not None:
            row["dte_error_30"] = abs((pair[0] - date).days - 30)
            row["dte_error_60"] = abs((pair[1] - date).days - 60)
        wings.extend({"quote_date": date, **check} for check in selection["wing_checks"])
        if selection["status"] == "available":
            legs = selection["legs"]
            positions.append(legs)
            row.update({key: selection[key] for key in ("gross_vega", "net_vega", "portfolio_delta", "initial_hedge")})
            if index >= 1 and index + 5 < len(calendar):
                schedule = trade_schedule(calendar[index - 1], spx_prices)
                pending.append({"legs": legs, "schedule": schedule, "quotes": [],
                                "contract_checks": [], "hedge_checks": []})
            else:
                row["lifecycle_boundary"] = True
        daily.append(row)
        if progress and (index + 1) % 25 == 0:
            print(f"{date.date()}: {index + 1}/{len(calendar)} development sessions checked", flush=True)
    if current is not None or pending:
        raise ValueError("Raw stream or lifecycle extends outside development.")
    return {"daily": pd.DataFrame(daily), "legs": pd.concat(positions, ignore_index=True) if positions else pd.DataFrame(),
            "wings": pd.DataFrame(wings), "lifecycle": pd.DataFrame(lifecycles)}
