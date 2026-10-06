"""Entry-date selection of listed, vega-normalised SPX downside RR spreads."""

import numpy as np
import pandas as pd

from src.data_cleaning import clean_option_data
from src.forecast_robustness import past_only_zscore
from src.forecasting import DEV_END, DEV_START, HOLDOUT_END
from src.forward_inference import infer_expiry_forwards
from src.historical_pipeline import CONSTANTS, RAW_COLUMNS
from src.iv_panel import build_iv_panel
from src.market_state import spx_trading_dates
from src.option_matching import match_calls_and_puts
from src.pricing import black_scholes_greeks


CONTRACT_MULTIPLIER = 100
TENOR_TOLERANCE = 7
DELTA_TOLERANCE = 0.05
CHAIN_COLUMNS = [
    "security_id", "quote_date", "expiry_date", "option_id", "symbol", "option_type",
    "strike", "days_to_expiry", "time_to_expiry", "best_bid", "best_ask", "mid_price",
    "forward", "discount_factor", "mid_iv", "is_otm", "forward_delta", "option_delta", "vega",
]


def executable_quote_mask(rows):
    bid = pd.to_numeric(rows["best_bid"], errors="coerce")
    ask = pd.to_numeric(rows["best_ask"], errors="coerce")
    return np.isfinite(bid) & np.isfinite(ask) & bid.gt(0) & ask.ge(bid)


def prepare_executable_chain(raw_options, spot, held_option_ids=()):
    """Use today's PM quotes for target-expiry candidates and any held contracts."""
    missing = set(RAW_COLUMNS) - set(raw_options)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    dates = pd.to_datetime(raw_options["date"], errors="raise")
    if raw_options.empty or dates.isna().any() or dates.nunique() != 1:
        raise ValueError("Entry preparation requires exactly one nonempty quote date.")
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("SPX spot must be finite and positive.")
    for column, expected in CONSTANTS.items():
        if not raw_options[column].eq(expected).fillna(False).all():
            raise ValueError(f"{column} must equal {expected!r}.")
    if not raw_options["am_settlement"].isin([0, 1]).all():
        raise ValueError("am_settlement must contain only 0 or 1.")
    pm = raw_options.loc[raw_options["am_settlement"].eq(0)].copy()
    if pm["optionid"].isna().any() or pm["optionid"].duplicated().any():
        raise ValueError("PM option IDs must be present and unique per date.")
    bid = pd.to_numeric(pm["best_bid"], errors="coerce")
    ask = pd.to_numeric(pm["best_offer"], errors="coerce")
    good_for_parity = np.isfinite(bid) & np.isfinite(ask) & bid.ge(0) & ask.ge(bid)
    diagnostics = {
        "raw_rows": len(raw_options), "pm_rows": len(pm),
        "zero_bid_rows": int(bid.eq(0).sum()),
        "invalid_quote_rows": int((~good_for_parity).sum()),
        "forward_failures": [], "iv_failures": 0,
    }
    cleaned = clean_option_data(pm.loc[good_for_parity])
    candidate = cleaned["days_to_expiry"].between(23, 37) | cleaned["days_to_expiry"].between(53, 67)
    needed_expiries = cleaned.loc[candidate | cleaned.option_id.isin(held_option_ids), "expiry_date"]
    slices = []
    for expiry, rows in cleaned.loc[cleaned.expiry_date.isin(needed_expiries)].groupby("expiry_date"):
        if (rows.days_to_expiry <= 0).any():
            continue
        try:
            forward = infer_expiry_forwards(match_calls_and_puts(rows))
            if forward.empty:
                raise ValueError("No matched forward estimate.")
            f = float(forward["forward"].iloc[0])
            otm = (rows.option_type.eq("put") & rows.strike.lt(f)) | (
                rows.option_type.eq("call") & rows.strike.gt(f)
            )
            targets = rows.days_to_expiry.between(23, 37) | rows.days_to_expiry.between(53, 67)
            wanted = (targets & otm) | rows.option_id.isin(held_option_ids)
            # A held zero-bid option can still provide a midpoint hedge mark.
            quotes = rows.loc[wanted & (executable_quote_mask(rows) | rows.option_id.isin(held_option_ids))]
            if quotes.empty:
                continue
            panel = build_iv_panel(quotes, forward)
        except ValueError as error:
            diagnostics["forward_failures"].append({"expiry_date": str(expiry.date()), "reason": str(error)})
            continue
        valid = np.isfinite(panel.mid_iv) & panel.mid_iv.gt(0)
        diagnostics["iv_failures"] += int((~valid).sum())
        panel = panel.loc[valid].copy()
        if panel.empty:
            continue
        greeks = black_scholes_greeks(
            panel.forward, panel.strike, panel.discount_factor, panel.time_to_expiry,
            panel.mid_iv, panel.option_type, spot,
        )
        panel["forward_delta"] = greeks["forward_delta"]
        panel["option_delta"] = CONTRACT_MULTIPLIER * greeks["spot_delta"]
        panel["vega"] = CONTRACT_MULTIPLIER * greeks["vega"]
        panel = panel.loc[np.isfinite(panel.vega) & panel.vega.gt(0)]
        slices.append(panel[CHAIN_COLUMNS])
    chain = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame(columns=CHAIN_COLUMNS)
    return {"chain": chain, "quotes": cleaned, "diagnostics": diagnostics}


def _entry_candidates(chain):
    if chain.empty:
        return chain.copy()
    if pd.to_datetime(chain.quote_date).nunique() != 1 or chain.option_id.duplicated().any():
        raise ValueError("Selection must use one quote date with unique option IDs.")
    finite = np.isfinite(chain[["mid_iv", "forward_delta", "vega", "option_delta"]]).all(axis=1)
    return chain.loc[executable_quote_mask(chain) & finite & chain.mid_iv.gt(0)
                     & chain.vega.gt(0) & chain.is_otm.eq(True)].copy()


def select_expiry_pair(chain):
    """Minimise DTE errors before applying the 25-delta distance rule."""
    candidates = _entry_candidates(chain)
    expiries = []
    for expiry, rows in candidates.groupby("expiry_date"):
        if set(rows.option_type) == {"call", "put"}:
            if rows.days_to_expiry.nunique() != 1:
                raise ValueError("Expiry DTE must be consistent.")
            expiries.append((expiry, int(rows.days_to_expiry.iloc[0])))
    pairs = [
        (abs(near_dte - 30) + abs(far_dte - 60), near, far)
        for near, near_dte in expiries for far, far_dte in expiries
        if near_dte < far_dte and abs(near_dte - 30) <= TENOR_TOLERANCE
        and abs(far_dte - 60) <= TENOR_TOLERANCE
    ]
    return min(pairs)[1:] if pairs else None


def select_delta_contract(chain, expiry, option_type):
    candidates = _entry_candidates(chain)
    candidates = candidates.loc[candidates.expiry_date.eq(expiry) & candidates.option_type.eq(option_type)].copy()
    if candidates.empty:
        return None
    target = -.25 if option_type == "put" else .25
    candidates["delta_distance"] = (candidates.forward_delta - target).abs()
    closest = candidates.sort_values(["delta_distance", "strike", "option_id"]).iloc[0]
    return closest


def select_rr_spread(chain, forecast):
    """Build fixed four-option quantities using only the sign of today's forecast."""
    result = {"status": "unavailable", "reason": "", "legs": pd.DataFrame(), "wing_checks": []}
    if not np.isfinite(forecast) or forecast == 0:
        result["reason"] = "no_signal"
        return result
    pair = select_expiry_pair(chain)
    if pair is None:
        result["reason"] = "no_expiry_pair"
        return result
    result["expiry_pair"] = pair
    selected = []
    for tenor, expiry in zip((30, 60), pair):
        for option_type in ("put", "call"):
            row = select_delta_contract(chain, expiry, option_type)
            distance = float(row.delta_distance) if row is not None else np.nan
            eligible = row is not None and distance <= DELTA_TOLERANCE + 1e-12
            result["wing_checks"].append({"target_days": tenor, "option_type": option_type,
                                          "delta_distance": distance, "eligible": eligible})
            if eligible:
                selected.append({**row.to_dict(), "target_days": tenor})
    if len(selected) != 4:
        result["reason"] = "missing_25_delta_wing"
        return result
    legs = pd.DataFrame(selected)
    sign = np.sign(forecast)
    legs["quantity"] = sign * np.array([1., -1., -1., 1.]) / (4 * legs.vega)
    legs["gross_vega"] = (legs.quantity * legs.vega).abs()
    legs["net_vega"] = legs.quantity * legs.vega
    legs["position_delta"] = legs.quantity * legs.option_delta
    return {**result, "status": "available", "legs": legs,
            "gross_vega": float(legs.gross_vega.sum()), "net_vega": float(legs.net_vega.sum()),
            "portfolio_delta": float(legs.position_delta.sum()),
            "initial_hedge": -float(legs.position_delta.sum())}


def _aligned_rr_state(daily, spx_prices):
    dates = pd.to_datetime(daily["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Signal dates must be present, sorted and unique.")
    calendar = spx_trading_dates(spx_prices)
    expected = calendar[(calendar >= DEV_START) & (calendar <= dates.max())]
    if not pd.DatetimeIndex(dates).equals(expected):
        raise ValueError("Signal rows must retain every SPX session from development start.")
    values = pd.to_numeric(daily["rr25_30_60"], errors="coerce")
    return values.where(np.isfinite(values))


def fit_rr_spread_signal(daily, spx_prices):
    """Freeze the existing RR25 M2 using development-matured endpoint changes."""
    _aligned_rr_state(daily, spx_prices)
    development = daily.loc[pd.to_datetime(daily.quote_date) <= DEV_END]
    if development.empty or pd.Timestamp(development.quote_date.iloc[-1]) != DEV_END:
        raise ValueError("Final signal fit must include development through 2024-12-31.")
    values = pd.to_numeric(development.rr25_30_60, errors="coerce")
    values = values.where(np.isfinite(values))
    z = past_only_zscore(values)
    y = values.shift(-5) - values
    valid = np.isfinite(z) & np.isfinite(y)
    if valid.sum() < 2 or np.ptp(z[valid]) == 0:
        raise ValueError("Insufficient development history for RR25 M2.")
    intercept, slope = np.linalg.lstsq(np.column_stack((np.ones(valid.sum()), z[valid])), y[valid], rcond=None)[0]
    return {"intercept": float(intercept), "slope": float(slope), "n_train": int(valid.sum()),
            "fit_end": development.quote_date.iloc[-1],
            "last_matured_origin": development.quote_date.iloc[-6]}


def forecast_rr_spread_signal(fitted, daily, spx_prices):
    """Generate locked-period signals only, never in-sample development trades."""
    values = _aligned_rr_state(daily, spx_prices)
    z = past_only_zscore(values)
    result = pd.DataFrame({"quote_date": daily.quote_date,
                           "session_index": np.arange(len(daily)),
                           "forecast": fitted["intercept"] + fitted["slope"] * z})
    dates = pd.to_datetime(result.quote_date)
    return result.loc[(dates > DEV_END) & (dates <= HOLDOUT_END)].reset_index(drop=True)
