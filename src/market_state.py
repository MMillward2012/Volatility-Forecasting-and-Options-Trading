"""Build SPX return, realized-volatility, and VIX state variables."""

import numpy as np
import pandas as pd


SPX_SECURITY_ID = 108105
TRADING_DAYS_PER_YEAR = 252


def _prepare_prices(prices, name, expected_secid=None):
    required = {"date", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"{name} prices are missing columns: {sorted(missing)}")
    for identifier in ("secid", "ticker"):
        if identifier in prices and prices[identifier].nunique(dropna=False) > 1:
            raise ValueError(f"{name} extract must contain only one {identifier}.")
    if "secid" in prices and expected_secid is not None:
        if not prices["secid"].eq(expected_secid).fillna(False).all():
            raise ValueError(f"{name} prices must have secid={expected_secid}.")

    columns = ["date", "close"]
    has_vendor_return = "return" in prices.columns
    if has_vendor_return:
        columns.append("return")
    result = prices[columns].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any():
        raise ValueError(f"{name} price dates cannot be missing.")
    if result["date"].duplicated().any():
        raise ValueError(f"{name} prices must contain at most one row per date.")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    finite_close = result["close"].dropna()
    if not np.isfinite(finite_close).all() or (finite_close <= 0).any():
        raise ValueError(f"{name} closes must be finite and positive when present.")
    if has_vendor_return:
        result["return"] = pd.to_numeric(result["return"], errors="coerce")
    return result.sort_values("date").reset_index(drop=True)


def build_market_state(spx_prices, vix_prices):
    """Build daily SPX log-return/RV measures and align same-date VIX closes.

    Inputs are raw per-security price extracts with date and close columns.
    OptionMetrics' optional return is retained as spx_vendor_return only for
    comparison; all calculated returns and realized volatility use SPX closes.
    """
    spx = _prepare_prices(spx_prices, "SPX", expected_secid=SPX_SECURITY_ID)
    vix = _prepare_prices(vix_prices, "VIX")

    dates = pd.DatetimeIndex(spx["date"]).union(pd.DatetimeIndex(vix["date"]))
    spx = spx.set_index("date").reindex(dates)
    vix_close = vix.set_index("date")["close"].reindex(dates)

    spx_return = np.log(spx["close"] / spx["close"].shift(1))
    return_sq = spx_return**2
    market = pd.DataFrame({
        "date": dates,
        "spx_close": spx["close"].to_numpy(),
        "spx_return": spx_return.to_numpy(),
        "spx_abs_return": spx_return.abs().to_numpy(),
        "spx_return_sq": return_sq.to_numpy(),
        "rv_5": np.sqrt(
            TRADING_DAYS_PER_YEAR / 5
            * return_sq.rolling(5, min_periods=5).sum()
        ).to_numpy(),
        "rv_20": np.sqrt(
            TRADING_DAYS_PER_YEAR / 20
            * return_sq.rolling(20, min_periods=20).sum()
        ).to_numpy(),
    })
    if "return" in spx:
        market["spx_vendor_return"] = spx["return"].to_numpy()

    market["vix_close"] = vix_close.to_numpy()
    return market


def load_market_state(spx_path, vix_path):
    """Load raw SPX and VIX security-price CSV extracts and build market state."""
    spx_prices = pd.read_csv(spx_path)
    vix_prices = pd.read_csv(vix_path)
    return build_market_state(spx_prices, vix_prices)


def merge_market_state(daily_surface, market_state):
    """Left-join market state onto daily option-surface rows by quote date."""
    if "quote_date" not in daily_surface:
        raise ValueError("daily_surface must contain quote_date.")
    if "date" not in market_state:
        raise ValueError("market_state must contain date.")

    surface = daily_surface.copy()
    market = market_state.copy()
    surface["quote_date"] = pd.to_datetime(surface["quote_date"], errors="raise")
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    if surface["quote_date"].isna().any() or surface["quote_date"].duplicated().any():
        raise ValueError("daily_surface quote_date values must be nonmissing and unique.")
    if market["date"].isna().any() or market["date"].duplicated().any():
        raise ValueError("market_state dates must be nonmissing and unique.")

    surface = surface.rename(columns={"quote_date": "date"})
    return surface.merge(
        market, on="date", how="left", validate="one_to_one", sort=False,
    ).sort_values("date", ignore_index=True)
