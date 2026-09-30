"""Build daily forecasting variables from long-form skew metric rows."""

from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "atm_skew_slope": ("slope_valid", "skew"),
    "rr25_downside": ("rr25_valid", "rr25"),
    "atm_iv": ("atm_valid", "atm_iv"),
}
TENORS = (30, 60, 90)
FORWARD_HORIZONS = (1, 5, 10)


def load_skew_metrics(path):
    """Load the historical long-form metric CSV."""
    return pd.read_csv(path, parse_dates=["quote_date"])


def build_daily_time_series(metrics):
    """Reshape metric rows and add daily changes and cross-tenor differences."""
    required = {"quote_date", "target_days", *METRICS}
    required.update(flag for flag, _ in METRICS.values())
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing required metric columns: {sorted(missing)}")

    rows = metrics.copy()
    rows["quote_date"] = pd.to_datetime(rows["quote_date"], errors="raise")
    if rows["quote_date"].isna().any():
        raise ValueError("quote_date cannot be missing.")
    rows["target_days"] = pd.to_numeric(rows["target_days"], errors="raise")
    if not rows["target_days"].isin(TENORS).all():
        raise ValueError(f"target_days must contain only {TENORS}.")
    if rows.duplicated(["quote_date", "target_days"]).any():
        raise ValueError("Each (quote_date, target_days) pair must be unique.")

    dates = pd.Index(rows["quote_date"].drop_duplicates().sort_values(), name="quote_date")
    result = pd.DataFrame(index=dates)
    for value_column, (valid_column, prefix) in METRICS.items():
        values = pd.to_numeric(rows[value_column], errors="coerce")
        valid = rows[valid_column].eq(True).fillna(False)
        values = values.where(valid & np.isfinite(values))
        wide = rows.assign(_value=values).pivot(
            index="quote_date", columns="target_days", values="_value"
        ).reindex(index=dates, columns=TENORS)
        wide.columns = [f"{prefix}_{int(days)}" for days in wide.columns]
        result = result.join(wide)

    level_columns = list(result.columns)
    for column in level_columns:
        result[f"d_{column}"] = result[column].diff()

    for prefix in ("skew", "rr25", "atm_iv"):
        result[f"{prefix}_30_60"] = result[f"{prefix}_30"] - result[f"{prefix}_60"]
        result[f"{prefix}_60_90"] = result[f"{prefix}_60"] - result[f"{prefix}_90"]

    result.index.name = "quote_date"
    return result.reset_index()


def add_forward_targets(daily, canonical_quote_dates):
    """Add fixed 1/5/10-session targets using an explicit trading-date calendar.

    The canonical calendar must contain every session in the period. It may include
    dates absent from ``daily``; those dates are inserted with missing values so a
    target can never silently shorten its horizon.
    """
    required = {"quote_date", "skew_30", "skew_30_60", "rr25_30_60"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing required daily-series columns: {sorted(missing)}")

    rows = daily.copy()
    rows["quote_date"] = pd.to_datetime(rows["quote_date"], errors="raise")
    if rows["quote_date"].isna().any():
        raise ValueError("daily quote_date cannot be missing.")
    if rows["quote_date"].duplicated().any():
        raise ValueError("daily quote_date values must be unique.")

    dates = pd.DatetimeIndex(pd.to_datetime(canonical_quote_dates, errors="raise"))
    if dates.isna().any() or dates.empty:
        raise ValueError("canonical_quote_dates must be nonempty and contain no missing dates.")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("canonical_quote_dates must be unique and chronologically sorted.")
    if not rows["quote_date"].isin(dates).all():
        raise ValueError("Every daily quote date must appear in canonical_quote_dates.")

    aligned = rows.set_index("quote_date").reindex(dates)
    target_sources = {
        "y_skew_spread": "skew_30_60",
        "y_skew_30": "skew_30",
        "y_rr25_spread": "rr25_30_60",
    }
    for target, source in target_sources.items():
        values = pd.to_numeric(aligned[source], errors="coerce")
        for horizon in FORWARD_HORIZONS:
            aligned[f"{target}_{horizon}d"] = values.shift(-horizon) - values

    aligned.index.name = "quote_date"
    return aligned.reset_index()
