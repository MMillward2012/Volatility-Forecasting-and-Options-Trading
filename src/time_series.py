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
