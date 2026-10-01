"""Leakage-safe development forecasts for the two five-session skew targets."""

import numpy as np
import pandas as pd

from src.market_state import spx_trading_dates
from src.time_series import add_forward_targets, build_daily_time_series


DEV_START = pd.Timestamp("2023-01-03")
DEV_END = pd.Timestamp("2024-12-31")
HOLDOUT_END = pd.Timestamp("2025-08-29")
HORIZON = 5
INITIAL_TRAINING_SESSIONS = 252
ZSCORE_WARMUP = 60
TARGET_STATES = {
    "y_skew_30_5d": ("skew_30", "z_skew_30"),
    "y_skew_spread_5d": ("skew_30_60", "z_skew_30_60"),
}


def split_forecast_dates(rows):
    """Separate development and locked-confirmation dates without changing rows."""
    dates = pd.to_datetime(rows["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("quote_date must be nonmissing and unique.")
    development = rows.loc[dates.between(DEV_START, DEV_END)].copy()
    holdout = rows.loc[dates.between(DEV_END + pd.Timedelta(days=1), HOLDOUT_END)].copy()
    return development, holdout


def add_development_zscores(rows):
    """Use 60 prior valid states, never the current or a future state, as reference."""
    result = rows.copy()
    dates = pd.to_datetime(result["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Development quote dates must be unique, present and sorted.")
    if not dates.between(DEV_START, DEV_END).all():
        raise ValueError("Development z-scores cannot include confirmation dates.")

    for state, zscore in TARGET_STATES.values():
        values = pd.to_numeric(result[state], errors="coerce")
        values = values.where(np.isfinite(values))
        history = values.shift(1).expanding(min_periods=ZSCORE_WARMUP)
        mean = history.mean()
        std = history.std(ddof=1)
        result[zscore] = ((values - mean) / std).where(std > 0)
    return result


def build_development_dataset(metrics, spx_prices):
    """Build QC-masked targets on every development SPX session, including gaps."""
    dates = spx_trading_dates(spx_prices)
    development_dates = dates[(dates >= DEV_START) & (dates <= DEV_END)]
    if development_dates.empty:
        raise ValueError("SPX prices contain no development sessions.")

    daily = build_daily_time_series(metrics, spx_prices)
    daily = daily.set_index("quote_date").reindex(development_dates)
    daily.index.name = "quote_date"
    dataset = add_forward_targets(daily.reset_index(), spx_prices)
    dataset.insert(1, "session_index", np.arange(len(dataset)))
    return add_development_zscores(dataset)


def forecast_development(dataset, spx_prices, target):
    """Issue M0/M1/M2 forecasts using only five-session-matured training labels."""
    if target not in TARGET_STATES:
        raise ValueError(f"Unsupported headline target: {target}")
    state, zscore = TARGET_STATES[target]
    required = {"quote_date", "session_index", state, "skew_30", "skew_30_60", target}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    rows = dataset.copy()
    dates = pd.to_datetime(rows["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Forecast dates must be unique, present and sorted.")
    if not dates.between(DEV_START, DEV_END).all():
        raise ValueError("Development forecasts cannot include confirmation dates.")
    calendar = spx_trading_dates(spx_prices)
    expected = calendar[(calendar >= DEV_START) & (calendar <= DEV_END)]
    if not pd.DatetimeIndex(dates).equals(expected):
        raise ValueError("Development rows must include every SPX session.")
    if not np.array_equal(rows["session_index"], np.arange(len(rows))):
        raise ValueError("Development rows must retain every SPX session.")

    rows = add_development_zscores(rows)
    outcomes = pd.to_numeric(rows[target], errors="coerce").to_numpy(dtype=float)
    states = pd.to_numeric(rows[state], errors="coerce").to_numpy(dtype=float)
    zscores = pd.to_numeric(rows[zscore], errors="coerce").to_numpy(dtype=float)
    forecasts = []
    for origin in range(INITIAL_TRAINING_SESSIONS, len(rows)):
        # The last eligible training origin is t-5: its outcome is known at t.
        matured = slice(None, origin - HORIZON + 1)
        train_y = outcomes[matured]
        train_z = zscores[matured]
        valid_y = np.isfinite(train_y)
        valid_pair = valid_y & np.isfinite(train_z)
        record = {
            "quote_date": dates.iloc[origin],
            "session_index": origin,
            "target": target,
            "m0_persistence": np.nan,
            "m1_mean": np.nan,
            "m2_mean_reversion": np.nan,
            "n_matured": int(valid_y.sum()),
        }
        if np.isfinite(states[origin]):
            record["m0_persistence"] = 0.0
            if valid_y.any():
                record["m1_mean"] = float(train_y[valid_y].mean())
            if np.isfinite(zscores[origin]) and valid_pair.sum() >= 2:
                x = train_z[valid_pair]
                if np.ptp(x) > 0:
                    design = np.column_stack((np.ones(len(x)), x))
                    intercept, slope = np.linalg.lstsq(design, train_y[valid_pair], rcond=None)[0]
                    record["m2_mean_reversion"] = float(intercept + slope * zscores[origin])
        forecasts.append(record)
    result = pd.DataFrame(forecasts)
    result["actual"] = outcomes[INITIAL_TRAINING_SESSIONS:]
    return result.reindex(columns=[
        "quote_date", "session_index", "target", "actual", "m0_persistence", "m1_mean",
        "m2_mean_reversion", "n_matured",
    ])


def score_development_forecasts(forecasts):
    """Score M0/M1/M2 on identical, realised development forecast origins."""
    models = ("m0_persistence", "m1_mean", "m2_mean_reversion")
    required = {"quote_date", "actual", *models}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")
    dates = pd.to_datetime(forecasts["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.between(DEV_START, DEV_END).all():
        raise ValueError("Scoring requires unique development forecast dates.")

    common = forecasts.copy()
    for column in ("actual", *models):
        values = pd.to_numeric(common[column], errors="coerce")
        common[column] = values.where(np.isfinite(values))
    common = common.dropna(subset=["actual", *models]).sort_values("quote_date")
    if common.empty:
        raise ValueError("No common realised dates for M0/M1/M2.")

    actual = common["actual"].to_numpy()
    persistence_sse = np.square(actual - common["m0_persistence"].to_numpy()).sum()
    summary = []
    for model in models:
        prediction = common[model].to_numpy()
        errors = actual - prediction
        sse = np.square(errors).sum()
        correlation = np.nan
        if np.std(prediction) > 0 and np.std(actual) > 0:
            correlation = float(np.corrcoef(prediction, actual)[0, 1])
        summary.append({
            "model": model,
            "n_common": len(common),
            "first_date": common["quote_date"].iloc[0],
            "last_date": common["quote_date"].iloc[-1],
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))),
            "r2_vs_persistence": 1 - sse / persistence_sse if persistence_sse > 0 else np.nan,
            "correlation": correlation,
            "directional_accuracy": (
                float(np.mean(np.sign(prediction) == np.sign(actual)))
                if model != "m0_persistence" else np.nan
            ),
        })
    return pd.DataFrame(summary), common
