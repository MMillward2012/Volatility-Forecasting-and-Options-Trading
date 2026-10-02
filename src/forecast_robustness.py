"""Pre-specified, post-confirmation checks of static M2 mean reversion."""

import numpy as np
import pandas as pd

from src.forecasting import (
    DEV_END, DEV_START, HOLDOUT_END, ZSCORE_WARMUP, linear_feature_sets,
)
from src.market_state import spx_trading_dates
from src.time_series import add_forward_targets


SPECIFICATIONS = {
    ("skew_30", 1): "y_skew_30_1d",
    ("skew_30", 5): "y_skew_30_5d",
    ("skew_30", 10): "y_skew_30_10d",
    ("skew_30_60", 1): "y_skew_spread_1d",
    ("skew_30_60", 5): "y_skew_spread_5d",
    ("skew_30_60", 10): "y_skew_spread_10d",
    ("rr25_30_60", 5): "y_rr25_spread_5d",
}
HAC_LAG = 4


def _validate_panel(panel, spx_prices):
    dates = pd.to_datetime(panel["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Quote dates must be unique, present and sorted.")
    calendar = spx_trading_dates(spx_prices)
    expected = calendar[(calendar >= DEV_START) & (calendar <= HOLDOUT_END)]
    if (expected.empty or expected[0] != DEV_START or expected[-1] != HOLDOUT_END
            or not pd.DatetimeIndex(dates).equals(expected)):
        raise ValueError("Robustness rows must cover every frozen SPX session.")
    if not np.array_equal(panel["session_index"], np.arange(len(panel))):
        raise ValueError("Robustness session_index cannot compress the SPX calendar.")
    return dates


def past_only_zscore(values):
    """Use the frozen 60-valid-observation expanding reference distribution."""
    state = pd.to_numeric(values, errors="coerce").astype(float)
    state = state.where(np.isfinite(state))
    history = state.shift(1).expanding(min_periods=ZSCORE_WARMUP)
    mean = history.mean()
    std = history.std(ddof=1)
    return ((state - mean) / std).where(std > 0)


def fit_static_m2(panel, spx_prices, state, horizon):
    """Fit one M2 on development labels whose endpoints remain in 2024."""
    if (state, horizon) not in SPECIFICATIONS:
        raise ValueError("Only the pre-specified state/horizon checks are supported.")
    dates = _validate_panel(panel, spx_prices)
    development = panel.loc[dates <= DEV_END]
    values = pd.to_numeric(development[state], errors="coerce").astype(float)
    values = values.where(np.isfinite(values))
    zscore = past_only_zscore(values)
    outcome = values.shift(-horizon) - values
    valid = np.isfinite(zscore) & np.isfinite(outcome)
    x = zscore[valid].to_numpy()
    y = outcome[valid].to_numpy()
    if len(y) < 2 or np.ptp(x) == 0:
        raise ValueError("Insufficient matured development observations for M2.")
    intercept, slope = np.linalg.lstsq(
        np.column_stack((np.ones(len(x)), x)), y, rcond=None
    )[0]
    return {
        "state": state,
        "horizon": horizon,
        "target": SPECIFICATIONS[(state, horizon)],
        "intercept": float(intercept),
        "slope": float(slope),
        "n_train": len(y),
        "last_matured_origin": development["quote_date"].iloc[-horizon - 1],
        "extreme_cutoff": float(np.quantile(np.abs(y), 0.99)),
    }


def forecast_static_m2(fitted, panel, spx_prices):
    """Use frozen coefficients with causal states, never 2025 outcomes."""
    dates = _validate_panel(panel, spx_prices)
    values = pd.to_numeric(panel[fitted["state"]], errors="coerce").astype(float)
    values = values.where(np.isfinite(values))
    zscore = past_only_zscore(values)
    holdout = dates > DEV_END
    result = panel.loc[holdout, ["quote_date", "session_index"]].copy()
    result["m0_persistence"] = np.where(np.isfinite(values[holdout]), 0.0, np.nan)
    forecast = fitted["intercept"] + fitted["slope"] * zscore[holdout]
    result["m2_mean_reversion"] = forecast.where(np.isfinite(values[holdout])).to_numpy()
    return result.reset_index(drop=True)


def build_robustness_outcomes(panel, spx_prices):
    """Construct the existing exact-session targets only after prediction."""
    dates = _validate_panel(panel, spx_prices)
    holdout = panel.loc[dates > DEV_END]
    return add_forward_targets(
        holdout[["quote_date", "skew_30", "skew_30_60", "rr25_30_60"]], spx_prices
    )[["quote_date", *SPECIFICATIONS.values()]]


def headline_30d_predictor_dates(panel, spx_prices):
    """Recover the frozen 30D M0/M2/M6 common-date mask without fitting M6."""
    dates = _validate_panel(panel, spx_prices)
    features = linear_feature_sets("y_skew_30_5d")["m5b_vix_state"]
    matrix = panel.loc[dates > DEV_END, list(features)].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(matrix).all(axis=1)
    return pd.DatetimeIndex(panel.loc[dates > DEV_END].loc[valid, "quote_date"])


def _pair_statistics(common):
    actual = common["actual"].to_numpy(dtype=float)
    m0 = common["m0_persistence"].to_numpy(dtype=float)
    m2 = common["m2_mean_reversion"].to_numpy(dtype=float)
    m0_error = actual - m0
    m2_error = actual - m2
    m0_sse = float(np.square(m0_error).sum())
    m2_sse = float(np.square(m2_error).sum())
    correlation = np.nan
    if len(common) > 1 and np.std(m2) > 0 and np.std(actual) > 0:
        correlation = float(np.corrcoef(m2, actual)[0, 1])
    return {
        "n": len(common),
        "rmse_m0": float(np.sqrt(np.mean(np.square(m0_error)))),
        "rmse_m2": float(np.sqrt(np.mean(np.square(m2_error)))),
        "mae_m0": float(np.mean(np.abs(m0_error))),
        "mae_m2": float(np.mean(np.abs(m2_error))),
        "r2_vs_m0": 1 - m2_sse / m0_sse if m0_sse > 0 else np.nan,
        "correlation": correlation,
        "directional_accuracy": float(np.mean(np.sign(m2) == np.sign(actual))),
    }


def score_static_m2(predictions, outcomes, target):
    """Score M0/M2 on identical valid confirmation origins."""
    if target not in SPECIFICATIONS.values():
        raise ValueError("Unsupported robustness target.")
    dates = pd.to_datetime(predictions["quote_date"], errors="raise")
    outcome_dates = pd.to_datetime(outcomes["quote_date"], errors="raise")
    if (dates.isna().any() or dates.duplicated().any()
            or outcome_dates.isna().any() or outcome_dates.duplicated().any()):
        raise ValueError("Scoring dates must be present and unique.")
    pred = predictions.copy()
    truth = outcomes[["quote_date", target]].copy()
    pred["quote_date"] = dates
    truth["quote_date"] = outcome_dates
    common = pred.merge(truth, on="quote_date", how="left", validate="one_to_one")
    common = common.rename(columns={target: "actual"})
    for column in ("actual", "m0_persistence", "m2_mean_reversion"):
        values = pd.to_numeric(common[column], errors="coerce")
        common[column] = values.where(np.isfinite(values))
    common = common.dropna(subset=["actual", "m0_persistence", "m2_mean_reversion"])
    common = common.sort_values("quote_date").reset_index(drop=True)
    if common.empty:
        raise ValueError("No common realised robustness dates.")
    return _pair_statistics(common), common


def nonoverlapping_offsets(common):
    """Partition by the original SPX session index, without compressing gaps."""
    rows = []
    for offset in range(5):
        subset = common.loc[common["session_index"].mod(5).eq(offset)]
        row = _pair_statistics(subset) if not subset.empty else {"n": 0}
        rows.append({"offset": offset, **row})
    return pd.DataFrame(rows)


def extreme_move_check(common, development_cutoff):
    """Apply the development-fixed 99th percentile threshold to confirmation."""
    if not np.isfinite(development_cutoff) or development_cutoff < 0:
        raise ValueError("Development cutoff must be finite and nonnegative.")
    retained = common.loc[common["actual"].abs().le(development_cutoff)]
    if retained.empty:
        raise ValueError("No confirmation rows remain after the fixed cutoff.")
    return {"development_cutoff": development_cutoff,
            "removed": len(common) - len(retained), **_pair_statistics(retained)}


def confirmation_top_one_percent_check(common):
    """Remove exactly the largest 1% of realised confirmation moves for diagnosis."""
    if len(common) < 2 or common["quote_date"].isna().any() or common["quote_date"].duplicated().any():
        raise ValueError("The confirmation sample needs at least two unique, dated rows.")
    k = max(1, int(np.ceil(0.01 * len(common))))
    ranked = common.assign(_absolute_move=common["actual"].abs()).sort_values(
        ["_absolute_move", "quote_date"], ascending=[False, True], kind="mergesort"
    )
    removed_dates = ranked["quote_date"].iloc[:k]
    retained = common.loc[~common["quote_date"].isin(removed_dates)]
    return {"removed": k, "removed_dates": tuple(removed_dates), **_pair_statistics(retained)}


def vix_regime_check(common, panel):
    """Describe existing predictions below/at-or-above VIX 20; never refit."""
    prices = panel[["quote_date", "vix_close"]]
    aligned = common.merge(prices, on="quote_date", how="left", validate="one_to_one")
    rows = []
    for label, mask in (("VIX < 20", aligned["vix_close"].lt(20)),
                        ("VIX >= 20", aligned["vix_close"].ge(20))):
        subset = aligned.loc[mask]
        rows.append({"regime": label, "small_sample": len(subset) < 30,
                     **(_pair_statistics(subset) if not subset.empty else {"n": 0})})
    return pd.DataFrame(rows)


def hac_loss_difference(common):
    """Bartlett/Newey-West SE for mean M0-minus-M2 squared loss, fixed lag 4."""
    actual = common["actual"].to_numpy(dtype=float)
    m0 = common["m0_persistence"].to_numpy(dtype=float)
    m2 = common["m2_mean_reversion"].to_numpy(dtype=float)
    differential = np.square(actual - m0) - np.square(actual - m2)
    n = len(differential)
    if n <= HAC_LAG or not np.isfinite(differential).all():
        raise ValueError("HAC requires more than four finite observations.")
    mean = float(differential.mean())
    centered = differential - mean
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, HAC_LAG + 1):
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2 * (1 - lag / (HAC_LAG + 1)) * autocovariance
    standard_error = float(np.sqrt(max(long_run_variance, 0.0) / n))
    return {"n": n, "lag": HAC_LAG, "mean_loss_difference": mean,
            "hac_standard_error": standard_error,
            "ci_lower": mean - 1.96 * standard_error,
            "ci_upper": mean + 1.96 * standard_error}


def cumulative_m2_gain(common):
    """Cumulative M2-versus-M0 squared-error improvement on scored dates."""
    gain = (np.square(common["actual"] - common["m0_persistence"])
            - np.square(common["actual"] - common["m2_mean_reversion"]))
    return pd.DataFrame({"quote_date": common["quote_date"], "gain": gain.cumsum()})
