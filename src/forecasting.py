"""Leakage-safe development forecasts for the two five-session skew targets."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.market_state import build_market_state, merge_market_state, spx_trading_dates
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
SURFACE_FEATURES = (
    "skew_60", "skew_30_60", "skew_60_90",
    "atm_iv_30", "atm_iv_30_60", "atm_iv_60_90",
)
REALIZED_FEATURES = (
    "spx_return", "spx_abs_return", "spx_return_sq", "rv_5", "rv_20",
)
LINEAR_MODELS = (
    "m3_own_dynamics", "m4_surface_state", "m5a_realized_state", "m5b_vix_state",
)
ALL_MODELS = (
    "m0_persistence", "m1_mean", "m2_mean_reversion", *LINEAR_MODELS,
)
ML_MODELS = ("m6_elastic_net", "m7_hist_boosting")
FULL_MODEL_LADDER = (*ALL_MODELS, *ML_MODELS)
ELASTIC_NET_GRID = tuple(
    (alpha, l1_ratio)
    for alpha in (1e-4, 1e-3, 1e-2)
    for l1_ratio in (0.25, 0.75)
)
BOOSTING_GRID = (50, 100)
INNER_BLOCK_SIZE = 30
MIN_INNER_TRAIN = 100


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


def build_linear_development_dataset(metrics, spx_prices, vix_prices):
    """Add declared market features to the development panel by exact date."""
    metric_dates = pd.to_datetime(metrics["quote_date"], errors="raise")
    spx_dates = pd.to_datetime(spx_prices["date"], errors="raise")
    vix_dates = pd.to_datetime(vix_prices["date"], errors="raise")
    if metric_dates.isna().any() or spx_dates.isna().any() or vix_dates.isna().any():
        raise ValueError("Input dates cannot be missing.")
    development = build_development_dataset(
        metrics.loc[metric_dates <= DEV_END], spx_prices.loc[spx_dates <= DEV_END]
    )
    market = build_market_state(
        spx_prices.loc[spx_dates <= DEV_END], vix_prices.loc[vix_dates <= DEV_END]
    )
    market = market[["date", *REALIZED_FEATURES, "vix_close"]]
    result = merge_market_state(development, market)
    result["d_skew_30_60"] = result["skew_30_60"].diff()
    return result


def linear_feature_sets(target):
    """Return the exact frozen M3–M5b predictor blocks for a headline target."""
    if target not in TARGET_STATES:
        raise ValueError(f"Unsupported headline target: {target}")
    zscore = TARGET_STATES[target][1]
    change = "d_skew_30" if target == "y_skew_30_5d" else "d_skew_30_60"
    own = (zscore, change)
    surface = own + SURFACE_FEATURES
    realized = surface + REALIZED_FEATURES
    return dict(zip(LINEAR_MODELS, (
        own, surface, realized, realized + ("vix_close",),
    )))


def forecast_linear_development(dataset, spx_prices, target):
    """Extend unchanged M0–M2 forecasts with expanding, purged M3–M5b OLS fits."""
    baseline = forecast_development(dataset, spx_prices, target)
    rows = add_development_zscores(dataset)
    rows["d_skew_30"] = rows["skew_30"].diff()
    rows["d_skew_30_60"] = rows["skew_30_60"].diff()
    feature_sets = linear_feature_sets(target)
    required = {column for features in feature_sets.values() for column in features}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing linear forecast columns: {sorted(missing)}")

    outcomes = pd.to_numeric(rows[target], errors="coerce").to_numpy(dtype=float)
    features = {
        column: pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
        for column in required
    }
    matrices = {
        model: np.column_stack([features[column] for column in columns])
        for model, columns in feature_sets.items()
    }
    coefficients = []
    for origin in range(INITIAL_TRAINING_SESSIONS, len(rows)):
        training_end = origin - HORIZON + 1
        for model, columns in feature_sets.items():
            matrix = matrices[model]
            valid = np.isfinite(outcomes[:training_end]) & np.isfinite(matrix[:training_end]).all(axis=1)
            train_x = matrix[:training_end][valid]
            train_y = outcomes[:training_end][valid]
            if len(train_y) < len(columns) + 1:
                continue
            design = np.column_stack((np.ones(len(train_y)), train_x))
            beta, _, rank, _ = np.linalg.lstsq(design, train_y, rcond=None)
            if rank < len(columns) + 1:
                continue
            for feature, value in zip(("intercept", *columns), beta):
                coefficients.append({
                    "quote_date": rows["quote_date"].iloc[origin],
                    "session_index": origin,
                    "target": target,
                    "model": model,
                    "n_train": len(train_y),
                    "feature": feature,
                    "coefficient": float(value),
                })
            if np.isfinite(matrix[origin]).all():
                baseline.loc[origin - INITIAL_TRAINING_SESSIONS, model] = float(
                    np.r_[1.0, matrix[origin]] @ beta
                )
    for model in LINEAR_MODELS:
        if model not in baseline:
            baseline[model] = np.nan
    return baseline, pd.DataFrame(coefficients, columns=[
        "quote_date", "session_index", "target", "model", "n_train",
        "feature", "coefficient",
    ])


def score_linear_development_forecasts(forecasts):
    """Score all seven frozen models on one strict common set of development dates."""
    required = {"quote_date", "actual", *ALL_MODELS}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")
    dates = pd.to_datetime(forecasts["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.between(DEV_START, DEV_END).all():
        raise ValueError("Scoring requires unique development forecast dates.")

    common = forecasts.copy()
    for column in ("actual", *ALL_MODELS):
        values = pd.to_numeric(common[column], errors="coerce")
        common[column] = values.where(np.isfinite(values))
    common = common.dropna(subset=["actual", *ALL_MODELS]).sort_values("quote_date")
    if common.empty:
        raise ValueError("No common realised dates for M0–M5b.")

    actual = common["actual"].to_numpy()
    persistence_sse = np.square(actual - common["m0_persistence"].to_numpy()).sum()
    mean_reversion_sse = np.square(actual - common["m2_mean_reversion"].to_numpy()).sum()
    summary = []
    for model in ALL_MODELS:
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
            "r2_vs_mean_reversion": (
                1 - sse / mean_reversion_sse
                if model in LINEAR_MODELS and mean_reversion_sse > 0 else np.nan
            ),
            "correlation": correlation,
            "directional_accuracy": (
                float(np.mean(np.sign(prediction) == np.sign(actual)))
                if model != "m0_persistence" else np.nan
            ),
        })
    return pd.DataFrame(summary), common


def inner_validation_blocks(origin):
    """Two latest 30-session blocks with labels matured by the outer origin."""
    end = origin - HORIZON + 1
    middle = end - INNER_BLOCK_SIZE
    start = middle - INNER_BLOCK_SIZE
    if start < 0:
        return None
    return ((start, middle), (middle, end))


def _fit_ml_model(model, candidate, x_train, y_train):
    if model == "m6_elastic_net":
        alpha, l1_ratio = candidate
        estimator = make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=10000),
        )
    elif model == "m7_hist_boosting":
        estimator = HistGradientBoostingRegressor(
            loss="squared_error", max_depth=2, min_samples_leaf=30,
            learning_rate=0.05, max_iter=candidate,
            early_stopping=False, random_state=0,
        )
    else:
        raise ValueError(f"Unsupported ML model: {model}")
    return estimator.fit(x_train, y_train)


def _select_candidate(model, candidate_rmse):
    """Resolve exact RMSE ties toward stronger regularisation or fewer trees."""
    if model == "m6_elastic_net":
        return min(candidate_rmse, key=lambda c: (
            candidate_rmse[c], -c[0], -c[1],
        ))
    if model == "m7_hist_boosting":
        return min(candidate_rmse, key=lambda c: (candidate_rmse[c], c))
    raise ValueError(f"Unsupported ML model: {model}")


def forecast_ml_development(dataset, spx_prices, target):
    """Extend unchanged M0–M5b with purged, nested M6/M7 development forecasts."""
    forecasts, _ = forecast_linear_development(dataset, spx_prices, target)
    rows = add_development_zscores(dataset)
    rows["d_skew_30"] = rows["skew_30"].diff()
    rows["d_skew_30_60"] = rows["skew_30_60"].diff()
    columns = linear_feature_sets(target)["m5b_vix_state"]
    matrix = np.column_stack([
        pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
        for column in columns
    ])
    outcomes = pd.to_numeric(rows[target], errors="coerce").to_numpy(dtype=float)
    complete = np.isfinite(matrix).all(axis=1) & np.isfinite(outcomes)
    diagnostics = []
    for origin in range(INITIAL_TRAINING_SESSIONS, len(rows)):
        outer_end = origin - HORIZON + 1
        outer_train = np.flatnonzero(complete[:outer_end])
        blocks = inner_validation_blocks(origin)
        for model, candidates in (
            ("m6_elastic_net", ELASTIC_NET_GRID),
            ("m7_hist_boosting", BOOSTING_GRID),
        ):
            record = {
                "target": target,
                "quote_date": rows["quote_date"].iloc[origin],
                "session_index": origin,
                "model": model,
                "outer_train_n": len(outer_train),
            }
            if not np.isfinite(matrix[origin]).all():
                diagnostics.append({**record, "status": "missing_origin"})
                continue
            if blocks is None or complete[:blocks[0][0] - HORIZON + 1].sum() < MIN_INNER_TRAIN:
                diagnostics.append({**record, "status": "insufficient_inner_history"})
                continue

            fold_data = []
            for start, stop in blocks:
                train = np.flatnonzero(complete[:start - HORIZON + 1])
                validation = np.flatnonzero(complete[start:stop]) + start
                if not len(train) or not len(validation):
                    break
                fold_data.append((train, validation))
            if len(fold_data) != 2:
                diagnostics.append({**record, "status": "incomplete_inner_blocks"})
                continue

            candidate_rmse = {}
            for candidate in candidates:
                squared_error = 0.0
                validation_count = 0
                for train, validation in fold_data:
                    fitted = _fit_ml_model(model, candidate, matrix[train], outcomes[train])
                    errors = outcomes[validation] - fitted.predict(matrix[validation])
                    squared_error += float(np.square(errors).sum())
                    validation_count += len(validation)
                candidate_rmse[candidate] = float(np.sqrt(squared_error / validation_count))
            selected = _select_candidate(model, candidate_rmse)
            fitted = _fit_ml_model(model, selected, matrix[outer_train], outcomes[outer_train])
            forecasts.loc[origin - INITIAL_TRAINING_SESSIONS, model] = float(
                fitted.predict(matrix[origin:origin + 1])[0]
            )
            for candidate, rmse in candidate_rmse.items():
                diagnostics.append({
                    **record,
                    "status": "selected" if candidate == selected else "candidate",
                    "alpha": candidate[0] if model == "m6_elastic_net" else np.nan,
                    "l1_ratio": candidate[1] if model == "m6_elastic_net" else np.nan,
                    "max_iter": candidate if model == "m7_hist_boosting" else np.nan,
                    "inner_rmse": rmse,
                    "inner_n": validation_count,
                    "fold1_n": len(fold_data[0][1]),
                    "fold2_n": len(fold_data[1][1]),
                    "selected": candidate == selected,
                    "nonzero_coefficients": (
                        int(np.count_nonzero(fitted[-1].coef_))
                        if model == "m6_elastic_net" and candidate == selected else np.nan
                    ),
                })
    for model in ML_MODELS:
        if model not in forecasts:
            forecasts[model] = np.nan
    return forecasts, pd.DataFrame(diagnostics)


def score_ml_development_forecasts(forecasts):
    """Score the frozen M0–M7 ladder on one strict set of development dates."""
    required = {"quote_date", "actual", *FULL_MODEL_LADDER}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")
    dates = pd.to_datetime(forecasts["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.between(DEV_START, DEV_END).all():
        raise ValueError("Scoring requires unique development forecast dates.")

    common = forecasts.copy()
    for column in ("actual", *FULL_MODEL_LADDER):
        values = pd.to_numeric(common[column], errors="coerce")
        common[column] = values.where(np.isfinite(values))
    common = common.dropna(subset=["actual", *FULL_MODEL_LADDER]).sort_values("quote_date")
    if common.empty:
        raise ValueError("No common realised dates for M0–M7.")

    actual = common["actual"].to_numpy()
    persistence_sse = np.square(actual - common["m0_persistence"].to_numpy()).sum()
    mean_reversion_sse = np.square(actual - common["m2_mean_reversion"].to_numpy()).sum()
    summary = []
    for model in FULL_MODEL_LADDER:
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
            "r2_vs_mean_reversion": (
                1 - sse / mean_reversion_sse
                if model not in FULL_MODEL_LADDER[:3] and mean_reversion_sse > 0 else np.nan
            ),
            "correlation": correlation,
            "directional_accuracy": (
                float(np.mean(np.sign(prediction) == np.sign(actual)))
                if model != "m0_persistence" else np.nan
            ),
        })
    return pd.DataFrame(summary), common


def add_locked_zscores(rows):
    """Extend the frozen past-only z-score calculation across confirmation dates."""
    result = rows.copy()
    dates = pd.to_datetime(result["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Quote dates must be unique, present and sorted.")
    if not dates.between(DEV_START, HOLDOUT_END).all():
        raise ValueError("Z-score dates must be in the frozen sample.")

    for state, zscore in TARGET_STATES.values():
        values = pd.to_numeric(result[state], errors="coerce")
        values = values.where(np.isfinite(values))
        history = values.shift(1).expanding(min_periods=ZSCORE_WARMUP)
        mean = history.mean()
        std = history.std(ddof=1)
        result[zscore] = ((values - mean) / std).where(std > 0)
    return result


def build_locked_feature_panel(metrics, spx_prices, vix_prices):
    """Build causal features on every 2023–2025 SPX session, without targets."""
    metric_dates = pd.to_datetime(metrics["quote_date"], errors="raise")
    spx_dates = pd.to_datetime(spx_prices["date"], errors="raise")
    vix_dates = pd.to_datetime(vix_prices["date"], errors="raise")
    if metric_dates.isna().any() or spx_dates.isna().any() or vix_dates.isna().any():
        raise ValueError("Input dates cannot be missing.")

    spx = spx_prices.loc[spx_dates <= HOLDOUT_END]
    vix = vix_prices.loc[vix_dates <= HOLDOUT_END]
    calendar = spx_trading_dates(spx)
    dates = calendar[(calendar >= DEV_START) & (calendar <= HOLDOUT_END)]
    if dates.empty or dates[0] != DEV_START or dates[-1] != HOLDOUT_END:
        raise ValueError("SPX prices must cover the full frozen sample.")

    selected_metrics = metrics.loc[metric_dates.between(DEV_START, HOLDOUT_END)]
    daily = build_daily_time_series(selected_metrics, spx)
    daily = daily.set_index("quote_date").reindex(dates)
    daily.index.name = "quote_date"
    features = daily.reset_index()
    features.insert(1, "session_index", np.arange(len(features)))
    features = add_locked_zscores(features)
    features["d_skew_30"] = features["skew_30"].diff()
    features["d_skew_30_60"] = features["skew_30_60"].diff()

    market = build_market_state(spx, vix)[["date", *REALIZED_FEATURES, "vix_close"]]
    return merge_market_state(features, market)


def fit_locked_models(dataset, spx_prices):
    """Select and fit final M6/M2 models using matured development labels only."""
    required = {"quote_date", "session_index", "skew_30", "skew_30_60", *TARGET_STATES}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing development columns: {sorted(missing)}")
    dates = pd.to_datetime(dataset["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Forecast dates must be unique, present and sorted.")
    if not dates.between(DEV_START, HOLDOUT_END).all():
        raise ValueError("Rows must stay inside the frozen sample.")

    development = dataset.loc[dates <= DEV_END].copy().reset_index(drop=True)
    spx_dates = pd.to_datetime(spx_prices["date"], errors="raise")
    if spx_dates.isna().any():
        raise ValueError("SPX dates cannot be missing.")
    calendar = spx_trading_dates(spx_prices.loc[spx_dates <= DEV_END])
    expected = calendar[(calendar >= DEV_START) & (calendar <= DEV_END)]
    if expected.empty or expected[0] != DEV_START or expected[-1] != DEV_END or not pd.DatetimeIndex(
        development["quote_date"]
    ).equals(expected):
        raise ValueError("Development rows must cover every SPX session through 2024-12-31.")
    if not np.array_equal(development["session_index"], np.arange(len(development))):
        raise ValueError("Development session_index cannot compress the SPX calendar.")

    development = add_development_zscores(development)
    development["d_skew_30"] = development["skew_30"].diff()
    development["d_skew_30_60"] = development["skew_30_60"].diff()
    feature_names = linear_feature_sets("y_skew_30_5d")["m5b_vix_state"]
    missing = set(feature_names) - set(development.columns)
    if missing:
        raise ValueError(f"Missing M6 features: {sorted(missing)}")

    matrix = np.column_stack([
        pd.to_numeric(development[name], errors="coerce").to_numpy(dtype=float)
        for name in feature_names
    ])
    y_30 = pd.to_numeric(development["y_skew_30_5d"], errors="coerce").to_numpy(dtype=float)
    complete = np.isfinite(matrix).all(axis=1) & np.isfinite(y_30)
    last_origin = len(development) - 1
    matured_end = last_origin - HORIZON + 1
    blocks = inner_validation_blocks(last_origin)
    if blocks is None or complete[:blocks[0][0] - HORIZON + 1].sum() < MIN_INNER_TRAIN:
        raise ValueError("Insufficient development history for frozen M6 validation.")

    fold_data = []
    for start, stop in blocks:
        train = np.flatnonzero(complete[:start - HORIZON + 1])
        validation = np.flatnonzero(complete[start:stop]) + start
        if not len(train) or not len(validation):
            raise ValueError("Both frozen inner-validation blocks need complete labels.")
        fold_data.append((train, validation))

    candidate_rmse = {}
    for candidate in ELASTIC_NET_GRID:
        squared_error = 0.0
        count = 0
        for train, validation in fold_data:
            fitted = _fit_ml_model("m6_elastic_net", candidate, matrix[train], y_30[train])
            squared_error += float(np.square(y_30[validation] - fitted.predict(matrix[validation])).sum())
            count += len(validation)
        candidate_rmse[candidate] = float(np.sqrt(squared_error / count))
    selected = _select_candidate("m6_elastic_net", candidate_rmse)
    final_train = np.flatnonzero(complete[:matured_end])
    m6 = _fit_ml_model("m6_elastic_net", selected, matrix[final_train], y_30[final_train])

    m2 = {}
    for target, (_, zscore) in TARGET_STATES.items():
        outcome = pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(development[zscore], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(outcome[:matured_end]) & np.isfinite(z[:matured_end])
        x, y = z[:matured_end][valid], outcome[:matured_end][valid]
        if len(y) < 2 or np.ptp(x) == 0:
            raise ValueError(f"Insufficient development observations for {target} M2.")
        intercept, slope = np.linalg.lstsq(
            np.column_stack((np.ones(len(x)), x)), y, rcond=None
        )[0]
        m2[target] = {"intercept": float(intercept), "slope": float(slope), "n_train": len(y)}

    return {
        "m6_model": m6,
        "m6_features": feature_names,
        "m6_alpha": selected[0],
        "m6_l1_ratio": selected[1],
        "m6_inner_rmse": candidate_rmse,
        "m6_inner_blocks": blocks,
        "m6_n_train": len(final_train),
        "m2_coefficients": m2,
        "last_matured_origin": development["quote_date"].iloc[matured_end - 1],
    }


def forecast_locked_holdout(fitted, feature_panel, spx_prices):
    """Predict 2025 from frozen models and causal features; no outcomes required."""
    required = {"quote_date", "session_index", "skew_30", "skew_30_60", *fitted["m6_features"]}
    missing = required - set(feature_panel.columns)
    # Z-scores and one-session changes are recalculated below, not trusted as inputs.
    missing -= {"z_skew_30", "d_skew_30"}
    if missing:
        raise ValueError(f"Missing locked feature columns: {sorted(missing)}")
    dates = pd.to_datetime(feature_panel["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique, present and sorted.")
    calendar = spx_trading_dates(spx_prices)
    expected = calendar[(calendar >= DEV_START) & (calendar <= HOLDOUT_END)]
    if (expected.empty or expected[0] != DEV_START or expected[-1] != HOLDOUT_END
            or not pd.DatetimeIndex(dates).equals(expected)):
        raise ValueError("Features must include every SPX session through 2025-08-29.")
    if not np.array_equal(feature_panel["session_index"], np.arange(len(feature_panel))):
        raise ValueError("Feature session_index cannot compress the SPX calendar.")

    rows = add_locked_zscores(feature_panel)
    rows["d_skew_30"] = rows["skew_30"].diff()
    rows["d_skew_30_60"] = rows["skew_30_60"].diff()
    holdout = rows.loc[dates > DEV_END].copy()
    matrix = np.column_stack([
        pd.to_numeric(holdout[name], errors="coerce").to_numpy(dtype=float)
        for name in fitted["m6_features"]
    ])
    predictions = {}
    for target, (state, zscore) in TARGET_STATES.items():
        prediction = holdout[["quote_date", "session_index"]].copy()
        valid_state = np.isfinite(pd.to_numeric(holdout[state], errors="coerce"))
        valid_z = np.isfinite(pd.to_numeric(holdout[zscore], errors="coerce"))
        coefficients = fitted["m2_coefficients"][target]
        prediction["m0_persistence"] = np.where(valid_state, 0.0, np.nan)
        prediction["m2_mean_reversion"] = np.where(
            valid_state & valid_z,
            coefficients["intercept"] + coefficients["slope"] * holdout[zscore],
            np.nan,
        )
        if target == "y_skew_30_5d":
            prediction["m6_elastic_net"] = np.nan
            valid_m6 = np.isfinite(matrix).all(axis=1)
            prediction.loc[valid_m6, "m6_elastic_net"] = fitted["m6_model"].predict(
                matrix[valid_m6]
            )
        predictions[target] = prediction.reset_index(drop=True)
    return predictions


def build_locked_outcomes(feature_panel, spx_prices):
    """Construct confirmation labels separately from the prediction API."""
    dates = pd.to_datetime(feature_panel["quote_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Feature dates must be unique and present.")
    holdout = feature_panel.loc[dates.between(DEV_END + pd.Timedelta(days=1), HOLDOUT_END)]
    calendar = spx_trading_dates(spx_prices)
    expected = calendar[(calendar > DEV_END) & (calendar <= HOLDOUT_END)]
    if not pd.DatetimeIndex(holdout["quote_date"]).equals(expected):
        raise ValueError("Outcome rows must preserve every 2025 SPX session.")
    targets = add_forward_targets(
        holdout[["quote_date", "skew_30", "skew_30_60", "rr25_30_60"]], spx_prices
    )
    return targets[["quote_date", "y_skew_30_5d", "y_skew_spread_5d"]]


def score_locked_holdout(predictions, outcomes, target):
    """Score frozen predictions only when separately supplied outcomes are available."""
    if target not in TARGET_STATES:
        raise ValueError(f"Unsupported headline target: {target}")
    models = ("m0_persistence", "m2_mean_reversion")
    if target == "y_skew_30_5d":
        models += ("m6_elastic_net",)
    required = {"quote_date", *models}
    if required - set(predictions.columns) or {"quote_date", target} - set(outcomes.columns):
        raise ValueError("Missing locked prediction or outcome columns.")
    dates = pd.to_datetime(predictions["quote_date"], errors="raise")
    outcome_dates = pd.to_datetime(outcomes["quote_date"], errors="raise")
    if (dates.isna().any() or dates.duplicated().any()
            or outcome_dates.isna().any() or outcome_dates.duplicated().any()
            or not dates.between(DEV_END + pd.Timedelta(days=1), HOLDOUT_END).all()
            or not outcome_dates.between(DEV_END + pd.Timedelta(days=1), HOLDOUT_END).all()):
        raise ValueError("Scoring requires unique locked-confirmation dates.")
    prediction_rows = predictions.copy()
    outcome_rows = outcomes[["quote_date", target]].copy()
    prediction_rows["quote_date"] = dates
    outcome_rows["quote_date"] = outcome_dates
    merged = prediction_rows.merge(
        outcome_rows, on="quote_date", how="left",
        validate="one_to_one", indicator=True,
    )
    if merged["_merge"].ne("both").any():
        raise ValueError("Every prediction date needs a corresponding outcome row.")
    common = merged.drop(columns="_merge").rename(columns={target: "actual"})
    for column in ("actual", *models):
        values = pd.to_numeric(common[column], errors="coerce")
        common[column] = values.where(np.isfinite(values))
    common = common.dropna(subset=["actual", *models]).sort_values("quote_date")
    if common.empty:
        raise ValueError("No common realised locked-confirmation dates.")

    actual = common["actual"].to_numpy()
    m0_sse = float(np.square(actual - common["m0_persistence"].to_numpy()).sum())
    m2_sse = float(np.square(actual - common["m2_mean_reversion"].to_numpy()).sum())
    summary = []
    for model in models:
        prediction = common[model].to_numpy()
        errors = actual - prediction
        sse = float(np.square(errors).sum())
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
            "correlation": correlation,
            "directional_accuracy": (
                float(np.mean(np.sign(prediction) == np.sign(actual)))
                if model != "m0_persistence" else np.nan
            ),
            "r2_vs_persistence": 1 - sse / m0_sse if m0_sse > 0 else np.nan,
            "r2_vs_mean_reversion": (
                1 - sse / m2_sse
                if model == "m6_elastic_net" and m2_sse > 0 else np.nan
            ),
        })
    return pd.DataFrame(summary), common


def cumulative_locked_gains(common, target):
    """Return the pre-specified cumulative squared-error comparisons."""
    if target not in TARGET_STATES:
        raise ValueError(f"Unsupported headline target: {target}")
    required = {"quote_date", "actual", "m0_persistence", "m2_mean_reversion"}
    if target == "y_skew_30_5d":
        required.add("m6_elastic_net")
    missing = required - set(common.columns)
    if missing:
        raise ValueError(f"Missing common-date columns: {sorted(missing)}")
    actual = common["actual"]
    m0_error = (actual - common["m0_persistence"]) ** 2
    m2_error = (actual - common["m2_mean_reversion"]) ** 2
    gains = common[["quote_date"]].copy()
    gains["m2_vs_m0"] = (m0_error - m2_error).cumsum()
    if target == "y_skew_30_5d":
        m6_error = (actual - common["m6_elastic_net"]) ** 2
        gains["m6_vs_m0"] = (m0_error - m6_error).cumsum()
        gains["m6_vs_m2"] = (m2_error - m6_error).cumsum()
    return gains
