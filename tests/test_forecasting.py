import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    add_development_zscores,
    build_development_dataset,
    forecast_development,
    score_development_forecasts,
    split_forecast_dates,
)


def spx_prices(dates):
    return pd.DataFrame({
        "secid": 108105, "ticker": "SPX", "date": dates, "close": 4000.0,
    })


def metric_rows(dates):
    rows = []
    for index, date in enumerate(dates):
        for tenor in (30, 60, 90):
            rows.append({
                "quote_date": date, "target_days": tenor,
                "atm_skew_slope": 0.1 + index / 100 + tenor / 1000,
                "slope_valid": True,
                "rr25_downside": 0.2 + index / 100,
                "rr25_valid": True,
                "atm_iv": 0.15 + tenor / 1000,
                "atm_valid": True,
            })
    return pd.DataFrame(rows)


def forecast_rows(periods=270):
    dates = pd.bdate_range("2023-01-03", periods=periods)
    state = np.linspace(-1, 1, periods)
    rows = pd.DataFrame({
        "quote_date": dates,
        "session_index": np.arange(periods),
        "skew_30": 0.3 + state,
        "skew_30_60": 0.1 + state,
    })
    rows = add_development_zscores(rows)
    rows["y_skew_30_5d"] = 2 + 3 * rows["z_skew_30"]
    rows["y_skew_spread_5d"] = -1 + 2 * rows["z_skew_30_60"]
    return rows, spx_prices(dates)


def test_date_split_is_fixed_and_does_not_change_input():
    rows = pd.DataFrame({
        "quote_date": pd.to_datetime(["2023-01-03", "2024-12-31", "2025-01-02", "2025-08-29"]),
        "value": [1, 2, 3, 4],
    })
    original = rows.copy(deep=True)
    development, holdout = split_forecast_dates(rows)
    assert development.value.tolist() == [1, 2]
    assert holdout.value.tolist() == [3, 4]
    pd.testing.assert_frame_equal(rows, original)


def test_zscore_uses_60_prior_valid_states_only():
    dates = pd.bdate_range("2023-01-03", periods=80)
    rows = pd.DataFrame({
        "quote_date": dates,
        "skew_30": np.arange(80, dtype=float),
        "skew_30_60": np.arange(80, dtype=float) / 2,
    })
    original = rows.copy(deep=True)
    result = add_development_zscores(rows)
    assert result.loc[:59, "z_skew_30"].isna().all()
    assert result.loc[60, "z_skew_30"] == pytest.approx(
        (60 - np.mean(np.arange(60))) / np.std(np.arange(60), ddof=1)
    )
    changed = rows.copy()
    changed.loc[61:, "skew_30"] = 1_000_000
    assert add_development_zscores(changed).loc[60, "z_skew_30"] == pytest.approx(
        result.loc[60, "z_skew_30"]
    )
    pd.testing.assert_frame_equal(rows, original)


def test_zscore_warmup_counts_valid_observations_not_calendar_rows():
    dates = pd.bdate_range("2023-01-03", periods=80)
    state = np.arange(80, dtype=float)
    state[20:30] = np.nan
    rows = pd.DataFrame({
        "quote_date": dates, "skew_30": state, "skew_30_60": state,
    })
    result = add_development_zscores(rows)
    assert pd.isna(result.loc[69, "z_skew_30"])
    assert np.isfinite(result.loc[70, "z_skew_30"])


def test_constant_state_has_no_zscore():
    dates = pd.bdate_range("2023-01-03", periods=80)
    rows = pd.DataFrame({
        "quote_date": dates, "skew_30": 0.3, "skew_30_60": 0.1,
    })
    result = add_development_zscores(rows)
    assert result[["z_skew_30", "z_skew_30_60"]].isna().all().all()


def test_development_dataset_uses_spx_sessions_and_does_not_take_2025_endpoints():
    dates = pd.bdate_range("2024-12-18", "2025-01-10")
    metrics = metric_rows(dates)
    prices = spx_prices(dates)
    result = build_development_dataset(metrics, prices)
    dev_dates = dates[dates <= pd.Timestamp("2024-12-31")]
    assert result.quote_date.tolist() == dev_dates.tolist()
    assert result.session_index.tolist() == list(range(len(dev_dates)))
    assert result.loc[0, "y_skew_30_5d"] == pytest.approx(0.05)
    assert result.tail(5)[["y_skew_30_5d", "y_skew_spread_5d"]].isna().all().all()
    changed = metrics.copy()
    changed.loc[changed.quote_date > pd.Timestamp("2024-12-31"), "atm_skew_slope"] = 999
    pd.testing.assert_frame_equal(result, build_development_dataset(changed, prices))


def test_missing_metric_session_is_retained_and_not_bridged():
    dates = pd.bdate_range("2023-01-03", periods=12)
    metrics = metric_rows(dates)
    metrics = metrics.loc[metrics.quote_date.ne(dates[2])]
    result = build_development_dataset(metrics, spx_prices(dates))
    assert result.quote_date.tolist() == dates.tolist()
    assert pd.isna(result.loc[2, "skew_30"])
    assert pd.isna(result.loc[3, "d_skew_30"])
    assert pd.notna(result.loc[0, "y_skew_30_5d"])
    assert pd.isna(result.loc[2, "y_skew_30_5d"])


def test_m0_m1_m2_use_only_matured_labels():
    rows, prices = forecast_rows()
    result = forecast_development(rows, prices, "y_skew_30_5d")
    first = result.iloc[0]
    assert first.quote_date == rows.loc[252, "quote_date"]
    assert first.n_matured == 188
    assert first.m0_persistence == 0
    assert first.m1_mean == pytest.approx(rows.loc[:247, "y_skew_30_5d"].mean())
    assert first.m2_mean_reversion == pytest.approx(2 + 3 * rows.loc[252, "z_skew_30"])

    changed = rows.copy()
    changed.loc[248, "y_skew_30_5d"] = 1000
    changed_result = forecast_development(changed, prices, "y_skew_30_5d")
    assert changed_result.loc[0, "m1_mean"] == pytest.approx(first.m1_mean)
    assert changed_result.loc[0, "m2_mean_reversion"] == pytest.approx(first.m2_mean_reversion)
    assert changed_result.loc[1, "m1_mean"] != pytest.approx(result.loc[1, "m1_mean"])


def test_spread_forecast_uses_spread_state_and_missing_current_state_abstains():
    rows, prices = forecast_rows()
    rows.loc[252, "skew_30_60"] = np.nan
    result = forecast_development(rows, prices, "y_skew_spread_5d")
    assert result.loc[0, ["m0_persistence", "m1_mean", "m2_mean_reversion"]].isna().all()
    expected_z = add_development_zscores(rows).loc[253, "z_skew_30_60"]
    assert result.loc[1, "m2_mean_reversion"] == pytest.approx(
        -1 + 2 * expected_z
    )


def test_forecast_recomputes_past_only_zscores():
    rows, prices = forecast_rows()
    expected = forecast_development(rows, prices, "y_skew_30_5d")
    rows["z_skew_30"] = 1_000_000
    actual = forecast_development(rows, prices, "y_skew_30_5d")
    pd.testing.assert_series_equal(
        actual["m2_mean_reversion"], expected["m2_mean_reversion"]
    )


def test_forecast_rejects_missing_sessions_and_confirmation_rows():
    rows, prices = forecast_rows()
    with pytest.raises(ValueError, match="every SPX session"):
        forecast_development(rows.drop(index=100), prices, "y_skew_30_5d")

    rows.loc[269, "quote_date"] = pd.Timestamp("2025-01-02")
    with pytest.raises(ValueError, match="confirmation dates"):
        forecast_development(rows, prices, "y_skew_30_5d")


def test_end_of_development_targets_are_unavailable():
    dates = pd.bdate_range("2023-01-03", "2024-12-31")
    prices = spx_prices(dates)
    dataset = build_development_dataset(metric_rows(dates), prices)
    predictions = forecast_development(dataset, prices, "y_skew_30_5d")
    assert predictions.tail(5)["actual"].isna().all()
    assert predictions.tail(5)["m0_persistence"].eq(0).all()


def test_scoring_uses_identical_realised_dates_for_all_models():
    forecasts = pd.DataFrame({
        "quote_date": pd.bdate_range("2024-01-02", periods=4),
        "actual": [1.0, 2.0, -1.0, np.nan],
        "m0_persistence": [0.0, 0.0, 0.0, 0.0],
        "m1_mean": [0.5, 1.0, 0.0, 0.0],
        "m2_mean_reversion": [1.0, np.nan, -1.0, 0.0],
    })
    summary, common = score_development_forecasts(forecasts)
    assert common.quote_date.tolist() == [forecasts.loc[0, "quote_date"], forecasts.loc[2, "quote_date"]]
    assert summary.n_common.tolist() == [2, 2, 2]
    assert summary.loc[0, "rmse"] == pytest.approx(1.0)
    assert summary.loc[2, "rmse"] == pytest.approx(0.0)
    assert summary.loc[2, "r2_vs_persistence"] == pytest.approx(1.0)
    assert pd.isna(summary.loc[0, "correlation"])
    assert pd.isna(summary.loc[0, "directional_accuracy"])


def test_scoring_rejects_holdout_dates_and_no_common_dates():
    forecasts = pd.DataFrame({
        "quote_date": [pd.Timestamp("2025-01-02")],
        "actual": [1.0],
        "m0_persistence": [0.0],
        "m1_mean": [0.0],
        "m2_mean_reversion": [0.0],
    })
    with pytest.raises(ValueError, match="development forecast dates"):
        score_development_forecasts(forecasts)

    forecasts.loc[0, "quote_date"] = pd.Timestamp("2024-12-31")
    forecasts.loc[0, "m2_mean_reversion"] = np.nan
    with pytest.raises(ValueError, match="No common realised dates"):
        score_development_forecasts(forecasts)
