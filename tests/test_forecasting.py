import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    add_development_zscores,
    build_development_dataset,
    build_linear_development_dataset,
    forecast_development,
    forecast_linear_development,
    linear_feature_sets,
    score_development_forecasts,
    score_linear_development_forecasts,
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


def linear_rows(periods=270):
    dates = pd.bdate_range("2023-01-03", periods=periods)
    step = np.arange(periods, dtype=float)
    state = 0.3 + step / 1000 + 0.05 * np.sin(step * 0.13)
    spread = 0.1 + step / 2000 + 0.04 * np.cos(step * 0.09)
    market_return = 0.01 * np.sin(step * 0.31)
    rows = pd.DataFrame({
        "quote_date": dates,
        "session_index": np.arange(periods),
        "skew_30": state,
        "skew_30_60": spread,
        "skew_60": 0.2 + 0.03 * np.sin(step * 0.07),
        "skew_60_90": 0.03 + 0.02 * np.cos(step * 0.11),
        "atm_iv_30": 0.15 + 0.01 * np.sin(step * 0.17),
        "atm_iv_30_60": 0.01 + 0.005 * np.cos(step * 0.19),
        "atm_iv_60_90": 0.01 + 0.004 * np.sin(step * 0.23),
        "spx_return": market_return,
        "spx_abs_return": np.abs(market_return),
        "spx_return_sq": market_return**2,
        "rv_5": 0.12 + 0.01 * np.sin(step * 0.29),
        "rv_20": 0.13 + 0.01 * np.cos(step * 0.37),
        "vix_close": 18 + np.sin(step * 0.41),
    })
    rows = add_development_zscores(rows)
    rows["d_skew_30"] = rows["skew_30"].diff()
    rows["d_skew_30_60"] = rows["skew_30_60"].diff()
    rows["y_skew_30_5d"] = (
        0.02 + 0.03 * rows["z_skew_30"] - 0.4 * rows["d_skew_30"]
        + 0.1 * rows["atm_iv_30"]
    )
    rows["y_skew_spread_5d"] = (
        -0.01 + 0.02 * rows["z_skew_30_60"]
        - 0.3 * rows["d_skew_30_60"] + 0.1 * rows["atm_iv_30_60"]
    )
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


def test_linear_feature_sets_match_frozen_protocol_exactly():
    surface = (
        "skew_60", "skew_30_60", "skew_60_90",
        "atm_iv_30", "atm_iv_30_60", "atm_iv_60_90",
    )
    realized = (
        "spx_return", "spx_abs_return", "spx_return_sq", "rv_5", "rv_20",
    )
    for target, own in (
        ("y_skew_30_5d", ("z_skew_30", "d_skew_30")),
        ("y_skew_spread_5d", ("z_skew_30_60", "d_skew_30_60")),
    ):
        sets = linear_feature_sets(target)
        assert list(sets) == [
            "m3_own_dynamics", "m4_surface_state", "m5a_realized_state", "m5b_vix_state",
        ]
        assert sets["m3_own_dynamics"] == own
        assert sets["m4_surface_state"] == own + surface
        assert sets["m5a_realized_state"] == own + surface + realized
        assert sets["m5b_vix_state"] == own + surface + realized + ("vix_close",)
        assert "spx_vendor_return" not in sets["m5b_vix_state"]


def test_linear_dataset_uses_exact_date_market_join_and_missing_spread_sessions():
    dates = pd.bdate_range("2023-01-03", periods=35)
    metrics = metric_rows(dates).loc[lambda rows: rows.quote_date.ne(dates[2])]
    prices = spx_prices(dates)
    prices["close"] = 4000 + np.arange(len(dates)) * 10
    prices["return"] = 999.0
    vix = pd.DataFrame({
        "ticker": "VIX", "date": dates.delete(5), "close": 20 + np.arange(len(dates) - 1),
    })
    result = build_linear_development_dataset(metrics, prices, vix)
    assert result.quote_date.tolist() == dates.tolist()
    assert pd.isna(result.loc[2, "d_skew_30_60"])
    assert pd.isna(result.loc[3, "d_skew_30_60"])
    assert result.loc[4, "spx_return"] == pytest.approx(np.log(prices.loc[4, "close"] / prices.loc[3, "close"]))
    assert pd.isna(result.loc[5, "vix_close"])
    assert result.loc[6, "vix_close"] == vix.loc[vix.date.eq(dates[6]), "close"].iloc[0]
    assert "spx_vendor_return" not in result


def test_2025_source_rows_cannot_change_linear_development_dataset():
    dates = pd.bdate_range("2024-12-18", "2025-01-10")
    metrics = metric_rows(dates)
    prices = spx_prices(dates)
    vix = pd.DataFrame({"ticker": "VIX", "date": dates, "close": 20.0})
    expected = build_linear_development_dataset(metrics, prices, vix)
    future = dates > pd.Timestamp("2024-12-31")
    metrics.loc[metrics.quote_date > pd.Timestamp("2024-12-31"), "atm_skew_slope"] = 999
    prices.loc[future, "close"] = 9999
    vix.loc[future, "close"] = 999
    actual = build_linear_development_dataset(metrics, prices, vix)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.quote_date.max() <= pd.Timestamp("2024-12-31")


def test_linear_dataset_rejects_missing_dates_before_filtering():
    dates = pd.bdate_range("2023-01-03", periods=10)
    vix = pd.DataFrame({"ticker": "VIX", "date": dates, "close": 20.0})
    vix.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError, match="dates cannot be missing"):
        build_linear_development_dataset(metric_rows(dates), spx_prices(dates), vix)


@pytest.mark.parametrize("target", ["y_skew_30_5d", "y_skew_spread_5d"])
def test_linear_models_use_current_change_and_purged_targets(target):
    rows, prices = linear_rows()
    baseline = forecast_development(rows, prices, target)
    result, coefficients = forecast_linear_development(rows, prices, target)
    pd.testing.assert_frame_equal(result[baseline.columns], baseline)
    first = result.iloc[0]
    assert first.session_index == 252
    for model in linear_feature_sets(target):
        assert np.isfinite(first[model])
        fit = coefficients.loc[
            coefficients.session_index.eq(252) & coefficients.model.eq(model)
        ]
        assert fit.n_train.iloc[0] == 188
    changed = rows.copy()
    changed.loc[248, target] = 1000
    changed.loc[253:, ["skew_30", "skew_30_60", "vix_close"]] = 1000
    future, _ = forecast_linear_development(changed, prices, target)
    for model in linear_feature_sets(target):
        assert future.loc[0, model] == pytest.approx(first[model])

    matured = rows.copy()
    matured.loc[247, target] = 1000
    eligible, _ = forecast_linear_development(matured, prices, target)
    assert eligible.loc[0, "m3_own_dynamics"] != pytest.approx(first.m3_own_dynamics)


def test_m4_ols_recovers_declared_linear_signal():
    rows, prices = linear_rows()
    predictions, coefficients = forecast_linear_development(rows, prices, "y_skew_30_5d")
    assert predictions.loc[0, "m4_surface_state"] == pytest.approx(rows.loc[252, "y_skew_30_5d"])
    fitted = coefficients.loc[
        coefficients.session_index.eq(252) & coefficients.model.eq("m4_surface_state")
    ].set_index("feature")["coefficient"]
    assert fitted["intercept"] == pytest.approx(0.02)
    assert fitted["z_skew_30"] == pytest.approx(0.03)
    assert fitted["d_skew_30"] == pytest.approx(-0.4)
    assert fitted["atm_iv_30"] == pytest.approx(0.1)


@pytest.mark.parametrize("target,prior_column", [
    ("y_skew_30_5d", "skew_30"),
    ("y_skew_spread_5d", "skew_30_60"),
])
def test_missing_prior_state_prevents_m3_change_without_erasing_m2(target, prior_column):
    rows, prices = linear_rows()
    rows.loc[251, prior_column] = np.nan
    result, _ = forecast_linear_development(rows, prices, target)
    assert np.isfinite(result.loc[0, "m2_mean_reversion"])
    assert pd.isna(result.loc[0, "m3_own_dynamics"])


def test_missing_surface_and_vix_predictors_only_abstain_downstream():
    rows, prices = linear_rows()
    rows.loc[252, "atm_iv_30"] = np.nan
    rows.loc[253, "vix_close"] = np.nan
    result, _ = forecast_linear_development(rows, prices, "y_skew_30_5d")
    assert np.isfinite(result.loc[0, "m3_own_dynamics"])
    assert result.loc[0, ["m4_surface_state", "m5a_realized_state", "m5b_vix_state"]].isna().all()
    assert np.isfinite(result.loc[1, "m5a_realized_state"])
    assert pd.isna(result.loc[1, "m5b_vix_state"])


def test_linear_forecast_rejects_confirmation_rows():
    rows, prices = linear_rows()
    rows.loc[269, "quote_date"] = pd.Timestamp("2025-01-02")
    with pytest.raises(ValueError, match="confirmation dates"):
        forecast_linear_development(rows, prices, "y_skew_30_5d")


def test_linear_scoring_uses_strict_common_dates_and_two_benchmark_r2s():
    dates = pd.bdate_range("2024-01-02", periods=3)
    forecasts = pd.DataFrame({
        "quote_date": dates,
        "actual": [2.0, -1.0, 10.0],
        "m0_persistence": [0.0, 0.0, 0.0],
        "m1_mean": [0.0, 0.0, 0.0],
        "m2_mean_reversion": [1.0, 0.0, 0.0],
        "m3_own_dynamics": [2.0, -1.0, 0.0],
        "m4_surface_state": [1.0, -1.0, 0.0],
        "m5a_realized_state": [1.0, 0.0, 0.0],
        "m5b_vix_state": [2.0, -1.0, np.nan],
    })
    summary, common = score_linear_development_forecasts(forecasts)
    assert common.quote_date.tolist() == dates[:2].tolist()
    assert summary.n_common.eq(2).all()
    m3 = summary.set_index("model").loc["m3_own_dynamics"]
    assert m3.r2_vs_persistence == pytest.approx(1.0)
    assert m3.r2_vs_mean_reversion == pytest.approx(1.0)
    m4 = summary.set_index("model").loc["m4_surface_state"]
    assert m4.r2_vs_persistence == pytest.approx(0.8)
    assert m4.r2_vs_mean_reversion == pytest.approx(0.5)
    assert pd.isna(summary.set_index("model").loc["m2_mean_reversion", "r2_vs_mean_reversion"])
    assert pd.isna(summary.set_index("model").loc["m0_persistence", "directional_accuracy"])
