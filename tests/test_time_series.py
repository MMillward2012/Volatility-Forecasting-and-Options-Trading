import numpy as np
import pandas as pd
import pytest

from src.time_series import add_forward_targets, build_daily_time_series, load_skew_metrics


def long_metrics():
    rows = []
    values = {
        "2025-01-02": {
            30: (0.3, 0.31, 0.2), 60: (0.2, 0.21, 0.19), 90: (0.1, 0.11, 0.18),
        },
        "2025-01-03": {
            30: (0.35, 0.32, 0.21), 60: (0.22, 0.22, 0.20), 90: (0.12, 0.12, 0.19),
        },
        "2025-01-06": {
            30: (0.34, 0.33, 0.22), 60: (0.23, 0.23, 0.21), 90: (0.13, 0.13, 0.20),
        },
    }
    for date, tenors in values.items():
        for tenor, (skew, rr25, atm_iv) in tenors.items():
            rows.append({
                "quote_date": date, "target_days": tenor,
                "atm_skew_slope": skew, "slope_valid": True,
                "rr25_downside": rr25, "rr25_valid": True,
                "atm_iv": atm_iv, "atm_valid": True,
            })
    return pd.DataFrame(rows)


def test_long_rows_map_to_sorted_wide_daily_variables():
    result = build_daily_time_series(long_metrics())

    assert result.quote_date.tolist() == list(pd.to_datetime([
        "2025-01-02", "2025-01-03", "2025-01-06",
    ]))
    assert result.loc[0, ["skew_30", "skew_60", "skew_90"]].tolist() == [0.3, 0.2, 0.1]
    assert result.loc[1, ["rr25_30", "rr25_60", "rr25_90"]].tolist() == [0.32, 0.22, 0.12]
    assert result.loc[0, ["atm_iv_30", "atm_iv_60", "atm_iv_90"]].tolist() == [0.2, 0.19, 0.18]


def test_qc_flags_mask_only_their_own_metric():
    source = long_metrics()
    source.loc[(source.quote_date == "2025-01-02") & (source.target_days == 30), "slope_valid"] = False
    source.loc[(source.quote_date == "2025-01-02") & (source.target_days == 60), "rr25_valid"] = False
    source.loc[(source.quote_date == "2025-01-02") & (source.target_days == 90), "atm_valid"] = False

    result = build_daily_time_series(source).iloc[0]

    assert np.isnan(result.skew_30) and result.rr25_30 == 0.31 and result.atm_iv_30 == 0.2
    assert result.skew_60 == 0.2 and np.isnan(result.rr25_60) and result.atm_iv_60 == 0.19
    assert result.skew_90 == 0.1 and result.rr25_90 == 0.11 and np.isnan(result.atm_iv_90)


def test_nonfinite_and_unparseable_metric_values_become_missing():
    source = long_metrics()
    source.loc[0, "atm_skew_slope"] = np.inf
    source["rr25_downside"] = source["rr25_downside"].astype(object)
    source.loc[1, "rr25_downside"] = "not a number"
    result = build_daily_time_series(source)
    assert np.isnan(result.loc[0, "skew_30"])
    assert np.isnan(result.loc[0, "d_skew_30"])
    assert np.isnan(result.loc[0, "rr25_60"])


def test_daily_changes_and_term_structure_differences():
    result = build_daily_time_series(long_metrics())

    assert np.isnan(result.loc[0, "d_skew_30"])
    assert result.loc[1, "d_skew_30"] == pytest.approx(0.05)
    assert result.loc[2, "d_skew_30"] == pytest.approx(-0.01)
    assert result.loc[1, "d_rr25_60"] == pytest.approx(0.01)
    assert result.loc[1, "d_atm_iv_90"] == pytest.approx(0.01)
    assert result.loc[0, "skew_30_60"] == pytest.approx(0.1)
    assert result.loc[0, "skew_60_90"] == pytest.approx(0.1)
    assert result.loc[0, "rr25_30_60"] == pytest.approx(0.1)
    assert result.loc[0, "atm_iv_60_90"] == pytest.approx(0.01)


def test_missing_date_observation_prevents_change_across_gap():
    source = long_metrics()
    source = source.loc[~(
        (source.quote_date == "2025-01-03") & (source.target_days == 30)
    )]
    result = build_daily_time_series(source)

    assert result.quote_date.tolist() == list(pd.to_datetime([
        "2025-01-02", "2025-01-03", "2025-01-06",
    ]))
    assert result.loc[0, "skew_30"] == pytest.approx(0.3)
    assert np.isnan(result.loc[1, "skew_30"])
    assert np.isnan(result.loc[1, "d_skew_30"])
    assert result.loc[2, "skew_30"] == pytest.approx(0.34)
    assert np.isnan(result.loc[2, "d_skew_30"])
    assert np.isnan(result.loc[1, "skew_30_60"])


def test_duplicate_date_tenor_rows_raise_instead_of_averaging():
    source = long_metrics()
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        build_daily_time_series(source)


def test_empty_tenor_date_is_retained_with_missing_values():
    source = long_metrics().loc[lambda rows: ~(
        (rows.quote_date == "2025-01-03") & (rows.target_days == 60)
    )]
    result = build_daily_time_series(source)
    row = result.loc[result.quote_date.eq(pd.Timestamp("2025-01-03"))].iloc[0]
    assert np.isnan(row.skew_60)
    assert np.isnan(row.rr25_60)
    assert np.isnan(row.atm_iv_60)


def test_input_dataframe_is_not_modified():
    source = long_metrics()
    original = source.copy(deep=True)
    build_daily_time_series(source)
    pd.testing.assert_frame_equal(source, original)


def test_csv_loader_reads_quote_dates(tmp_path):
    path = tmp_path / "metrics.csv"
    long_metrics().to_csv(path, index=False)
    loaded = load_skew_metrics(path)
    assert pd.api.types.is_datetime64_any_dtype(loaded["quote_date"])
    assert len(loaded) == 9


def daily_target_fixture(periods=12):
    dates = pd.bdate_range("2025-01-02", periods=periods)
    step = np.arange(periods, dtype=float)
    daily = pd.DataFrame({
        "quote_date": dates,
        "skew_30": 0.30 + 0.10 * step,
        "skew_60": 0.20 + 0.04 * step,
        "skew_30_60": 0.10 + 0.06 * step,
        "rr25_30": 0.25 + 0.08 * step,
        "rr25_60": 0.15 + 0.03 * step,
        "rr25_30_60": 0.10 + 0.05 * step,
    })
    return daily, dates


def test_forward_targets_use_fixed_one_five_and_ten_session_horizons():
    daily, dates = daily_target_fixture()

    result = add_forward_targets(daily, dates)

    assert result.quote_date.tolist() == dates.tolist()
    for horizon in (1, 5, 10):
        assert result.loc[0, f"y_skew_spread_{horizon}d"] == pytest.approx(0.06 * horizon)
        assert result.loc[0, f"y_skew_30_{horizon}d"] == pytest.approx(0.10 * horizon)
        assert result.loc[0, f"y_rr25_spread_{horizon}d"] == pytest.approx(0.05 * horizon)


def test_missing_endpoint_makes_only_dependent_targets_missing():
    daily, dates = daily_target_fixture()
    daily.loc[1, "skew_30"] = np.nan
    daily.loc[1, "skew_30_60"] = np.nan
    daily.loc[1, "rr25_30_60"] = np.nan

    result = add_forward_targets(daily, dates)

    assert np.isnan(result.loc[0, "y_skew_spread_1d"])
    assert np.isnan(result.loc[0, "y_skew_30_1d"])
    assert np.isnan(result.loc[0, "y_rr25_spread_1d"])
    assert np.isfinite(result.loc[0, "y_skew_spread_5d"])
    assert np.isfinite(result.loc[0, "y_skew_30_5d"])
    assert np.isfinite(result.loc[0, "y_rr25_spread_5d"])


def test_missing_intermediate_metric_does_not_break_valid_endpoint_target():
    daily, dates = daily_target_fixture()
    daily.loc[2, ["skew_30", "skew_30_60", "rr25_30_60"]] = np.nan

    result = add_forward_targets(daily, dates)

    assert result.loc[0, "y_skew_spread_5d"] == pytest.approx(0.30)
    assert result.loc[0, "y_skew_30_5d"] == pytest.approx(0.50)
    assert result.loc[0, "y_rr25_spread_5d"] == pytest.approx(0.25)


def test_final_rows_without_future_endpoints_are_missing():
    daily, dates = daily_target_fixture()

    result = add_forward_targets(daily, dates)

    for horizon in (1, 5, 10):
        columns = [
            f"y_skew_spread_{horizon}d", f"y_skew_30_{horizon}d",
            f"y_rr25_spread_{horizon}d",
        ]
        assert result.tail(horizon)[columns].isna().all().all()


def test_absent_trading_date_is_inserted_and_does_not_shorten_horizon():
    daily, dates = daily_target_fixture()
    daily = daily.drop(index=2).reset_index(drop=True)

    result = add_forward_targets(daily, dates)

    assert result.quote_date.tolist() == dates.tolist()
    assert result.loc[2, "quote_date"] == dates[2]
    assert np.isnan(result.loc[2, "skew_30"])
    assert np.isnan(result.loc[1, "y_skew_30_1d"])
    assert result.loc[1, "y_skew_30_5d"] == pytest.approx(0.50)
