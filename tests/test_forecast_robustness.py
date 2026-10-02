import numpy as np
import pandas as pd
import pytest

from src.forecast_robustness import (
    HAC_LAG,
    build_robustness_outcomes,
    confirmation_top_one_percent_check,
    cumulative_m2_gain,
    extreme_move_check,
    fit_static_m2,
    forecast_static_m2,
    hac_loss_difference,
    headline_30d_predictor_dates,
    nonoverlapping_offsets,
    past_only_zscore,
    score_static_m2,
    vix_regime_check,
)
from src.forecasting import DEV_END, add_locked_zscores, fit_locked_models


@pytest.fixture
def sample():
    dates = pd.bdate_range("2023-01-03", "2025-08-29")
    step = np.arange(len(dates), dtype=float)
    panel = pd.DataFrame({
        "quote_date": dates,
        "session_index": np.arange(len(dates)),
        "skew_30": .3 + .04 * np.sin(step / 11) + .0002 * step,
        "skew_30_60": .1 + .03 * np.cos(step / 13),
        "rr25_30_60": .02 + .015 * np.sin(step / 17),
        "vix_close": 19 + 2 * np.sin(step / 19),
    })
    prices = pd.DataFrame({"secid": 108105, "ticker": "SPX", "date": dates, "close": 4000.0})
    return panel, prices


@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_exact_session_targets_and_unavailable_tail(sample, horizon):
    panel, prices = sample
    target = f"y_skew_30_{horizon}d"
    outcomes = build_robustness_outcomes(panel, prices)
    holdout = panel.loc[panel.quote_date > DEV_END].reset_index(drop=True)
    assert outcomes.loc[0, target] == pytest.approx(
        holdout.loc[horizon, "skew_30"] - holdout.loc[0, "skew_30"]
    )
    assert outcomes[target].tail(horizon).isna().all()
    assert outcomes.loc[0, "y_rr25_spread_5d"] == pytest.approx(
        holdout.loc[5, "rr25_30_60"] - holdout.loc[0, "rr25_30_60"]
    )
    absent = panel.drop(index=panel.index[-20])
    with pytest.raises(ValueError, match="every frozen SPX session"):
        build_robustness_outcomes(absent, prices)


def test_past_only_moments_ignore_current_future_and_count_valid():
    values = pd.Series(np.arange(82, dtype=float))
    values.loc[10] = np.nan
    z = past_only_zscore(values)
    assert z.iloc[:61].isna().all()
    prior = values.iloc[:61].dropna()
    assert z.iloc[61] == pytest.approx((values.iloc[61] - prior.mean()) / prior.std(ddof=1))
    changed = values.copy()
    changed.iloc[62:] = 1e5
    assert past_only_zscore(changed).iloc[61] == pytest.approx(z.iloc[61])
    changed = values.copy()
    changed.iloc[60] = 1e5
    assert past_only_zscore(changed).iloc[61] != pytest.approx(z.iloc[61])


def test_development_fit_is_static_and_excludes_unmatured_or_2025_values(sample):
    panel, prices = sample
    fitted = fit_static_m2(panel, prices, "skew_30", 5)
    assert fitted["last_matured_origin"] == panel.loc[panel.quote_date <= DEV_END, "quote_date"].iloc[-6]
    dev = panel.loc[panel.quote_date <= DEV_END, "skew_30"].reset_index(drop=True)
    dev_z = past_only_zscore(dev)
    dev_outcome = dev.shift(-5) - dev
    eligible = dev_z.notna() & dev_outcome.notna()
    assert fitted["extreme_cutoff"] == pytest.approx(
        np.quantile(dev_outcome.loc[eligible].abs(), .99)
    )
    prediction = forecast_static_m2(fitted, panel, prices)
    mutated = panel.copy()
    mutated.loc[mutated.quote_date > DEV_END, "skew_30"] += 100
    assert fit_static_m2(mutated, prices, "skew_30", 5) == fitted
    mutated = panel.copy()
    mutated.loc[mutated.quote_date > DEV_END, "y_skew_30_5d"] = 1e6
    assert fit_static_m2(mutated, prices, "skew_30", 5) == fitted
    pd.testing.assert_frame_equal(forecast_static_m2(fitted, mutated, prices), prediction)
    mutated = panel.copy()
    mutated.loc[mutated.quote_date <= DEV_END, "skew_30"] = np.nan
    with pytest.raises(ValueError, match="Insufficient matured"):
        fit_static_m2(mutated, prices, "skew_30", 5)


def test_future_state_does_not_change_earlier_forecast_or_bridge_missing_endpoint(sample):
    panel, prices = sample
    fitted = fit_static_m2(panel, prices, "skew_30", 5)
    original = forecast_static_m2(fitted, panel, prices)
    future = panel.copy()
    future.loc[future.index[-1], "skew_30"] = 1000
    assert forecast_static_m2(fitted, future, prices).loc[0, "m2_mean_reversion"] == pytest.approx(
        original.loc[0, "m2_mean_reversion"]
    )
    holdout_start = panel.index[panel.quote_date > DEV_END][0]
    missing = panel.copy()
    missing.loc[holdout_start + 5, "skew_30"] = np.nan
    targets = build_robustness_outcomes(missing, prices)
    assert pd.isna(targets.loc[0, "y_skew_30_5d"])
    assert pd.notna(targets.loc[1, "y_skew_30_5d"])
    assert pd.isna(targets.loc[5, "y_skew_30_5d"])


def test_five_day_m2_matches_frozen_headline_fit_and_predictions(sample):
    panel, prices = sample
    dev = panel.loc[panel.quote_date <= DEV_END].copy().reset_index(drop=True)
    dev = add_locked_zscores(dev)
    dev["y_skew_30_5d"] = dev.skew_30.shift(-5) - dev.skew_30
    dev["y_skew_spread_5d"] = dev.skew_30_60.shift(-5) - dev.skew_30_60
    # The frozen fitter also needs the fixed M6 universe; synthetic features suffice.
    from src.forecasting import linear_feature_sets
    for name in linear_feature_sets("y_skew_30_5d")["m5b_vix_state"]:
        if name not in dev:
            dev[name] = np.sin(np.arange(len(dev)) / (3 + len(name)))
    frozen = fit_locked_models(dev, prices)
    for state, target in (("skew_30", "y_skew_30_5d"),
                          ("skew_30_60", "y_skew_spread_5d")):
        fitted = fit_static_m2(panel, prices, state, 5)
        assert fitted["intercept"] == pytest.approx(frozen["m2_coefficients"][target]["intercept"])
        assert fitted["slope"] == pytest.approx(frozen["m2_coefficients"][target]["slope"])
        prediction = forecast_static_m2(fitted, panel, prices)
        full_z = past_only_zscore(panel[state])
        expected = fitted["intercept"] + fitted["slope"] * full_z.loc[panel.quote_date > DEV_END]
        np.testing.assert_allclose(prediction.m2_mean_reversion, expected)


def test_headline_mask_and_common_scoring(sample):
    panel, prices = sample
    from src.forecasting import linear_feature_sets
    for name in linear_feature_sets("y_skew_30_5d")["m5b_vix_state"]:
        if name not in panel:
            panel[name] = .1
    panel["z_skew_30"] = past_only_zscore(panel.skew_30)
    dates = headline_30d_predictor_dates(panel, prices)
    assert len(dates) == (panel.quote_date > DEV_END).sum()
    panel.loc[panel.quote_date == dates[2], "vix_close"] = np.nan
    assert dates[2] not in headline_30d_predictor_dates(panel, prices)
    fitted = fit_static_m2(panel, prices, "skew_30", 5)
    forecasts = forecast_static_m2(fitted, panel, prices)
    outcomes = build_robustness_outcomes(panel, prices)
    summary, common = score_static_m2(forecasts, outcomes, fitted["target"])
    assert summary["n"] == len(forecasts) - 5
    assert summary["r2_vs_m0"] == pytest.approx(
        1 - np.square(common.actual - common.m2_mean_reversion).sum()
        / np.square(common.actual).sum()
    )
    assert len(cumulative_m2_gain(common)) == len(common)


def test_offsets_use_original_session_indices_and_fixed_extreme_cutoff():
    common = pd.DataFrame({
        "quote_date": pd.bdate_range("2025-01-02", periods=10),
        "session_index": [501, 503, 504, 506, 508, 509, 511, 512, 514, 516],
        "actual": [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
        "m0_persistence": 0., "m2_mean_reversion": 1.,
    })
    offsets = nonoverlapping_offsets(common)
    assert offsets.loc[offsets.offset == 1, "n"].iloc[0] == 4
    trimmed = extreme_move_check(common, 8.5)
    assert trimmed["removed"] == 2
    assert trimmed["n"] == 8
    assert trimmed["development_cutoff"] == 8.5


def test_confirmation_top_one_percent_removes_exact_k_with_date_tie_break():
    dates = pd.bdate_range("2025-01-02", periods=158)
    common = pd.DataFrame({
        "quote_date": dates,
        "actual": np.r_[np.ones(155), -10., 10., -10.],
        "m0_persistence": 0.,
        "m2_mean_reversion": .5,
    }).sample(frac=1, random_state=17).reset_index(drop=True)
    result = confirmation_top_one_percent_check(common)
    assert result["removed"] == 2
    assert result["n"] == 156
    assert result["removed_dates"] == (dates[155], dates[156])
    expected = common.loc[~common.quote_date.isin(result["removed_dates"])]
    assert result["rmse_m0"] == pytest.approx(np.sqrt(np.square(expected.actual).mean()))
    assert result["mae_m2"] == pytest.approx(np.abs(expected.actual - .5).mean())
    assert result["r2_vs_m0"] == pytest.approx(
        1 - np.square(expected.actual - .5).sum() / np.square(expected.actual).sum()
    )


def test_vix_regimes_reuse_predictions_without_refit(monkeypatch):
    import src.forecast_robustness as robustness
    monkeypatch.setattr(robustness, "fit_static_m2", lambda *args: pytest.fail("refit"))
    common = pd.DataFrame({
        "quote_date": pd.bdate_range("2025-01-02", periods=3),
        "actual": [1., 1., 1.], "m0_persistence": 0., "m2_mean_reversion": .5,
    })
    panel = pd.DataFrame({"quote_date": common.quote_date, "vix_close": [19.9, 20., np.nan]})
    result = vix_regime_check(common, panel)
    assert result.n.tolist() == [1, 1]
    assert result.small_sample.tolist() == [True, True]


def test_hac_is_fixed_lag_four_and_matches_manual_bartlett_formula():
    actual = np.array([1., -2., 2., -1., 3., -3., 2., 1., -1., 2.])
    common = pd.DataFrame({"actual": actual, "m0_persistence": 0.,
                           "m2_mean_reversion": np.full(len(actual), .5)})
    result = hac_loss_difference(common)
    d = actual**2 - (actual - .5)**2
    centered = d - d.mean()
    lrv = np.dot(centered, centered) / len(d)
    for lag in range(1, 5):
        lrv += 2 * (1 - lag / 5) * np.dot(centered[lag:], centered[:-lag]) / len(d)
    assert HAC_LAG == result["lag"] == 4
    assert result["mean_loss_difference"] == pytest.approx(d.mean())
    assert result["hac_standard_error"] == pytest.approx(np.sqrt(max(lrv, 0) / len(d)))
    assert result["ci_lower"] == pytest.approx(d.mean() - 1.96 * result["hac_standard_error"])
