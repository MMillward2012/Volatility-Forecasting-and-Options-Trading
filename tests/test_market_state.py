import numpy as np
import pandas as pd
import pytest

from src.market_state import build_market_state, load_market_state, merge_market_state


def prices_fixture(periods=25):
    dates = pd.bdate_range("2022-12-01", periods=periods)
    step = np.arange(periods, dtype=float)
    spx = pd.DataFrame({
        "secid": 108105,
        "date": dates,
        "close": 4000 * np.exp(0.001 * step),
        # Deliberately unrelated values prove returns are calculated from close.
        "return": np.full(periods, 0.25),
    })
    vix = pd.DataFrame({"date": dates[::-1], "close": 20 + step[::-1]})
    return spx, vix


def test_log_return_uses_spx_close_not_vendor_return():
    spx, vix = prices_fixture()
    state = build_market_state(spx, vix)

    assert np.isnan(state.loc[0, "spx_return"])
    assert state.loc[1, "spx_return"] == pytest.approx(np.log(spx.loc[1, "close"] / spx.loc[0, "close"]))
    assert state.loc[1, "spx_return"] != state.loc[1, "spx_vendor_return"]
    assert state.loc[1, "spx_abs_return"] == pytest.approx(abs(state.loc[1, "spx_return"]))
    assert state.loc[1, "spx_return_sq"] == pytest.approx(state.loc[1, "spx_return"] ** 2)


def test_rv5_and_rv20_use_requested_annualized_sums():
    spx, vix = prices_fixture()
    state = build_market_state(spx, vix)
    returns = np.log(spx.close / spx.close.shift(1))

    expected_rv5 = np.sqrt(252 / 5 * np.sum(returns.iloc[1:6] ** 2))
    expected_rv20 = np.sqrt(252 / 20 * np.sum(returns.iloc[1:21] ** 2))
    assert np.isnan(state.loc[4, "rv_5"])
    assert state.loc[5, "rv_5"] == pytest.approx(expected_rv5)
    assert np.isnan(state.loc[19, "rv_20"])
    assert state.loc[20, "rv_20"] == pytest.approx(expected_rv20)


def test_rolling_returns_do_not_bridge_missing_spx_close():
    spx, vix = prices_fixture()
    missing_date = spx.loc[10, "date"]
    spx = spx.loc[spx.date.ne(missing_date)]

    state = build_market_state(spx, vix)

    assert state.loc[10, "date"] == missing_date
    assert np.isnan(state.loc[10, "spx_close"])
    assert np.isnan(state.loc[10, "spx_return"])
    assert np.isnan(state.loc[11, "spx_return"])
    assert np.isnan(state.loc[14, "rv_5"])
    assert np.isfinite(state.loc[16, "rv_5"])
    assert np.isnan(state.loc[24, "rv_20"])


def test_vix_is_joined_by_date_and_is_never_forward_filled():
    spx, vix = prices_fixture()
    vix = vix.loc[vix.date.ne(spx.loc[5, "date"])]

    state = build_market_state(spx, vix)

    assert state.loc[0, "vix_close"] == pytest.approx(20)
    assert np.isnan(state.loc[5, "vix_close"])
    assert state.loc[6, "vix_close"] == pytest.approx(26)


def test_market_state_loader_reads_raw_price_csvs(tmp_path):
    spx, vix = prices_fixture()
    spx_path, vix_path = tmp_path / "spx.csv", tmp_path / "vix.csv"
    spx.to_csv(spx_path, index=False)
    vix.to_csv(vix_path, index=False)

    state = load_market_state(spx_path, vix_path)

    assert len(state) == len(spx)
    assert state.date.is_monotonic_increasing


def test_merge_keeps_surface_dates_and_drops_warmup_only_dates():
    spx, vix = prices_fixture()
    market = build_market_state(spx, vix)
    surface_dates = spx.date.iloc[20:23].reset_index(drop=True)
    surface = pd.DataFrame({"quote_date": surface_dates, "skew_30": [1.0, 2.0, 3.0]})

    merged = merge_market_state(surface, market)

    assert merged.date.tolist() == surface_dates.tolist()
    assert len(merged) == len(surface)
    assert merged.skew_30.tolist() == [1.0, 2.0, 3.0]
    assert merged.rv_20.notna().all()
    assert not merged.date.isin(spx.date.iloc[:20]).any()
