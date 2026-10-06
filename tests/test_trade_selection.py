import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.forecast_robustness import past_only_zscore
from src.historical_pipeline import CONSTANTS
from src.pricing import black_scholes_call_price, black_scholes_put_price, black_scholes_greeks
from src.trade_selection import (
    fit_rr_spread_signal, forecast_rr_spread_signal, prepare_executable_chain,
    select_delta_contract, select_expiry_pair, select_rr_spread,
)


def synthetic_chain(date="2023-01-03", dtes=(30, 60), deltas=(.20, .25, .30)):
    date = pd.Timestamp(date)
    rows = []
    for dte in dtes:
        tau = dte / 365
        for option_type in ("put", "call"):
            for delta in deltas:
                d1 = norm.ppf(delta if option_type == "call" else 1 - delta)
                strike = 100 * np.exp(.5 * .2**2 * tau - d1 * .2 * np.sqrt(tau))
                price_fn = black_scholes_call_price if option_type == "call" else black_scholes_put_price
                mid = float(price_fn(100, strike, .99, tau, .2))
                greeks = black_scholes_greeks(100, strike, .99, tau, .2, option_type, 100)
                rows.append({"security_id": 108105, "quote_date": date, "expiry_date": date + pd.Timedelta(days=dte),
                             "option_id": len(rows) + 1, "symbol": "synthetic", "option_type": option_type,
                             "strike": strike, "days_to_expiry": dte, "time_to_expiry": tau,
                             "best_bid": mid - .01, "best_ask": mid + .01, "mid_price": mid,
                             "forward": 100., "discount_factor": .99, "mid_iv": .2, "is_otm": True,
                             "forward_delta": greeks["forward_delta"],
                             "option_delta": 100 * greeks["spot_delta"], "vega": 100 * greeks["vega"]})
    return pd.DataFrame(rows)


def synthetic_prices(dates):
    return pd.DataFrame({"date": dates, "close": 100., "ticker": "SPX", "secid": 108105})


def test_expiry_pair_minimises_cost_and_is_distinct():
    chain = synthetic_chain(dtes=(24, 29, 32, 56, 61, 65))
    near, far = select_expiry_pair(chain)
    assert (near - chain.quote_date.iloc[0]).days == 29
    assert (far - chain.quote_date.iloc[0]).days == 61
    assert near != far
    assert select_expiry_pair(synthetic_chain(dtes=(30,))) is None


@pytest.mark.parametrize("dtes", [(22, 60), (30, 68)])
def test_expiry_tolerance_not_relaxed(dtes):
    assert select_expiry_pair(synthetic_chain(dtes=dtes)) is None


def test_inclusive_tolerances_and_deterministic_ties():
    chain = synthetic_chain(dtes=(23, 67), deltas=(.20,))
    assert select_rr_spread(chain, 1)["status"] == "available"
    chain = synthetic_chain(dtes=(29, 31, 59, 61))
    near, far = select_expiry_pair(chain.sample(frac=1, random_state=7))
    assert (near - chain.quote_date.iloc[0]).days == 29
    assert (far - chain.quote_date.iloc[0]).days == 59


def test_delta_wings_direction_and_unit_vega():
    chain = synthetic_chain()
    positive = select_rr_spread(chain, 1)
    negative = select_rr_spread(chain, -3)
    legs = positive["legs"]
    np.testing.assert_allclose(legs.forward_delta, [-.25, .25, -.25, .25])
    np.testing.assert_array_equal(np.sign(legs.quantity), [1, -1, -1, 1])
    np.testing.assert_allclose(legs.quantity, -negative["legs"].quantity)
    np.testing.assert_allclose(legs.quantity, select_rr_spread(chain, 100)["legs"].quantity)
    np.testing.assert_allclose(legs.gross_vega, .25)
    assert positive["gross_vega"] == pytest.approx(1)
    assert positive["net_vega"] == pytest.approx(0)
    assert positive["initial_hedge"] == pytest.approx(-positive["portfolio_delta"])


def test_delta_distance_failure_does_not_switch_expiry():
    chain = synthetic_chain(dtes=(30, 31, 60))
    date = chain.quote_date.iloc[0]
    chain.loc[chain.expiry_date.eq(date + pd.Timedelta(days=30)), "forward_delta"] = .5
    selection = select_rr_spread(chain, 1)
    assert selection["reason"] == "missing_25_delta_wing"
    assert selection["expiry_pair"][0] == date + pd.Timedelta(days=30)
    assert not all(check["eligible"] for check in selection["wing_checks"])


@pytest.mark.parametrize("bid,ask", [(0, 1), (2, 1), (np.nan, 1), (1, np.inf)])
def test_invalid_executable_quotes_rejected(bid, ask):
    chain = synthetic_chain()
    chain.loc[(chain.forward_delta - .25).abs() < 1e-12, ["best_bid", "best_ask"]] = [bid, ask]
    row = select_delta_contract(chain, chain.expiry_date.iloc[0], "call")
    assert abs(row.forward_delta - .25) > 1e-12


def test_future_columns_and_vendor_delta_do_not_affect_selection():
    chain = synthetic_chain()
    expected = select_rr_spread(chain, 1)["legs"]
    chain["future_exit_bid"] = np.nan
    chain["vendor_delta"] = 99
    actual = select_rr_spread(chain.sample(frac=1, random_state=3), 1)["legs"]
    np.testing.assert_array_equal(actual.option_id, expected.option_id)
    np.testing.assert_allclose(actual.quantity, expected.quantity)


def test_selection_rejects_mixed_quote_dates_and_duplicate_ids():
    chain = synthetic_chain()
    changed = chain.copy()
    changed.loc[0, "quote_date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="one quote date"):
        select_rr_spread(changed, 1)
    with pytest.raises(ValueError, match="unique option IDs"):
        select_rr_spread(pd.concat([chain, chain.iloc[:1]]), 1)


@pytest.mark.parametrize("forecast", [0, np.nan, np.inf])
def test_no_signal(forecast):
    assert select_rr_spread(synthetic_chain(), forecast)["reason"] == "no_signal"


def test_raw_preparation_reuses_parity_and_iv_and_filters_am():
    date = pd.Timestamp("2023-01-03")
    raw = []
    for dte in (30, 60):
        for strike in (90., 95., 100., 105., 110.):
            for flag, fn in (("C", black_scholes_call_price), ("P", black_scholes_put_price)):
                mid = float(fn(100, strike, .99, dte / 365, .2))
                raw.append({**CONSTANTS, "date": date, "exdate": date + pd.Timedelta(days=dte),
                            "optionid": len(raw) + 1, "symbol": "test", "cp_flag": flag,
                            "strike_price": strike * 1000, "best_bid": mid * .99, "best_offer": mid * 1.01,
                            "volume": 0, "open_interest": 0, "am_settlement": 0,
                            "expiry_indicator": "w", "impl_volatility": np.nan,
                            "delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan})
    raw = pd.DataFrame(raw)
    am = raw.iloc[:1].assign(am_settlement=1, optionid=999)
    result = prepare_executable_chain(pd.concat([raw, am]), 100)
    chain = result["chain"]
    assert 999 not in chain.option_id.to_list()
    np.testing.assert_allclose(chain.forward, 100)
    np.testing.assert_allclose(chain.discount_factor, .99)
    np.testing.assert_allclose(chain.mid_iv, .2, atol=1e-7)
    assert result["diagnostics"]["pm_rows"] == 20


def test_signal_is_development_matured_and_causal():
    dates = pd.bdate_range("2023-01-03", "2025-02-28")
    prices = synthetic_prices(dates)
    state = .02 + .005 * np.sin(np.arange(len(dates)) / 9)
    daily = pd.DataFrame({"quote_date": dates, "rr25_30_60": state})
    fitted = fit_rr_spread_signal(daily, prices)
    development = daily.loc[daily.quote_date <= "2024-12-31"]
    z = past_only_zscore(development.rr25_30_60)
    y = development.rr25_30_60.shift(-5) - development.rr25_30_60
    valid = z.notna() & y.notna()
    expected = np.linalg.lstsq(np.column_stack([np.ones(valid.sum()), z[valid]]), y[valid], rcond=None)[0]
    np.testing.assert_allclose([fitted["intercept"], fitted["slope"]], expected)
    assert fitted["last_matured_origin"] == development.quote_date.iloc[-6]
    before = forecast_rr_spread_signal(fitted, daily, prices)
    assert before.quote_date.min() > pd.Timestamp("2024-12-31")
    assert before.session_index.iloc[0] == len(development)
    changed = daily.copy()
    changed.loc[changed.quote_date > "2025-01-20", "rr25_30_60"] = 100
    changed["future_target"] = 999
    assert fit_rr_spread_signal(changed, prices) == fitted
    after = forecast_rr_spread_signal(fitted, changed, prices)
    np.testing.assert_allclose(before.loc[before.quote_date <= "2025-01-20", "forecast"],
                               after.loc[after.quote_date <= "2025-01-20", "forecast"], equal_nan=True)
    with pytest.raises(ValueError, match="every SPX session"):
        fit_rr_spread_signal(daily.drop(index=100), prices)


def test_greeks_match_price_derivatives_and_rr_delta_convention(monkeypatch):
    from src.skew_metrics import _forward_delta
    for kind, fn in (("call", black_scholes_call_price), ("put", black_scholes_put_price)):
        f, k, d, tau, sigma, spot = 103., 100., .98, .2, .23, 101.
        greek = black_scholes_greeks(f, k, d, tau, sigma, kind, spot)
        bump = 1e-4
        spot_diff = (fn(f * (spot + bump) / spot, k, d, tau, sigma)
                     - fn(f * (spot - bump) / spot, k, d, tau, sigma)) / (2 * bump)
        vol_diff = (fn(f, k, d, tau, sigma + bump) - fn(f, k, d, tau, sigma - bump)) / (2 * bump)
        assert greek["spot_delta"] == pytest.approx(spot_diff, rel=1e-7)
        assert greek["vega"] == pytest.approx(vol_diff, rel=1e-6)
        # RR25 uses absolute put delta; trading uses its signed equivalent.
        monkeypatch.setattr("src.skew_metrics.evaluate_surface", lambda *args: {"total_variance": sigma**2 * tau})
        repo_delta = _forward_delta(np.log(k / f), tau, {}, kind)
        assert abs(greek["forward_delta"]) == pytest.approx(repo_delta)
