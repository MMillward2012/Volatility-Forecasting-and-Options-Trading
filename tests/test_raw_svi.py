import numpy as np
import pandas as pd
import pytest

from src.raw_svi import (
    fit_raw_svi_surface,
    raw_svi_butterfly_diagnostic,
    raw_svi_total_variance,
)


def test_raw_svi_total_variance_matches_formula():
    k = np.array([-0.1, 0.0, 0.1])
    a, b, rho, m, sigma = 0.02, 0.3, -0.4, 0.01, 0.15
    shifted_k = k - m
    expected = a + b * (
        rho * shifted_k + np.sqrt(shifted_k**2 + sigma**2)
    )

    np.testing.assert_allclose(
        raw_svi_total_variance(k, a, b, rho, m, sigma), expected
    )


def test_fit_raw_svi_surface_fits_each_expiry_independently():
    params_by_expiry = {
        "2025-11-28": (0.01, 0.2, -0.35, -0.02, 0.12),
        "2026-02-27": (0.02, 0.25, -0.2, 0.01, 0.16),
    }
    rows = []
    for expiry, params in params_by_expiry.items():
        tau = 0.25 if expiry == "2025-11-28" else 0.5
        for k in np.linspace(-0.3, 0.3, 31):
            market_w = raw_svi_total_variance(k, *params)
            rows.append(
                {
                    "security_id": 108105,
                    "quote_date": "2025-08-29",
                    "expiry_date": expiry,
                    "time_to_expiry": tau,
                    "log_moneyness": k,
                    "mid_iv": np.sqrt(market_w / tau),
                    "best_bid": 1.0,
                    "best_ask": 1.2,
                    "mid_price": 1.1,
                    "relative_spread": 0.2 / 1.1,
                    "use_for_surface": True,
                }
            )

    result = fit_raw_svi_surface(pd.DataFrame(rows))

    assert len(result["parameters_by_expiry"]) == 2
    assert len(result["observations"]) == len(rows)
    np.testing.assert_allclose(result["observations"]["residual_w"], 0, atol=1e-8)

    constrained = fit_raw_svi_surface(pd.DataFrame(rows), enforce_arbitrage=True)
    assert constrained["arbitrage_constrained"]
    assert len(constrained["parameters_by_expiry"]) == 2


def test_arbitrage_aware_fit_uses_feasible_multistart_candidate():
    tau = 0.25
    market_params = (0.02, 2.4, 0.0, 0.0, 0.2)
    rows = []
    for k in np.linspace(-0.3, 0.3, 31):
        market_w = raw_svi_total_variance(k, *market_params)
        rows.append(
            {
                "security_id": 108105,
                "quote_date": "2025-08-29",
                "expiry_date": "2025-11-28",
                "time_to_expiry": tau,
                "log_moneyness": k,
                "mid_iv": np.sqrt(market_w / tau),
                "best_bid": 1.0,
                "best_ask": 1.2,
                "mid_price": 1.1,
                "relative_spread": 0.2 / 1.1,
                "use_for_surface": True,
            }
        )

    panel = pd.DataFrame(rows)
    unconstrained = fit_raw_svi_surface(panel)["parameters_by_expiry"].iloc[0]
    constrained = fit_raw_svi_surface(panel, enforce_arbitrage=True)[
        "parameters_by_expiry"
    ].iloc[0]

    assert unconstrained["b"] * (1 + abs(unconstrained["rho"])) > 2
    assert constrained["feasible_starts"] > 1
    assert constrained["b"] * (1 + constrained["rho"]) < 2
    assert constrained["b"] * (1 - constrained["rho"]) < 2
    assert constrained["rmse"] > unconstrained["rmse"]


def test_fit_raw_svi_surface_requires_five_quotes_per_expiry():
    rows = pd.DataFrame(
        {
            "security_id": [108105] * 4,
            "quote_date": ["2025-08-29"] * 4,
            "expiry_date": ["2025-11-28"] * 4,
            "time_to_expiry": [0.25] * 4,
            "log_moneyness": [-0.1, 0.0, 0.1, 0.2],
            "mid_iv": [0.2] * 4,
            "best_bid": [1.0] * 4,
            "best_ask": [1.2] * 4,
            "mid_price": [1.1] * 4,
            "relative_spread": [0.2 / 1.1] * 4,
            "use_for_surface": [True] * 4,
        }
    )

    with pytest.raises(ValueError, match="at least five quotes"):
        fit_raw_svi_surface(rows)


def test_raw_svi_butterfly_diagnostic_checks_safe_slice():
    result = raw_svi_butterfly_diagnostic(0.01, 0.2, -0.3, 0.0, 0.15)

    assert result["nonnegative_total_variance"]
    assert result["g_nonnegative_on_grid"]
    assert result["right_wing_condition"]


def test_raw_svi_butterfly_diagnostic_flags_negative_variance():
    result = raw_svi_butterfly_diagnostic(-0.1, 0.3, 0.5, 0.0, 0.1)

    assert not result["nonnegative_total_variance"]
    assert not result["g_nonnegative_on_grid"]


def test_raw_svi_butterfly_diagnostic_checks_both_wings():
    right_violation = raw_svi_butterfly_diagnostic(0.1, 1.2, 0.8, 0.0, 0.2)
    left_violation = raw_svi_butterfly_diagnostic(0.1, 1.2, -0.8, 0.0, 0.2)

    assert not right_violation["right_wing_condition"]
    assert right_violation["left_wing_condition"]
    assert left_violation["right_wing_condition"]
    assert not left_violation["left_wing_condition"]


def test_raw_svi_butterfly_diagnostic_separates_quote_region_from_extrapolation():
    params = (-0.1, 1.0, 0.0, 0.0, 0.3)
    quoted = raw_svi_butterfly_diagnostic(*params, np.linspace(-0.3, 0.2, 401))
    extrapolated = raw_svi_butterfly_diagnostic(*params, np.linspace(-5.0, 5.0, 4001))

    assert quoted["g_nonnegative_on_grid"]
    assert not extrapolated["g_nonnegative_on_grid"]
