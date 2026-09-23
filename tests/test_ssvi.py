import numpy as np
import pandas as pd

from src.ssvi import fit_ssvi_surface, ssvi_total_variance


def test_fit_ssvi_surface_recovers_shared_parameters_and_expiry_thetas():
    rho, eta, gamma = -0.4, 1.2, 0.5
    rows = []
    expiries = [
        ("2025-09-29", 0.002, 31 / 365),
        ("2025-11-28", 0.006, 91 / 365),
        ("2026-08-28", 0.02, 364 / 365),
    ]

    for expiry, theta, tau in expiries:
        for k in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            market_w = ssvi_total_variance(k, theta, rho, eta, gamma)
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

    excluded = dict(rows[0], mid_iv=5.0, best_bid=0.0)
    panel = pd.DataFrame(rows + [excluded])

    result = fit_ssvi_surface(panel)

    np.testing.assert_allclose(
        [result["rho"], result["eta"], result["gamma"]],
        [rho, eta, gamma],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        result["theta_by_expiry"]["theta_atm"],
        [theta for _, theta, _ in expiries],
    )
    np.testing.assert_allclose(result["observations"]["residual_w"], 0, atol=1e-8)
    assert len(result["observations"]) == len(rows)

    unscreened = fit_ssvi_surface(panel, apply_quote_screen=False)
    assert len(unscreened["observations"]) == len(rows) + 1
