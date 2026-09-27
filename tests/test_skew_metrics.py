import numpy as np
import pytest

from src.skew_metrics import calculate_skew_metrics


def _flat_surface(upper_k=0.3):
    k_grid = np.linspace(-0.3, upper_k, 121)
    maturities = np.array([30.0, 90.0]) / 365
    return {
        "time_to_expiry": maturities,
        "log_moneyness": k_grid,
        "sampled_support_min": np.array([-0.3, -0.3]),
        "sampled_support_max": np.array([upper_k, upper_k]),
        "repaired_total_variance": np.vstack([0.04 * maturities[0] * np.ones_like(k_grid),
                                              0.04 * maturities[1] * np.ones_like(k_grid)]),
    }


def test_skew_metrics_calculate_atm_slope_and_flat_smile_rr25():
    result = calculate_skew_metrics(_flat_surface(), "2025-08-29")
    row = result["metrics"].set_index("target_days").loc[60]

    assert row["atm_valid"]
    assert row["slope_valid"]
    assert row["rr25_valid"]
    assert row["atm_iv"] == pytest.approx(0.2, abs=1e-10)
    assert row["atm_skew_slope"] == pytest.approx(0.0, abs=1e-10)
    assert row["rr25_downside"] == pytest.approx(0.0, abs=1e-10)
    assert row["k_25_put"] < 0 < row["k_25_call"]


def test_unsupported_wing_does_not_erase_atm_or_slope_metrics():
    result = calculate_skew_metrics(_flat_surface(upper_k=0.05), "2025-08-29")
    row = result["metrics"].set_index("target_days").loc[60]

    assert row["atm_valid"]
    assert row["slope_valid"]
    assert np.isfinite(row["atm_iv"])
    assert np.isfinite(row["atm_skew_slope"])
    assert not row["rr25_valid"]
    assert np.isnan(row["rr25_downside"])
    assert not result["smiles"].loc[
        result["smiles"]["target_days"].eq(60), "surface_valid"
    ].any()


def test_downside_skew_and_risk_reversal_are_positive_on_negative_skew():
    surface = _flat_surface()
    k_grid = surface["log_moneyness"]
    maturities = surface["time_to_expiry"]
    surface["repaired_total_variance"] = np.vstack(
        [tau * (0.2 - 0.2 * k_grid) ** 2 for tau in maturities]
    )

    row = calculate_skew_metrics(surface, "2025-08-29")["metrics"].set_index(
        "target_days"
    ).loc[60]

    assert row["slope_valid"] and row["rr25_valid"]
    assert row["atm_skew_slope"] > 0
    assert row["rr25_downside"] > 0
