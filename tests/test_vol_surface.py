import numpy as np
import pandas as pd
import pytest

from src.vol_surface import (
    build_total_variance_grid,
    count_calendar_crossings,
    enforce_calendar_monotonicity,
    evaluate_surface,
)


def test_build_total_variance_grid_orders_slices_by_maturity():
    parameters = pd.DataFrame(
        {
            "time_to_expiry": [0.5, 0.25],
            "k_min": [-0.2, -0.1],
            "k_max": [0.2, 0.1],
            "a": [0.02, 0.01],
            "b": [0.2, 0.1],
            "rho": [0.0, 0.0],
            "m": [0.0, 0.0],
            "sigma": [0.1, 0.1],
        }
    )

    grid = build_total_variance_grid(
        parameters,
        k_grid=np.array([-0.25, -0.2, -0.1, 0.0, 0.1, 0.2, 0.25]),
    )

    np.testing.assert_array_equal(grid["time_to_expiry"], [0.25, 0.5])
    assert grid["raw_total_variance"].shape == (2, len(grid["log_moneyness"]))
    lower_support_index = np.flatnonzero(np.isclose(grid["log_moneyness"], -0.2))[0]
    assert np.isnan(grid["raw_total_variance"][0, lower_support_index])
    np.testing.assert_array_equal(
        np.isfinite(grid["raw_total_variance"]), grid["support_mask"]
    )
    np.testing.assert_array_equal(grid["sampled_support_min"], [-0.1, -0.2])
    np.testing.assert_array_equal(grid["sampled_support_max"], [0.1, 0.2])


def test_calendar_projection_isotonic_and_minimum_squared_adjustment():
    raw = np.array([[0.02010, 0.01], [0.02002, 0.02], [0.02030, 0.015]])

    repaired = enforce_calendar_monotonicity(raw)

    np.testing.assert_allclose(repaired[:, 0], [0.02006, 0.02006, 0.02030])
    np.testing.assert_allclose(repaired[:, 1], [0.01, 0.0175, 0.0175])
    assert (np.diff(repaired, axis=0) >= 0).all()


def test_calendar_projection_ignores_unsupported_maturities():
    raw = np.array([[0.03, np.nan], [0.02, 0.04], [0.04, 0.03]])

    repaired = enforce_calendar_monotonicity(raw)

    np.testing.assert_allclose(repaired[:, 0], [0.025, 0.025, 0.04])
    assert np.isnan(repaired[0, 1])
    np.testing.assert_allclose(repaired[1:, 1], [0.035, 0.035])


def test_calendar_crossing_count_compares_consecutive_supported_maturities():
    raw = np.array([[0.03, np.nan], [np.nan, 0.01], [0.02, np.nan]])

    assert count_calendar_crossings(raw) == 1
    assert count_calendar_crossings(enforce_calendar_monotonicity(raw)) == 0


def test_evaluate_surface_interpolates_variance_before_converting_to_iv():
    surface = {
        "time_to_expiry": np.array([1.0, 2.0]),
        "log_moneyness": np.array([0.0, 1.0]),
        "sampled_support_min": np.array([0.0, 0.0]),
        "sampled_support_max": np.array([1.0, 1.0]),
        "repaired_total_variance": np.array([[0.04, 0.09], [0.08, 0.13]]),
    }

    result = evaluate_surface(np.array([0.0, 0.5, 1.0]), 1.5, surface)

    np.testing.assert_allclose(result["total_variance"], [0.06, 0.085, 0.11])
    np.testing.assert_allclose(
        result["implied_volatility"], np.sqrt(np.array([0.06, 0.085, 0.11]) / 1.5)
    )


def test_evaluate_surface_rejects_extrapolation():
    surface = {
        "time_to_expiry": np.array([1.0, 2.0]),
        "log_moneyness": np.array([-0.5, 0.25]),
        "sampled_support_min": np.array([-0.2, -0.1]),
        "sampled_support_max": np.array([0.1, 0.2]),
        "repaired_total_variance": np.array([[0.04, 0.05], [0.06, 0.07]]),
    }

    with pytest.raises(ValueError, match="not bracketed"):
        evaluate_surface(0.0, 2.1, surface)
    with pytest.raises(ValueError, match="log-moneyness grid"):
        evaluate_surface(0.3, 1.5, surface)


def test_evaluate_surface_requires_two_supported_maturities_for_interpolation():
    surface = {
        "time_to_expiry": np.array([1.0, 2.0, 3.0]),
        "log_moneyness": np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
        "sampled_support_min": np.array([-0.2, -0.1, -0.2]),
        "sampled_support_max": np.array([0.2, 0.1, 0.2]),
        "repaired_total_variance": np.array(
            [[0.01] * 5, [np.nan, 0.04, 0.04, 0.04, np.nan], [0.09] * 5]
        ),
    }

    with pytest.raises(ValueError, match="each surrounding expiry"):
        evaluate_surface(-0.15, 1.5, surface)
    with pytest.raises(ValueError, match="each surrounding expiry"):
        evaluate_surface(-0.15, 2.0, surface)

    exact = evaluate_surface(-0.15, 1.0, surface)
    assert exact["total_variance"] == pytest.approx(0.01)

    inside = evaluate_surface(np.array([-0.1, 0.0, 0.1]), 1.5, surface)
    np.testing.assert_allclose(inside["total_variance"], 0.025)
    with pytest.raises(ValueError, match="each surrounding expiry"):
        evaluate_surface(0.1 + 1e-9, 1.5, surface)


def test_exact_expiry_uses_its_own_support_and_preserves_grid_values():
    surface = {
        "time_to_expiry": np.array([1.0, 2.0]),
        "log_moneyness": np.array([-0.1, 0.0, 0.1]),
        "sampled_support_min": np.array([-0.1, 0.0]),
        "sampled_support_max": np.array([0.1, 0.1]),
        "repaired_total_variance": np.array([[0.03, 0.02, 0.025], [np.nan, 0.04, 0.05]]),
    }

    result = evaluate_surface(surface["log_moneyness"], 1.0, surface)

    np.testing.assert_allclose(result["total_variance"], surface["repaired_total_variance"][0])
