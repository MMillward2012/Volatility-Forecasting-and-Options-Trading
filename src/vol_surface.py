import numpy as np
from scipy.stats import norm

from src.pricing import black_scholes_call_price
from src.raw_svi import raw_svi_total_variance


def build_total_variance_grid(parameters_by_expiry, k_grid=None):
    """Evaluate Raw SVI slices only within each expiry's observed k support."""
    required = ["time_to_expiry", "k_min", "k_max", "a", "b", "rho", "m", "sigma"]
    missing = set(required) - set(parameters_by_expiry.columns)
    if missing:
        raise ValueError(f"Missing Raw SVI parameter columns: {sorted(missing)}")
    if parameters_by_expiry.empty:
        raise ValueError("At least one fitted expiry is required.")

    parameters = parameters_by_expiry.sort_values("time_to_expiry")
    maturities = parameters["time_to_expiry"].to_numpy(dtype=float)
    support_min = parameters["k_min"].to_numpy(dtype=float)
    support_max = parameters["k_max"].to_numpy(dtype=float)
    if not np.isfinite(maturities).all() or (maturities <= 0).any():
        raise ValueError("Maturities must be finite and positive.")
    if np.any(np.diff(maturities) <= 0):
        raise ValueError("Maturities must be unique.")
    if (
        not np.isfinite(support_min).all()
        or not np.isfinite(support_max).all()
        or (support_min > support_max).any()
    ):
        raise ValueError("Each expiry needs finite, ordered observed k bounds.")

    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.25, 151)
    k_grid = np.asarray(k_grid, dtype=float)
    if k_grid.ndim != 1 or len(k_grid) < 2:
        raise ValueError("k_grid must be a one-dimensional grid with at least two points.")
    if not np.isfinite(k_grid).all() or np.any(np.diff(k_grid) <= 0):
        raise ValueError("k_grid must be finite and strictly increasing.")

    support_mask = (
        (k_grid[None, :] >= support_min[:, None])
        & (k_grid[None, :] <= support_max[:, None])
    )
    sampled_support_min = np.full(len(maturities), np.nan)
    sampled_support_max = np.full(len(maturities), np.nan)
    for i in range(len(maturities)):
        supported_k = k_grid[support_mask[i]]
        if supported_k.size:
            sampled_support_min[i] = supported_k[0]
            sampled_support_max[i] = supported_k[-1]
    raw_variance = np.full(support_mask.shape, np.nan)
    for i, (_, row) in enumerate(parameters.iterrows()):
        supported_k = k_grid[support_mask[i]]
        if supported_k.size:
            values = raw_svi_total_variance(
                supported_k,
                *(row[column] for column in ["a", "b", "rho", "m", "sigma"]),
            )
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Fitted total variance must be finite and positive on its support.")
            raw_variance[i, support_mask[i]] = values

    return {
        "time_to_expiry": maturities,
        "log_moneyness": k_grid,
        "support_min": support_min,
        "support_max": support_max,
        "sampled_support_min": sampled_support_min,
        "sampled_support_max": sampled_support_max,
        "support_mask": support_mask,
        "raw_total_variance": raw_variance,
    }


def _isotonic_non_decreasing(values):
    """Return the equal-weight least-squares nondecreasing projection."""
    means = []
    weights = []
    for value in values:
        means.append(float(value))
        weights.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (
                means[-2] * weights[-2] + means[-1] * weights[-1]
            ) / weight
            means[-2:] = [mean]
            weights[-2:] = [weight]
    return np.repeat(means, weights)


def enforce_calendar_monotonicity(total_variance_grid):
    """Apply PAVA at each k using only maturities with finite supported values."""
    total_variance_grid = np.asarray(total_variance_grid, dtype=float)
    if total_variance_grid.ndim != 2 or total_variance_grid.shape[0] == 0:
        raise ValueError("total_variance_grid must be a nonempty 2D array.")
    if np.isinf(total_variance_grid).any():
        raise ValueError("total_variance_grid cannot contain infinite values.")

    repaired = total_variance_grid.copy()
    for column in range(total_variance_grid.shape[1]):
        supported = np.isfinite(total_variance_grid[:, column])
        if supported.any():
            repaired[supported, column] = _isotonic_non_decreasing(
                total_variance_grid[supported, column]
            )
    return repaired


def count_calendar_crossings(total_variance_grid, tolerance=1e-10):
    """Count decreases between consecutive supported maturities at each k."""
    values = np.asarray(total_variance_grid, dtype=float)
    return sum(
        int(np.count_nonzero(np.diff(column[np.isfinite(column)]) < -tolerance))
        for column in values.T
    )


def evaluate_surface(k, time_to_expiry, surface_grid):
    """Interpolate variance using fixed surrounding expiries and their shared support."""
    k_grid = np.asarray(surface_grid["log_moneyness"], dtype=float)
    maturities = np.asarray(surface_grid["time_to_expiry"], dtype=float)
    support_min = np.asarray(surface_grid["sampled_support_min"], dtype=float)
    support_max = np.asarray(surface_grid["sampled_support_max"], dtype=float)
    total_variance = np.asarray(surface_grid["repaired_total_variance"], dtype=float)
    k_values = np.asarray(k, dtype=float)
    tau = float(time_to_expiry)

    if (
        k_grid.ndim != 1
        or len(k_grid) < 2
        or not np.isfinite(k_grid).all()
        or np.any(np.diff(k_grid) <= 0)
    ):
        raise ValueError("Surface log-moneyness grid must be finite and increasing.")
    if (
        maturities.ndim != 1
        or len(maturities) == 0
        or not np.isfinite(maturities).all()
        or np.any(np.diff(maturities) <= 0)
    ):
        raise ValueError("Surface maturities must be finite and increasing.")
    if total_variance.shape != (len(maturities), len(k_grid)):
        raise ValueError("Repaired total-variance grid has an incompatible shape.")
    if np.isinf(total_variance).any():
        raise ValueError("Repaired total variance cannot contain infinite values.")
    if support_min.shape != maturities.shape or support_max.shape != maturities.shape:
        raise ValueError("Support bounds must contain one value per maturity.")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("time_to_expiry must be finite and positive.")
    if not np.isfinite(k_values).all():
        raise ValueError("k must contain only finite values.")
    if tau < maturities[0] or tau > maturities[-1]:
        raise ValueError("time_to_expiry is not bracketed by fitted expiries.")

    right_index = np.searchsorted(maturities, tau)
    if maturities[right_index] == tau:
        maturity_indices = [right_index]
    else:
        maturity_indices = [right_index - 1, right_index]

    scalar_input = k_values.ndim == 0
    flat_k = k_values.reshape(-1)
    if ((flat_k < k_grid[0]) | (flat_k > k_grid[-1])).any():
        raise ValueError("k lies outside the common log-moneyness grid.")

    slice_values = []
    for index in maturity_indices:
        supported = (support_min[index] <= flat_k) & (flat_k <= support_max[index])
        if not supported.all():
            raise ValueError(
                "k must lie within the sampled support of each surrounding expiry."
            )
        finite_k = np.isfinite(total_variance[index])
        slice_values.append(
            np.interp(flat_k, k_grid[finite_k], total_variance[index, finite_k])
        )

    values = slice_values[0]
    if len(maturity_indices) == 2:
        near_tau, far_tau = maturities[maturity_indices]
        weight = (tau - near_tau) / (far_tau - near_tau)
        values = (1 - weight) * values + weight * slice_values[1]

    interpolated_variance = np.asarray(values).reshape(k_values.shape)
    implied_volatility = np.sqrt(interpolated_variance / tau)
    if scalar_input:
        interpolated_variance = float(interpolated_variance)
        implied_volatility = float(implied_volatility)
    return {
        "total_variance": interpolated_variance,
        "implied_volatility": implied_volatility,
    }


def check_surface_quality(time_to_expiry, surface_grid, k_range=None):
    """Report sampled call-price slopes and convexity on a requested support range."""
    tau = float(time_to_expiry)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("time_to_expiry must be finite and positive.")
    if k_range is not None:
        k_range = np.asarray(k_range, dtype=float)
        if k_range.shape != (2,) or not np.isfinite(k_range).all() or k_range[0] >= k_range[1]:
            raise ValueError("k_range must contain two finite, increasing bounds.")

    result = {
        "time_to_expiry": tau,
        "use_for_skew": False,
        "failure_reason": "",
        "k_min": np.nan,
        "k_max": np.nan,
        "n_test_strikes": 0,
        "call_slopes_in_bounds": False,
        "minimum_slope_change": np.nan,
        "convexity_violations": np.nan,
    }
    maturities = np.asarray(surface_grid["time_to_expiry"])
    if tau < maturities[0] or tau > maturities[-1]:
        result["failure_reason"] = "outside_maturity_range"
        return result

    right = np.searchsorted(maturities, tau)
    left = right if maturities[right] == tau else right - 1
    lower = np.max(surface_grid["sampled_support_min"][left:right + 1])
    upper = np.min(surface_grid["sampled_support_max"][left:right + 1])
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        result["failure_reason"] = "insufficient_shared_support"
        return result
    if k_range is not None:
        if k_range[0] < lower or k_range[1] > upper:
            result["failure_reason"] = "requested_range_not_supported"
            return result
        lower, upper = k_range
    result.update(k_min=float(lower), k_max=float(upper))

    k_grid = np.asarray(surface_grid["log_moneyness"])
    k_nodes = k_grid[(k_grid >= lower) & (k_grid <= upper)]
    knot_step = 0.01 * np.min(np.diff(k_grid))
    test_k = np.unique(np.r_[
        np.linspace(lower, upper, 2001),
        k_nodes,
        k_nodes[1:-1] - knot_step,
        k_nodes[1:-1] + knot_step,
    ])
    test_k = np.unique(np.clip(test_k, lower, upper))
    strikes = np.exp(test_k)
    distinct_strikes = np.r_[True, np.diff(strikes) > 0]
    test_k = test_k[distinct_strikes]
    strikes = strikes[distinct_strikes]
    evaluated = evaluate_surface(test_k, tau, surface_grid)
    if (
        not np.isfinite(evaluated["implied_volatility"]).all()
        or (evaluated["implied_volatility"] <= 0).any()
    ):
        result["failure_reason"] = "invalid_total_variance"
        return result

    calls = black_scholes_call_price(1.0, strikes, 1.0, tau, evaluated["implied_volatility"])
    call_slopes = np.diff(calls) / np.diff(strikes)
    slope_changes = np.diff(call_slopes)
    slopes_in_bounds = bool(((call_slopes >= -1 - 1e-8) & (call_slopes <= 1e-8)).all())
    violations = int((slope_changes < -1e-8).sum())
    slope_locations = (test_k[:-1] + test_k[1:]) / 2
    reasons = []
    if not slopes_in_bounds:
        reasons.append("call_slope_bounds")
    if violations:
        reasons.append("butterfly_convexity")
    result.update(
        use_for_skew=not reasons,
        failure_reason=";".join(reasons),
        n_test_strikes=len(strikes),
        call_slopes_in_bounds=slopes_in_bounds,
        minimum_call_slope=float(call_slopes.min()),
        maximum_call_slope=float(call_slopes.max()),
        k_at_minimum_call_slope=float(slope_locations[np.argmin(call_slopes)]),
        k_at_maximum_call_slope=float(slope_locations[np.argmax(call_slopes)]),
        minimum_slope_change=float(slope_changes.min()),
        convexity_violations=violations,
    )
    return result
