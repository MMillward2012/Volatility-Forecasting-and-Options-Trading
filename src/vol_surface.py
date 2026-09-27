import numpy as np

from src.raw_svi import raw_svi_total_variance


def build_total_variance_grid(parameters_by_expiry, k_grid=None):
    """Evaluate fitted Raw SVI slices on an ordered common maturity and k grid."""
    required = ["time_to_expiry", "a", "b", "rho", "m", "sigma"]
    missing = set(required) - set(parameters_by_expiry.columns)
    if missing:
        raise ValueError(f"Missing Raw SVI parameter columns: {sorted(missing)}")
    if parameters_by_expiry.empty:
        raise ValueError("At least one fitted expiry is required.")

    parameters = parameters_by_expiry.sort_values("time_to_expiry")
    maturities = parameters["time_to_expiry"].to_numpy(dtype=float)
    if not np.isfinite(maturities).all() or (maturities <= 0).any():
        raise ValueError("Maturities must be finite and positive.")
    if np.any(np.diff(maturities) <= 0):
        raise ValueError("Maturities must be unique.")

    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.25, 151)
    k_grid = np.asarray(k_grid, dtype=float)
    if k_grid.ndim != 1 or len(k_grid) < 2:
        raise ValueError("k_grid must be a one-dimensional grid with at least two points.")
    if not np.isfinite(k_grid).all() or np.any(np.diff(k_grid) <= 0):
        raise ValueError("k_grid must be finite and strictly increasing.")

    total_variance = np.vstack(
        [
            raw_svi_total_variance(
                k_grid,
                *(row[column] for column in ["a", "b", "rho", "m", "sigma"]),
            )
            for _, row in parameters.iterrows()
        ]
    )
    if not np.isfinite(total_variance).all() or (total_variance <= 0).any():
        raise ValueError("Fitted total variance must be finite and positive on k_grid.")

    return {
        "time_to_expiry": maturities,
        "log_moneyness": k_grid,
        "raw_total_variance": total_variance,
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
    """Project each k-column to its closest nondecreasing maturity sequence."""
    total_variance_grid = np.asarray(total_variance_grid, dtype=float)
    if total_variance_grid.ndim != 2 or total_variance_grid.shape[0] == 0:
        raise ValueError("total_variance_grid must be a nonempty 2D array.")
    if not np.isfinite(total_variance_grid).all():
        raise ValueError("total_variance_grid must contain only finite values.")

    return np.column_stack(
        [
            _isotonic_non_decreasing(total_variance_grid[:, column])
            for column in range(total_variance_grid.shape[1])
        ]
    )


def evaluate_surface(k, time_to_expiry, surface_grid):
    """Interpolate repaired total variance in k and maturity, then return IV."""
    k_grid = np.asarray(surface_grid["log_moneyness"], dtype=float)
    maturities = np.asarray(surface_grid["time_to_expiry"], dtype=float)
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
    if not np.isfinite(total_variance).all() or (total_variance <= 0).any():
        raise ValueError("Repaired total variance must be finite and positive.")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("time_to_expiry must be finite and positive.")
    if tau < maturities[0] or tau > maturities[-1]:
        raise ValueError("time_to_expiry must lie within the fitted maturity range.")
    if not np.isfinite(k_values).all():
        raise ValueError("k must contain only finite values.")
    if (k_values < k_grid[0]).any() or (k_values > k_grid[-1]).any():
        raise ValueError("k must lie within the fitted log-moneyness grid.")

    scalar_input = k_values.ndim == 0
    flat_k = k_values.reshape(-1)
    variance_by_maturity = np.vstack(
        [np.interp(flat_k, k_grid, row) for row in total_variance]
    )
    interpolated_variance = np.array(
        [np.interp(tau, maturities, variance_by_maturity[:, i]) for i in range(len(flat_k))]
    ).reshape(k_values.shape)
    implied_volatility = np.sqrt(interpolated_variance / tau)
    if scalar_input:
        interpolated_variance = float(interpolated_variance)
        implied_volatility = float(implied_volatility)
    return {
        "total_variance": interpolated_variance,
        "implied_volatility": implied_volatility,
    }
