import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize


def raw_svi_total_variance(k, a, b, rho, m, sigma):
    """Evaluate raw SVI total variance for log-moneyness k."""
    shifted_k = np.asarray(k) - m
    return a + b * (rho * shifted_k + np.sqrt(shifted_k**2 + sigma**2))


def _raw_svi_g(k, params):
    a, b, rho, m, sigma = params
    k = np.asarray(k, dtype=float)
    x = k - m
    w = raw_svi_total_variance(k, *params)
    w_k = b * (rho + x / np.sqrt(x**2 + sigma**2))
    w_kk = b * sigma**2 / (x**2 + sigma**2) ** 1.5
    w = np.maximum(w, 1e-12)
    return (
        (1 - k * w_k / (2 * w)) ** 2
        - (w_k**2 / 4) * (1 / w + 1 / 4)
        + w_kk / 2
    )


def raw_svi_butterfly_diagnostic(a, b, rho, m, sigma, k_grid=None):
    """Check variance positivity, Durrleman g(k), and both wing slopes."""
    if b < 0 or abs(rho) >= 1 or sigma <= 0:
        raise ValueError("Raw SVI parameters need b >= 0, |rho| < 1, and sigma > 0.")
    if k_grid is None:
        k_grid = np.linspace(-5.0, 5.0, 4001)

    k = np.asarray(k_grid, dtype=float)
    w = raw_svi_total_variance(k, a, b, rho, m, sigma)
    if (w <= 0).any():
        g = np.full_like(w, np.nan)
    else:
        g = _raw_svi_g(k, (a, b, rho, m, sigma))

    minimum_w = a + b * sigma * np.sqrt(1 - rho**2)
    minimum_g_index = int(np.nanargmin(g)) if np.isfinite(g).any() else None
    minimum_g = float(g[minimum_g_index]) if minimum_g_index is not None else np.nan
    right_wing_slope = b * (1 + rho)
    left_wing_slope = b * (1 - rho)
    return {
        "minimum_total_variance": float(minimum_w),
        "minimum_g_on_grid": minimum_g,
        "k_at_minimum_g": float(k[minimum_g_index]) if minimum_g_index is not None else np.nan,
        "right_wing_slope": float(right_wing_slope),
        "left_wing_slope": float(left_wing_slope),
        "nonnegative_total_variance": bool(minimum_w >= 0),
        "g_nonnegative_on_grid": bool(minimum_g >= -1e-10),
        "right_wing_condition": bool(right_wing_slope < 2),
        "left_wing_condition": bool(left_wing_slope < 2),
    }


def fit_raw_svi_surface(
    panel, apply_quote_screen=True, initial_guess=None, enforce_arbitrage=False
):
    """Fit one five-parameter raw SVI slice per expiry, optionally with arb constraints."""
    if panel.empty or panel["quote_date"].nunique() != 1:
        raise ValueError("panel must contain one nonempty quote date.")
    if panel["security_id"].nunique() != 1:
        raise ValueError("panel must contain one security_id.")

    use_for_fit = panel["use_for_surface"].fillna(False)
    if apply_quote_screen:
        quote_columns = [
            "best_bid", "best_ask", "mid_price", "mid_iv", "relative_spread"
        ]
        finite_quotes = np.isfinite(
            panel[quote_columns].to_numpy(dtype=float)
        ).all(axis=1)
        use_for_fit = (
            use_for_fit
            & finite_quotes
            & panel["best_bid"].gt(0)
            & panel["best_ask"].gt(panel["best_bid"])
            & panel["mid_price"].gt(0)
            & panel["relative_spread"].between(0, 0.50, inclusive="right")
        )

    rows = panel.loc[use_for_fit].copy()
    if rows.empty:
        raise ValueError("No OTM quotes are available for raw SVI fitting.")
    for column in ["log_moneyness", "mid_iv", "time_to_expiry"]:
        values = rows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite for fitted quotes.")
    if not (rows["mid_iv"] > 0).all() or not (rows["time_to_expiry"] > 0).all():
        raise ValueError("Fitted quotes need positive IV and time to expiry.")

    parameter_rows = []
    fitted_slices = []
    for expiry, expiry_rows in rows.groupby("expiry_date", sort=True):
        if len(expiry_rows) < 5:
            raise ValueError(f"Expiry {expiry} needs at least five quotes to fit raw SVI.")

        k = expiry_rows["log_moneyness"].to_numpy(dtype=float)
        market_w = (
            expiry_rows["mid_iv"].to_numpy(dtype=float) ** 2
            * expiry_rows["time_to_expiry"].to_numpy(dtype=float)
        )
        if initial_guess is None and enforce_arbitrage:
            x0 = [market_w.min() - 0.001, 0.01, 0.0, 0.0, 0.1]
        elif initial_guess is None:
            x0 = [max(float(market_w.min()) - 0.01, -1.0), 0.1, -0.3, 0.0, 0.1]
        else:
            x0 = np.asarray(initial_guess, dtype=float)
            if x0.shape != (5,) or not np.isfinite(x0).all():
                raise ValueError("initial_guess must contain five finite Raw SVI parameters.")

        lower = [-1.0, 1e-8, -0.999, -5.0, 1e-8]
        upper = [1.0, 10.0, 0.999, 5.0, 5.0]
        residuals = lambda params: raw_svi_total_variance(k, *params) - market_w
        if enforce_arbitrage:
            dense_grid = np.linspace(-5.0, 5.0, 4001)
            constraint_grid = np.unique(
                np.concatenate(
                    [
                        np.linspace(-5.0, 5.0, 1001),
                        np.linspace(k.min(), k.max(), 301),
                        k,
                    ]
                )
            )

            def minimum_variance(params):
                a, b, rho, _, sigma = params
                return a + b * sigma * np.sqrt(1 - rho**2)

            scale = max(float(np.sqrt(np.mean(market_w**2))), 1e-8)
            for _ in range(4):
                constraints = [
                    {"type": "ineq", "fun": lambda p: minimum_variance(p) - 1e-12},
                    {"type": "ineq", "fun": lambda p: 2 - p[1] * (1 + p[2]) - 1e-8},
                    {"type": "ineq", "fun": lambda p: 2 - p[1] * (1 - p[2]) - 1e-8},
                    {"type": "ineq", "fun": lambda p: _raw_svi_g(constraint_grid, p)},
                ]
                fit = minimize(
                    lambda params: np.mean((residuals(params) / scale) ** 2),
                    x0=x0,
                    method="SLSQP",
                    bounds=list(zip(lower, upper)),
                    constraints=constraints,
                    options={"maxiter": 1000, "ftol": 1e-10},
                )
                if not fit.success:
                    break
                dense_g = _raw_svi_g(dense_grid, fit.x)
                if np.nanmin(dense_g) >= -1e-8:
                    break
                constraint_grid = np.unique(
                    np.concatenate([constraint_grid, dense_grid[dense_g < -1e-8]])
                )
                x0 = fit.x
        else:
            fit = least_squares(
                residuals,
                x0=x0,
                bounds=(lower, upper),
                max_nfev=10000,
            )
        if not fit.success:
            raise ValueError(f"Raw SVI calibration failed for {expiry}: {fit.message}")

        a, b, rho, m, sigma = fit.x
        if enforce_arbitrage:
            dense_checks = raw_svi_butterfly_diagnostic(a, b, rho, m, sigma, dense_grid)
            quoted_checks = raw_svi_butterfly_diagnostic(
                a, b, rho, m, sigma, np.linspace(k.min(), k.max(), 401)
            )
            if (
                not dense_checks["nonnegative_total_variance"]
                or dense_checks["minimum_g_on_grid"] < -1e-8
                or not dense_checks["right_wing_condition"]
                or not dense_checks["left_wing_condition"]
                or quoted_checks["minimum_g_on_grid"] < -1e-8
            ):
                raise ValueError(
                    f"Raw SVI fit for {expiry} failed dense arbitrage checks: "
                    f"min g={dense_checks['minimum_g_on_grid']:.3g}, "
                    f"left slope={dense_checks['left_wing_slope']:.3g}, "
                    f"right slope={dense_checks['right_wing_slope']:.3g}."
                )
        expiry_rows = expiry_rows.copy()
        expiry_rows["market_w"] = market_w
        expiry_rows["fitted_w"] = raw_svi_total_variance(k, *fit.x)
        expiry_rows["residual_w"] = expiry_rows["fitted_w"] - market_w
        fitted_slices.append(expiry_rows)
        parameter_rows.append(
            {
                "quote_date": expiry_rows["quote_date"].iloc[0],
                "expiry_date": expiry,
                "a": a,
                "b": b,
                "rho": rho,
                "m": m,
                "sigma": sigma,
                "rmse": float(np.sqrt(np.mean(expiry_rows["residual_w"] ** 2))),
            }
        )

    return {
        "parameters_by_expiry": pd.DataFrame(parameter_rows),
        "observations": pd.concat(fitted_slices, ignore_index=True),
        "arbitrage_constrained": enforce_arbitrage,
    }
