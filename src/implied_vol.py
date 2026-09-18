import numpy as np
from scipy.optimize import brentq

from src.pricing import black_scholes_call_price, black_scholes_put_price


def implied_volatility_call(
    market_price,
    forward,
    strike,
    discount_factor,
    time_to_expiry,
):
    """Solve for the call volatility that matches the market price."""

    def objective(volatility):
        return (
            black_scholes_call_price(
                forward,
                strike,
                discount_factor,
                time_to_expiry,
                volatility,
            )
            - market_price
        )

    lower = 1e-8
    upper = 5.0
    maximum_upper = 20.0

    objective_at_lower = objective(lower)

    minimum_price = discount_factor * max(forward - strike, 0.0)
    maximum_price = discount_factor * forward
    if not np.isfinite(market_price) or not minimum_price <= market_price < maximum_price:
        raise ValueError("Market call price must be finite and within Black price bounds.")
    if market_price == minimum_price:
        return 0.0

    if objective_at_lower > 0:
        raise ValueError(
            "Implied volatility is below the minimum volatility bound."
        )

    objective_at_upper = objective(upper)

    while objective_at_upper < 0 and upper < maximum_upper:
        upper = min(upper * 2.0, maximum_upper)
        objective_at_upper = objective(upper)

    if objective_at_upper < 0:
        raise ValueError(
            "Could not bracket implied volatility below the maximum upper bound."
        )

    return brentq(objective, lower, upper)


def implied_volatility_put(
    market_price,
    forward,
    strike,
    discount_factor,
    time_to_expiry,
):
    """Solve for the put volatility that matches the market price."""

    def objective(volatility):
        return (
            black_scholes_put_price(
                forward,
                strike,
                discount_factor,
                time_to_expiry,
                volatility,
            )
            - market_price
        )

    lower = 1e-8
    upper = 5.0
    maximum_upper = 20.0

    objective_at_lower = objective(lower)

    minimum_price = discount_factor * max(strike - forward, 0.0)
    maximum_price = discount_factor * strike
    if not np.isfinite(market_price) or not minimum_price <= market_price < maximum_price:
        raise ValueError("Market put price must be finite and within Black price bounds.")
    if market_price == minimum_price:
        return 0.0

    if objective_at_lower > 0:
        raise ValueError(
            "Implied volatility is below the minimum volatility bound."
        )

    objective_at_upper = objective(upper)

    while objective_at_upper < 0 and upper < maximum_upper:
        upper = min(upper * 2.0, maximum_upper)
        objective_at_upper = objective(upper)

    if objective_at_upper < 0:
        raise ValueError(
            "Could not bracket implied volatility below the maximum upper bound."
        )

    return brentq(objective, lower, upper)
