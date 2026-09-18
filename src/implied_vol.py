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

    if objective_at_lower > 0:
        raise ValueError(
            "Market price is below the minimum Black call price."
        )

    objective_at_upper = objective(upper)

    # If the Black price is still below the market price at the current upper
    # volatility bound, then the root must lie at a higher volatility because
    # the Black call price increases monotonically with volatility.
    # Double the upper bound until the objective becomes non-negative,
    # meaning the root has been bracketed, or until the safety cap is reached.
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

    if objective_at_lower > 0:
        raise ValueError(
            "Market price is below the minimum Black put price."
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
