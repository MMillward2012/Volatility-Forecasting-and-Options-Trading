import numpy as np
from scipy.stats import norm


def _validate_positive_finite(name, value):
    """Return value as an array after validating positive finite values."""
    value = np.asarray(value, dtype=float)

    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all(value > 0):
        raise ValueError(f"{name} must contain only positive values")

    return value


def _validate_pricing_inputs(forward, strike, discount_factor, time_to_expiry, volatility):
    return (
        _validate_positive_finite("forward", forward),
        _validate_positive_finite("strike", strike),
        _validate_positive_finite("discount_factor", discount_factor),
        _validate_positive_finite("time_to_expiry", time_to_expiry),
        _validate_positive_finite("volatility", volatility),
    )


def black_scholes_call_price(forward, strike, discount_factor, time_to_expiry, volatility):
    """
    Calculate the Black-Scholes price of a European call option.
    """
    (
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    ) = _validate_pricing_inputs(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    d1 = (np.log(forward / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    call_price = discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))

    return call_price


def black_scholes_put_price(forward, strike, discount_factor, time_to_expiry, volatility):
    """
    Calculate the Black-Scholes price of a European put option.
    """
    (
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    ) = _validate_pricing_inputs(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    d1 = (np.log(forward / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    put_price = discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))

    return put_price


def black_scholes_greeks(forward, strike, discount_factor, time_to_expiry,
                        volatility, option_type, spot):
    """Return signed forward delta, spot delta and vega in quote-price units."""
    forward, strike, discount_factor, time_to_expiry, volatility = _validate_pricing_inputs(
        forward, strike, discount_factor, time_to_expiry, volatility
    )
    spot = _validate_positive_finite("spot", spot)
    option_type = np.asarray(option_type)
    if not np.isin(option_type, ["call", "put"]).all():
        raise ValueError("option_type must contain only call or put")
    d1 = (np.log(forward / strike) + 0.5 * volatility**2 * time_to_expiry) / (
        volatility * np.sqrt(time_to_expiry)
    )
    forward_delta = norm.cdf(d1) - (option_type == "put")
    return {
        "forward_delta": forward_delta,
        "spot_delta": discount_factor * forward / spot * forward_delta,
        "vega": discount_factor * forward * norm.pdf(d1) * np.sqrt(time_to_expiry),
    }
