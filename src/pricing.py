import numpy as np
from scipy.stats import norm

def black_scholes_call_price(forward, strike, discount_factor, time_to_expiry, volatility):
    """
    Calculate the Black-Scholes price of a European call option.
    """

    d1 = (np.log(forward / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    call_price = discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))

    return call_price


def black_scholes_put_price(forward, strike, discount_factor, time_to_expiry, volatility):
    """
    Calculate the Black-Scholes price of a European put option.
    """

    d1 = (np.log(forward / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    put_price = discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))

    return put_price
