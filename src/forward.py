import numpy as np


def infer_forward_and_discount_factor(call_prices, put_prices, strikes):
    """
    Infer the forward price and discount factor from matched European
    call and put prices across multiple strikes for the same expiry.
    """
    call_prices = np.asarray(call_prices, dtype=float)
    put_prices = np.asarray(put_prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)

    if call_prices.ndim != 1 or put_prices.ndim != 1 or strikes.ndim != 1:
        raise ValueError("call_prices, put_prices, and strikes must be 1-dimensional")
    if not (len(call_prices) == len(put_prices) == len(strikes)):
        raise ValueError("call_prices, put_prices, and strikes must have the same length")
    if len(strikes) < 2 or len(np.unique(strikes)) < 2:
        raise ValueError("at least two distinct strikes are required")
    if not np.all(np.isfinite(call_prices)):
        raise ValueError("call_prices must contain only finite values")
    if not np.all(np.isfinite(put_prices)):
        raise ValueError("put_prices must contain only finite values")
    if np.any(call_prices < 0) or np.any(put_prices < 0):
        raise ValueError("call_prices and put_prices must be non-negative")
    if not np.all(np.isfinite(strikes)):
        raise ValueError("strikes must contain only finite values")
    if not np.all(strikes > 0):
        raise ValueError("strikes must contain only positive values")

    parity_values = call_prices - put_prices

    slope, intercept = np.polyfit(strikes, parity_values, 1)

    discount_factor = -slope
    if not np.isfinite(discount_factor) or discount_factor <= 0:
        raise ValueError("inferred discount_factor must be finite and positive")

    forward = intercept / discount_factor
    if not np.isfinite(forward) or forward <= 0:
        raise ValueError("inferred forward must be finite and positive")

    return forward, discount_factor
