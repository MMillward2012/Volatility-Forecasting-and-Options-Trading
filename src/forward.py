import numpy as np


def infer_forward_and_discount_factor(call_prices, put_prices, strikes):
    """
    Infer the forward price and discount factor from matched European
    call and put prices across multiple strikes for the same expiry.
    """

    parity_values = call_prices - put_prices

    slope, intercept = np.polyfit(strikes, parity_values, 1)

    discount_factor = -slope
    forward = intercept / discount_factor

    return forward, discount_factor