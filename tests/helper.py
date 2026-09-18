import numpy as np
import pandas as pd

from src.options_chain import calculate_time_to_expiry
from src.pricing import black_scholes_call_price, black_scholes_put_price


def synthetic_option_chain():
    quote_date = pd.Timestamp("2026-01-01")
    expiry_date = pd.Timestamp("2026-07-01")
    forward = 100.0
    discount_factor = 0.98
    volatility = 0.25
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    time_to_expiry = calculate_time_to_expiry(quote_date, expiry_date)

    call_prices = black_scholes_call_price(
        forward,
        strikes,
        discount_factor,
        time_to_expiry,
        volatility,
    )
    put_prices = black_scholes_put_price(
        forward,
        strikes,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    rows = []
    for strike, call_price, put_price in zip(strikes, call_prices, put_prices):
        rows.extend(
            [
                {
                    "quote_date": quote_date,
                    "expiry_date": expiry_date,
                    "strike": strike,
                    "option_type": "call",
                    "bid": call_price,
                    "ask": call_price,
                },
                {
                    "quote_date": quote_date,
                    "expiry_date": expiry_date,
                    "strike": strike,
                    "option_type": "put",
                    "bid": put_price,
                    "ask": put_price,
                },
            ]
        )

    return pd.DataFrame(rows)
