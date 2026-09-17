import pytest

from src.pricing import (black_scholes_call_price, black_scholes_put_price)

def test_put_call_parity():
    forward = 100.0
    strike = 110.0
    discount_factor = 0.98
    time_to_expiry = 0.5
    volatility = 0.25

    call = black_scholes_call_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
)

    put = black_scholes_put_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    assert call - put == pytest.approx(
        discount_factor * (forward - strike)
    )

def test_call_decreases_with_strike():
    forward = 100.0
    strike = 110.0
    discount_factor = 0.98
    time_to_expiry = 0.5
    volatility = 0.25

    call1 = black_scholes_call_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    call2 = black_scholes_call_price(
        forward,
        strike + 10.0,  # Increase strike price
        discount_factor,
        time_to_expiry,
        volatility,
    )

    assert call1 > call2, "Call price should decrease as strike price increases"