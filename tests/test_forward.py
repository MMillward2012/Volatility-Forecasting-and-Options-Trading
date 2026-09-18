import pytest

import numpy as np

from src.pricing import (black_scholes_call_price, black_scholes_put_price)
from src.forward import infer_forward_and_discount_factor


def test_infer_forward_and_discount_factor():
    forward = 100.0
    discount_factor = 0.98
    time_to_expiry = 0.5
    volatility = 0.25

    strikes = np.array([80, 90, 100, 110, 120], dtype=float)

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

    inferred_forward, inferred_discount_factor = (
        infer_forward_and_discount_factor(
            call_prices,
            put_prices,
            strikes,
        )
    )

    assert inferred_forward == pytest.approx(forward)
    assert inferred_discount_factor == pytest.approx(discount_factor)


@pytest.mark.parametrize(
    "call_prices, put_prices, strikes",
    [
        ([1.0], [0.5], [100.0]),
        ([1.0, 2.0], [0.5], [100.0, 110.0]),
        ([1.0, 2.0], [0.5, 1.0], [100.0, 100.0]),
        ([1.0, np.nan], [0.5, 1.0], [100.0, 110.0]),
        ([1.0, 2.0], [0.5, 1.0], [100.0, -110.0]),
    ],
)
def test_infer_forward_rejects_invalid_inputs(call_prices, put_prices, strikes):
    with pytest.raises(ValueError):
        infer_forward_and_discount_factor(call_prices, put_prices, strikes)
