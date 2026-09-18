import pytest

from src.implied_vol import implied_volatility_call, implied_volatility_put
from src.pricing import black_scholes_call_price, black_scholes_put_price


def test_implied_volatility_call():
    forward = 100.0
    strike = 110.0
    discount_factor = 0.98
    time_to_expiry = 0.5
    volatility = 0.25

    market_price = black_scholes_call_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    implied_volatility = implied_volatility_call(
        market_price,
        forward,
        strike,
        discount_factor,
        time_to_expiry,
    )

    assert implied_volatility == pytest.approx(volatility)


def test_implied_volatility_expands_upper_bound():
    forward = 100.0
    strike = 110.0
    discount_factor = 0.98
    time_to_expiry = 0.5
    volatility = 8.0

    market_price = black_scholes_call_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    implied_volatility = implied_volatility_call(
        market_price,
        forward,
        strike,
        discount_factor,
        time_to_expiry,
    )

    assert implied_volatility == pytest.approx(volatility)


def test_implied_volatility_call_respects_upper_bound():
    forward = 100.0
    strike = 110.0
    discount_factor = 0.98
    time_to_expiry = 0.5

    market_price = black_scholes_call_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        20.0,
    ) + 1.0

    with pytest.raises(ValueError, match="Could not bracket"):
        implied_volatility_call(
            market_price,
            forward,
            strike,
            discount_factor,
            time_to_expiry,
        )


def test_implied_volatility_put():
    forward = 100.0
    strike = 110.0
    discount_factor = 0.98
    time_to_expiry = 0.5
    volatility = 0.25

    market_price = black_scholes_put_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    implied_volatility = implied_volatility_put(
        market_price,
        forward,
        strike,
        discount_factor,
        time_to_expiry,
    )

    assert implied_volatility == pytest.approx(volatility)


def test_implied_volatility_put_expands_upper_bound():
    forward = 100.0
    strike = 110.0
    discount_factor = 0.98
    time_to_expiry = 0.5
    volatility = 8.0

    market_price = black_scholes_put_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        volatility,
    )

    implied_volatility = implied_volatility_put(
        market_price,
        forward,
        strike,
        discount_factor,
        time_to_expiry,
    )

    assert implied_volatility == pytest.approx(volatility)


def test_implied_volatility_put_respects_upper_bound():
    forward = 100.0
    strike = 110.0
    discount_factor = 0.98
    time_to_expiry = 0.5

    market_price = black_scholes_put_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        20.0,
    ) + 1.0

    with pytest.raises(ValueError, match="Could not bracket"):
        implied_volatility_put(
            market_price,
            forward,
            strike,
            discount_factor,
            time_to_expiry,
        )
