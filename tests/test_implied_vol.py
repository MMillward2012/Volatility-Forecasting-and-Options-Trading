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
    time_to_expiry = 0.01

    market_price = black_scholes_call_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        30.0,
    )

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
    time_to_expiry = 0.01

    market_price = black_scholes_put_price(
        forward,
        strike,
        discount_factor,
        time_to_expiry,
        30.0,
    )

    with pytest.raises(ValueError, match="Could not bracket"):
        implied_volatility_put(
            market_price,
            forward,
            strike,
            discount_factor,
            time_to_expiry,
        )


@pytest.mark.parametrize("solver,pricer", [
    (implied_volatility_call, black_scholes_call_price),
    (implied_volatility_put, black_scholes_put_price),
])
def test_implied_volatility_recovers_at_cap(solver, pricer):
    market_price = pricer(100.0, 110.0, 0.98, 0.01, 20.0)

    assert solver(market_price, 100.0, 110.0, 0.98, 0.01) == pytest.approx(20.0)


@pytest.mark.parametrize("solver,market_price", [
    (implied_volatility_call, -1.0),
    (implied_volatility_call, 98.0),
    (implied_volatility_call, 99.0),
    (implied_volatility_put, 9.0),
    (implied_volatility_put, 0.98 * 110.0),
    (implied_volatility_put, 109.0),
])
def test_implied_volatility_rejects_prices_outside_bounds(solver, market_price):
    with pytest.raises(ValueError, match="Black price bounds"):
        solver(market_price, 100.0, 110.0, 0.98, 2.0)


@pytest.mark.parametrize("solver", [implied_volatility_call, implied_volatility_put])
@pytest.mark.parametrize("market_price", [float("nan"), float("inf"), -float("inf")])
def test_implied_volatility_rejects_non_finite_prices(solver, market_price):
    with pytest.raises(ValueError, match="finite"):
        solver(market_price, 100.0, 110.0, 0.98, 0.5)


@pytest.mark.parametrize("solver,forward,strike,intrinsic", [
    (implied_volatility_call, 110.0, 100.0, 0.98 * 10.0),
    (implied_volatility_call, 100.0, 100.0, 0.0),
    (implied_volatility_call, 100.0, 110.0, 0.0),
    (implied_volatility_put, 100.0, 110.0, 0.98 * 10.0),
    (implied_volatility_put, 100.0, 100.0, 0.0),
    (implied_volatility_put, 110.0, 100.0, 0.0),
])
def test_implied_volatility_returns_zero_at_intrinsic(solver, forward, strike, intrinsic):
    assert solver(intrinsic, forward, strike, 0.98, 0.5) == 0.0


@pytest.mark.parametrize("solver", [implied_volatility_call, implied_volatility_put])
@pytest.mark.parametrize("parameter", ["forward", "strike", "discount_factor", "time_to_expiry"])
def test_implied_volatility_validates_inputs_at_intrinsic(solver, parameter):
    inputs = dict(forward=100.0, strike=100.0, discount_factor=0.98, time_to_expiry=0.5)
    inputs[parameter] = 0.0

    with pytest.raises(ValueError, match=parameter):
        solver(0.0, **inputs)
