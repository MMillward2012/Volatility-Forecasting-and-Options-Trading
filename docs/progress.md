# Project progress

## `src/pricing.py`

### Implemented

- `black_scholes_call_price(...)` prices a European call using the Black formula.
- `black_scholes_put_price(...)` prices the corresponding European put.
- Inputs are converted to finite, strictly positive NumPy values.

### Mathematical logic

For a forward price $F$, strike $K$, discount factor $D$, time to expiry $\tau = T-t$, and volatility $\sigma$:

$$
d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2\tau}{\sigma\sqrt{\tau}},
\qquad
d_2 = d_1 - \sigma\sqrt{\tau}.
$$

The call and put prices are:

$$
C = D\left(FN(d_1) - KN(d_2)\right),
\qquad
P = D\left(KN(-d_2) - FN(-d_1)\right).
$$

### Tests

`tests/test_pricing.py` covers call-put parity, call-price monotonicity with respect to strike, and rejection of non-positive or non-finite inputs.

## `src/forward.py`

### Implemented

- `infer_forward_and_discount_factor(...)` fits call-minus-put values against strikes to infer the forward price and discount factor.
- Inputs must be one-dimensional, finite, aligned arrays with at least two distinct positive strikes.

### Mathematical logic

For matched European call and put prices under forward pricing:

$$
C - P = D(F - K) = DF - DK.
$$

Therefore, fitting $C-P$ as a linear function of strike gives a slope of $-D$ and an intercept of $DF$. The implementation recovers:

$$
D = -\text{slope},
\qquad
F = \frac{\text{intercept}}{D}.
$$

### Tests

`tests/test_forward.py` covers synthetic recovery of the forward price and discount factor, plus invalid array shapes, lengths, values, and strikes.

## `src/options_chain.py`

### Implemented

- `calculate_mid_price(bid, ask)` calculates the midpoint between bid and ask prices.
- `calculate_time_to_expiry(quote_date, expiry_date)` calculates time to expiry in years using the ACT/365 convention.
- `match_calls_and_puts(option_chain)` calculates quote midpoints and inner-joins calls and puts with the same quote date, expiry date, and strike.
- Both functions support scalar values and NumPy/pandas-friendly inputs where natural.

### Mathematical logic

The mid price is the average of the bid and ask:

$$
M = \frac{\mathrm{bid} + \mathrm{ask}}{2}.
$$

Using ACT/365, time to expiry is the actual number of calendar days between the quote and expiry dates divided by 365:

$$
\tau = \frac{\mathrm{expiry\ date} - \mathrm{quote\ date}}{365}.
$$

The matching function prepares aligned call and put prices for downstream
forward and discount-factor inference. It transforms raw quote rows such as:

| quote_date | expiry_date | strike | option_type | bid | ask |
| --- | --- | ---: | --- | ---: | ---: |
| 2026-01-01 | 2026-07-01 | 80 | call | ... | ... |
| 2026-01-01 | 2026-07-01 | 80 | put | ... | ... |

into one row per matched strike:

| quote_date | expiry_date | strike | call_mid | put_mid |
| --- | --- | ---: | ---: | ---: |
| 2026-01-01 | 2026-07-01 | 80 | ... | ... |

### Tests

`tests/test_options_chain.py` covers scalar and array mid-price calculations, scalar and pandas Series time-to-expiry calculations, complete call-put matching, and exclusion of unmatched strikes. `tests/helper.py` provides synthetic option-chain data for tests only; it is not production functionality.

## `src/implied_vol.py`

### Implemented

- `implied_volatility_call(...)` solves for call implied volatility using `scipy.optimize.brentq`.
- `implied_volatility_put(...)` solves for put implied volatility using the same approach.
- Each solver starts with an upper volatility of $5$, doubles it while the relevant Black option price remains below the market price, and stops at the current safety cap of $20$.

### Mathematical logic

For calls, the solver defines:

$$
f(\sigma)
=
C_{\mathrm{BS}}(\sigma)
-
C_{\mathrm{market}}.
$$

For puts, the equivalent objective is:

$$
f(\sigma)
=
P_{\mathrm{BS}}(\sigma)
-
P_{\mathrm{market}}.
$$

The relevant Black option price is strictly increasing in volatility, so
$f(\sigma)$ is also increasing. Each solver begins with the bracket

$$
\sigma_{\mathrm{lower}} = 10^{-8},
\qquad
\sigma_{\mathrm{upper}} = 5.
$$

If $f(\sigma_{\mathrm{upper}}) < 0$, the model price is still below the
market price, so the required implied volatility must be higher. The upper
bound is therefore doubled until $f(\sigma_{\mathrm{upper}}) \geq 0$ or the
safety cap of $20$ is reached.

Once the objective has opposite signs at the two bounds, Brent's method
solves

$$
f(\sigma)=0.
$$

### Tests

`tests/test_implied_vol.py` covers synthetic call and put implied-volatility recovery at volatilities $0.25$ and $8.0$. The higher-volatility cases exercise upper-bound growth, and additional tests verify that both solvers raise an error when the root cannot be bracketed below the safety cap.

## Test suite

The current test suite covers:

- Black call and put pricing.
- Put-call parity and pricing input validation.
- Synthetic forward and discount-factor recovery.
- Call and put implied-volatility recovery and adaptive bracketing.
- Call and put implied-volatility safety-cap handling.

Run the suite with:

```bash
pytest
```

## Project configuration

- `pytest.ini` configures the project root on the pytest import path.
