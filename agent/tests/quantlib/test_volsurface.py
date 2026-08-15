"""Pin the SVI vol-surface module in ``src.quantlib.volsurface``.

Three independent kinds of check, mirroring the style of ``test_options.py``:

  * Analytic values of the raw-SVI formula itself (``w(m0) = a + b*sigma``,
    the IV/``sqrt(w/T)`` link) so a self-consistently wrong formula fails.
  * Fit round-trips: the quadratic smile generator is exactly the kind of
    well-behaved surface raw SVI exists to fit, so a fit that cannot recover
    it to ~0.1 vol point is broken — and recovering it also pins the
    parameterisation (a, b, rho, m0, sigma) to the generator's true shape.
  * Surface semantics: variance-consistent time interpolation (flat smile ⇒
    flat IV at every expiry), bracketed interpolation between fitted
    expiries, surface-consistent Greeks that equal ``bs_greeks`` fed the
    surface IV, and fast-fail validation on degenerate input.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.quantlib.options import bs_greeks
from src.quantlib.volsurface import (
    VolSurface,
    fit_svi_expiry,
    smile_quotes,
    svi_total_variance,
    svi_volatility,
)

_SPOT = 100.0
_R = 0.05


def _smile(expiries=(30, 60, 90), strikes=None, atm=0.25, skew=-0.2,
           curvature=0.1):
    """Deterministic quadratic-smile quote set used across the suite."""
    if strikes is None:
        strikes = np.linspace(70, 130, 25)
    return smile_quotes(_SPOT, list(expiries), list(strikes), atm,
                        skew=skew, curvature=curvature)


def _pairs(quotes, expiry_days):
    return [(q["strike"], q["implied_volatility"])
            for q in quotes if q["expiry_days"] == expiry_days]


class TestSviFormula:
    """Analytic pins on the raw-SVI maths."""

    def test_total_variance_at_minimum(self) -> None:
        # At m == m0 the sqrt collapses to sigma: w = a + b*sigma.
        assert svi_total_variance(0.0, 0.05, 0.1, -0.4, 0.0, 0.15) == pytest.approx(
            0.05 + 0.1 * 0.15, abs=1e-12
        )

    def test_total_variance_symmetric_in_m0(self) -> None:
        # Shifting m0 shifts the smile rigidly in moneyness.
        w = svi_total_variance(0.3, 0.05, 0.1, -0.4, 0.1, 0.15)
        assert w == pytest.approx(
            svi_total_variance(0.2, 0.05, 0.1, -0.4, 0.0, 0.15), abs=1e-12
        )

    def test_volatility_is_sqrt_w_over_t(self) -> None:
        w = svi_total_variance(0.05, 0.04, 0.2, -0.3, 0.0, 0.25)
        T = 0.25
        assert svi_volatility(0.05, T, 0.04, 0.2, -0.3, 0.0, 0.25) == pytest.approx(
            math.sqrt(w / T), abs=1e-12
        )

    def test_volatility_raises_on_nonpositive_T(self) -> None:
        with pytest.raises(ValueError):
            svi_volatility(0.0, 0.0, 0.05, 0.1, 0.0, 0.0, 0.15)

    def test_volatility_raises_on_negative_total_variance(self) -> None:
        # b < 0 is outside SVI's valid half-plane; at m == m0, w = a + b*sigma,
        # and with b = -1 the total variance goes negative.
        with pytest.raises(ValueError):
            svi_volatility(0.0, 0.25, 0.05, -1.0, 0.0, 0.0, 0.15)


class TestSmileQuotes:
    """The deterministic synthetic-quote generator."""

    def test_flat_smile_is_exactly_atm_everywhere(self) -> None:
        quotes = _smile(atm=0.25, skew=0.0, curvature=0.0)
        assert quotes
        assert all(q["implied_volatility"] == pytest.approx(0.25, abs=1e-12)
                   for q in quotes)

    def test_negative_skew_prices_puts_above_calls(self) -> None:
        quotes = _smile(skew=-0.2, curvature=0.0)
        put = [q for q in quotes if q["strike"] < _SPOT]
        call = [q for q in quotes if q["strike"] > _SPOT]
        assert put and call
        assert np.mean([q["implied_volatility"] for q in put]) > np.mean(
            [q["implied_volatility"] for q in call]
        )

    def test_grid_ordering_and_count(self) -> None:
        expiries = [30, 60]
        strikes = [80.0, 100.0, 120.0]
        quotes = _smile(expiries=expiries, strikes=strikes)
        assert len(quotes) == len(expiries) * len(strikes)
        # Ordered by expiry, then strike.
        assert [q["strike"] for q in quotes[:3]] == strikes
        assert quotes[3]["expiry_days"] == 60.0

    def test_clamps_extreme_wings(self) -> None:
        quotes = _smile(strikes=np.linspace(1.0, 500.0, 20), atm=0.25,
                        skew=0.0, curvature=0.5)
        ivs = [q["implied_volatility"] for q in quotes]
        assert min(ivs) >= 0.005 - 1e-12
        assert max(ivs) <= 3.0 + 1e-12

    def test_rejects_nonpositive_inputs(self) -> None:
        with pytest.raises(ValueError):
            smile_quotes(0.0, [30], [100], 0.25)
        with pytest.raises(ValueError):
            smile_quotes(100.0, [30], [100], 0.0)
        with pytest.raises(ValueError):
            smile_quotes(100.0, [-30], [100], 0.25)
        with pytest.raises(ValueError):
            smile_quotes(100.0, [30], [0.0], 0.25)


class TestFitSviExpiry:
    """The per-expiry raw-SVI fitter."""

    def test_round_trip_recovers_quadratic_smile(self) -> None:
        for expiry_days in (30, 60, 90):
            fit = fit_svi_expiry(_pairs(_smile(), expiry_days),
                                 expiry_days / 365.0, _SPOT)
            assert fit.n_quotes == 25
            assert fit.rmse < 1e-3, f"expiry {expiry_days}: rmse {fit.rmse}"
            assert fit.max_residual < 2e-3

    def test_round_trip_recovers_parameter_shape(self) -> None:
        fit = fit_svi_expiry(_pairs(_smile(), 60), 60 / 365.0, _SPOT)
        p = fit.params
        assert p["a"] >= 0.0
        assert p["b"] >= 0.0
        assert -1.0 < p["rho"] < 1.0
        assert p["sigma"] > 0.0
        # Negative skew input ⇒ negative fitted rho (puts above calls).
        assert p["rho"] < 0.0

    def test_wings_ok_on_sane_smile(self) -> None:
        fit = fit_svi_expiry(_pairs(_smile(), 30), 30 / 365.0, _SPOT)
        assert fit.wings_ok is True

    def test_insufficient_quotes_raise(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            fit_svi_expiry([(90.0, 0.2), (100.0, 0.25), (110.0, 0.3)],
                           30 / 365.0, _SPOT)

    def test_nonfinite_quotes_are_dropped_not_fatal(self) -> None:
        quotes = _pairs(_smile(), 30)
        poisoned = quotes[:4] + [(120.0, float("nan")), (75.0, float("inf"))]
        fit = fit_svi_expiry(poisoned, 30 / 365.0, _SPOT)
        assert fit.n_quotes == 4  # only the valid prefix survived
        assert math.isfinite(fit.rmse)

    def test_identical_moneyness_raises(self) -> None:
        with pytest.raises(ValueError, match="same moneyness"):
            fit_svi_expiry([(100.0, 0.2), (100.0, 0.25), (100.0, 0.3),
                            (100.0, 0.35)], 30 / 365.0, _SPOT)

    def test_nonpositive_T_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_svi_expiry(_pairs(_smile(), 30), 0.0, _SPOT)


class TestVolSurface:
    """Surface semantics: interpolation, Greeks, validation."""

    def test_grid_dimensions_and_finite_values(self) -> None:
        surf = VolSurface(_smile(), _SPOT)
        grid = surf.grid([80, 90, 100, 110, 120], [30, 60, 90])
        assert len(grid) == 3
        assert all(len(row) == 5 for row in grid)
        assert all(math.isfinite(v) for row in grid for v in row)

    def test_atm_iv_close_to_generator_level(self) -> None:
        surf = VolSurface(_smile(atm=0.25), _SPOT)
        assert surf.iv(_SPOT, 60.0) == pytest.approx(0.25, abs=2e-3)

    def test_flat_smile_is_flat_in_time(self) -> None:
        # Variance interpolation: w = iv^2*T, so a flat IV surface must stay
        # flat at every expiry — the property the scheme exists to preserve.
        surf = VolSurface(_smile(atm=0.25, skew=0.0, curvature=0.0), _SPOT)
        for expiry in (30, 45, 60, 90, 180):
            assert surf.iv(100.0, expiry) == pytest.approx(0.25, abs=5e-3)

    def test_interpolated_iv_is_bracketed_by_neighbours(self) -> None:
        surf = VolSurface(_smile(), _SPOT)
        strike = 95.0
        lo = surf.iv(strike, 30.0)
        hi = surf.iv(strike, 60.0)
        mid = surf.iv(strike, 45.0)
        assert min(lo, hi) - 1e-9 <= mid <= max(lo, hi) + 1e-9

    def test_out_of_range_expiry_holds_flat(self) -> None:
        surf = VolSurface(_smile(), _SPOT)
        # 5 days and 3650 days both clamp to the nearest fitted expiry.
        assert surf.iv(100.0, 5.0) == pytest.approx(surf.iv(100.0, 30.0), abs=1e-9)
        assert surf.iv(100.0, 3650.0) == pytest.approx(surf.iv(100.0, 90.0), abs=1e-9)

    def test_single_expiry_is_flat_in_time(self) -> None:
        surf = VolSurface(_smile(expiries=(60,)), _SPOT)
        assert surf.iv(100.0, 60.0) == pytest.approx(surf.iv(100.0, 120.0), abs=1e-9)
        assert "flat in time" in " ".join(surf.limitations())

    def test_greeks_equal_bs_greeks_at_surface_iv(self) -> None:
        surf = VolSurface(_smile(), _SPOT)
        for strike in (90.0, 100.0, 110.0):
            sigma = surf.iv(strike, 60.0)
            expected = bs_greeks(_SPOT, strike, 60.0 / 365.0, _R, sigma, "call")
            assert surf.greeks(strike, 60.0) == pytest.approx(expected, abs=1e-12)

    def test_greeks_put_call_delta_identity(self) -> None:
        # call delta - put delta == 1 must survive the smile.
        surf = VolSurface(_smile(), _SPOT)
        for strike in (90.0, 100.0, 110.0):
            c = surf.greeks(strike, 30.0, "call")["delta"]
            p = surf.greeks(strike, 30.0, "put")["delta"]
            assert c - p == pytest.approx(1.0, abs=1e-9)

    def test_smiled_skew_bends_greeks(self) -> None:
        # A skewed surface must price an OTM call's delta differently from a
        # flat surface at the same ATM vol — otherwise the smile is decorative.
        skewed = VolSurface(_smile(atm=0.25, skew=-0.4), _SPOT)
        flat = VolSurface(_smile(atm=0.25, skew=0.0, curvature=0.0), _SPOT)
        d_skewed = skewed.greeks(115.0, 30.0, "call")["delta"]
        d_flat = flat.greeks(115.0, 30.0, "call")["delta"]
        assert d_skewed != pytest.approx(d_flat, abs=1e-6)

    def test_malformed_quote_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="malformed"):
            VolSurface([{"strike": 100.0, "expiry_days": 30.0}], _SPOT)

    def test_empty_quotes_raise(self) -> None:
        with pytest.raises(ValueError, match="no valid quotes"):
            VolSurface([], _SPOT)

    def test_nonpositive_spot_raises(self) -> None:
        with pytest.raises(ValueError):
            VolSurface(_smile(), 0.0)

    def test_fitted_expiry_days_sorted(self) -> None:
        surf = VolSurface(_smile(expiries=(90, 30, 60)), _SPOT)
        assert surf.fitted_expiry_days == [30.0, 60.0, 90.0]
