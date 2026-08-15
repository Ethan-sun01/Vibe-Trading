"""Implied-volatility surface construction: SVI smiles + time interpolation.

The Options Lab shipped analytic payoff/scenario analysis and a live chain, but
the *surface* itself — the object a payoff diagram is drawn on top of — had no
implementation anywhere. This module builds one from option quotes, the same
way a trading desk would:

1. **Per-expiry SVI fit.** For each expiry, the raw SVI parameterisation
   (Gatheral, *The Volatility Surface*) is fitted to the observed
   (strike, implied-volatility) quotes by least squares on annualised IV.
2. **Time interpolation.** Between fitted expiries, total variance
   ``w = sigma_iv^2 * T`` is interpolated linearly in time to the target
   expiry — the standard variance-consistent scheme (linear-in-``w``, not
   linear-in-``sigma``, so ATM variance does not bend backwards). Outside the
   fitted range the nearest fitted expiry is used (documented flat
   extrapolation).
3. **Surface-consistent Greeks.** ``greeks()`` reuses the one Black-Scholes
   implementation from ``src.quantlib.options`` with ``sigma`` taken *from the
   fitted surface*, so the smile bends the Greeks instead of a single flat vol
   pretending it does not exist.

Moneyness convention, fixed here so callers never have to guess: ``m = ln(K/F)``
where ``F = S * exp((r - q) * T)`` is the forward for *that* expiry. Each expiry
is fitted in its own forward-moneyness space; time interpolation evaluates each
fitted smile at the same strike ``K`` (not the same moneyness) and interpolates
the resulting total variances.

The raw SVI form (a, b, rho, m0, sigma) maps moneyness ``m`` to *total*
variance::

    w(m) = a + b * ( rho * (m - m0) + sqrt((m - m0)**2 + sigma**2) )

and the annualised IV at expiry ``T`` is ``sqrt(w(m) / T)``. The
``wings_ok`` flag reports the Gatheral no-arbitrage bound on the wings,
``b * (1 + |rho|) <= 4 / T``, as a diagnostic — the fit is not re-run to
enforce it, because the surface here is an analytic *view* of the quotes, and
silently warping the fit to satisfy the bound would make the view lie about
what was observed.

``smile_quotes()`` fabricates a stylised smile (ATM level + skew + curvature in
log-moneyness) so the demo, the CLI and the tests share one deterministic
quote generator. It is explicitly a synthetic-data tool, not a market feed.

Degenerate input fails fast with a ``ValueError`` naming the exact problem —
a surface built from two quotes or a NaN IV is a chart of noise, not a view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from src.quantlib.options import bs_greeks

__all__ = [
    "svi_total_variance",
    "svi_volatility",
    "fit_svi_expiry",
    "smile_quotes",
]

#: Narrowest annualised IV a quote may carry (0.01%). Anything below is a
#: zero-vol print, and a zero-vol print has no smile to fit.
_MIN_IV = 1e-4
#: Widest annualised IV a quote may carry (300%). Above this is a bad print,
#: not volatility, and it would dominate a least-squares fit.
_MAX_IV = 3.0
#: Fewest quotes needed to fit one SVI smile (5 free parameters).
_MIN_QUOTES_PER_EXPIRY = 4
#: Floor applied to the synthetic smile so deep wings never go non-positive.
_SYNTHETIC_IV_FLOOR = 0.005
#: Key used to bucket quotes by expiry; quotes 30 and 30.000001 days apart
#: belong to the same expiration.
_EXPIRY_ROUND = 3
#: Relative tolerance used to judge ``wings_ok`` against the Gatheral bound.
_WINGS_TOL = 1e-6


def _as_float_array(values: Sequence[float], what: str) -> np.ndarray:
    """Coerce a sequence of finite floats, refusing NaN/Inf with a clear error."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{what} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} must be finite")
    return arr


def _forward(spot: float, T_years: float, risk_free_rate: float,
             dividend_yield: float) -> float:
    """Cost-of-carry forward price ``S * exp((r - q) * T)``."""
    return float(spot * np.exp((risk_free_rate - dividend_yield) * T_years))


def _moneyness(strike: float, forward: float) -> float:
    """Log-moneyness ``ln(K / F)``; strikes at the forward are moneyness 0."""
    return float(np.log(strike / forward))


def svi_total_variance(m: float, a: float, b: float, rho: float, m0: float,
                       sigma: float) -> float:
    """Raw SVI total variance ``w(m)`` at log-moneyness ``m``.

    Args:
        m: Log-moneyness ``ln(K/F)``.
        a: Vertical translation (ATM total-variance level).
        b: Overall slope of the wings.
        rho: Skew direction; negative rho prices puts above calls.
        m0: Moneyness of the smile minimum.
        sigma: Curvature around the minimum.

    Returns:
        Total implied variance ``sigma_iv**2 * T``. A float for scalar input;
        a numpy array when ``m`` is an array (the SVI fitter evaluates the
        whole smile at once).
    """
    value = a + b * (rho * (m - m0) + np.sqrt((m - m0) ** 2 + sigma ** 2))
    if np.ndim(value) == 0:
        return float(value)
    return value


def svi_volatility(m: float, T_years: float, a: float, b: float, rho: float,
                   m0: float, sigma: float) -> float:
    """Annualised implied volatility from a raw SVI smile.

    Args:
        m: Log-moneyness ``ln(K/F)``.
        T_years: Time to expiry in years; must be positive.
        a, b, rho, m0, sigma: Raw SVI parameters (see
            :func:`svi_total_variance`).

    Returns:
        ``sqrt(w(m) / T_years)``.

    Raises:
        ValueError: If ``T_years`` is not positive.
    """
    if T_years <= 0:
        raise ValueError(f"T_years must be > 0, got {T_years}")
    w = svi_total_variance(m, a, b, rho, m0, sigma)
    if w <= 0:
        # Total variance cannot be negative; a non-positive value means the
        # parameter point is outside SVI's valid half-plane (b < 0 or a < 0).
        raise ValueError(
            f"raw SVI total variance must be positive, got {w}; "
            "check that a >= 0 and b >= 0"
        )
    return float(np.sqrt(w / T_years))


@dataclass(frozen=True)
class SviFit:
    """One fitted raw-SVI smile for a single expiry."""

    expiry_days: float
    """Expiration in calendar days (as bucketed by the fitter)."""
    T_years: float
    """Expiration in years (``expiry_days / 365``)."""
    n_quotes: int
    """Number of quotes the fit was run on (after dropping non-finite/zero)."""
    params: Dict[str, float] = field(default_factory=dict)
    """Fitted ``a``/``b``/``rho``/``m0``/``sigma``."""
    rmse: float = 0.0
    """Root-mean-square IV residual in annualised-vol points."""
    max_residual: float = 0.0
    """Worst single-quote IV residual in annualised-vol points."""
    wings_ok: bool = True
    """Gatheral wing bound ``b*(1+|rho|) <= 4/T`` holds on this fit."""


def fit_svi_expiry(
    quotes: Sequence[Tuple[float, float]],
    T_years: float,
    spot: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> SviFit:
    """Fit a raw-SVI smile to ``(strike, implied_volatility)`` quotes.

    Args:
        quotes: Sequence of ``(strike, implied_volatility)`` pairs for one
            expiry. At least :data:`_MIN_QUOTES_PER_EXPIRY` valid quotes are
            required — 5 free parameters cannot be pinned by 3 points.
        T_years: Time to expiry in years; must be positive.
        spot: Underlying spot used to build the forward.
        risk_free_rate: Continuously compounded annual rate (default 0.05).
        dividend_yield: Continuous dividend yield (default 0).

    Returns:
        :class:`SviFit` with fitted parameters and fit diagnostics. Non-finite
        or non-positive IV quotes are dropped *before* validation, so one bad
        print cannot poison an otherwise healthy expiry.

    Raises:
        ValueError: If ``T_years``/``spot`` are not positive, fewer than
            :data:`_MIN_QUOTES_PER_EXPIRY` valid quotes remain, all quotes are
            degenerate (identical moneyness), or the least-squares fit fails to
            converge.
    """
    if T_years <= 0:
        raise ValueError(f"T_years must be > 0, got {T_years}")
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")

    strikes: list[float] = []
    ivs: list[float] = []
    for strike, iv in quotes:
        if not np.isfinite(strike) or strike <= 0:
            continue
        if not np.isfinite(iv) or iv <= 0:
            continue
        strikes.append(float(strike))
        ivs.append(float(iv))
    if len(strikes) < _MIN_QUOTES_PER_EXPIRY:
        raise ValueError(
            f"need at least {_MIN_QUOTES_PER_EXPIRY} valid quotes per expiry "
            f"to fit an SVI smile, got {len(strikes)}"
        )

    forward = _forward(spot, T_years, risk_free_rate, dividend_yield)
    m = np.array([_moneyness(k, forward) for k in strikes], dtype=float)
    iv_obs = np.array(ivs, dtype=float)
    if np.ptp(m) < 1e-12:
        raise ValueError("all quotes share the same moneyness; a smile needs spread")

    # Initial guess, data-driven so the fit starts near the answer: m0 at the
    # quoted smile minimum, b sized to the observed IV spread, rho signed by
    # the moneyness/IV correlation, a at the ATM total variance.
    lowest = int(np.argmin(iv_obs))
    m0_init = float(m[lowest])
    sigma_init = max(float(np.ptp(m)) * 0.5, 0.05)
    wing_span = float(np.max(np.sqrt((m - m0_init) ** 2 + sigma_init ** 2)))
    b_init = max(float(np.ptp(iv_obs) * np.sqrt(T_years) / max(wing_span, 1e-9)), 1e-4)
    # Correlation is undefined on a constant axis (flat smile); np.corrcoef
    # then divides by zero stddev and warns, so guard the axes first.
    m_std = float(np.std(m))
    iv_std = float(np.std(iv_obs))
    if m_std > 1e-12 and iv_std > 1e-12:
        corr = np.corrcoef(m, iv_obs)[0, 1]
        rho_init = float(np.clip(-corr, -0.5, 0.5)) if np.isfinite(corr) else 0.0
    else:
        rho_init = 0.0
    median_iv = float(np.median(iv_obs))
    init = np.array([
        median_iv ** 2 * T_years,  # a: ATM total variance
        b_init,                    # b
        rho_init,                  # rho
        m0_init,                   # m0
        sigma_init,                # sigma
    ])
    lower = np.array([1e-8, 1e-8, -0.999, float(m.min()) - 1.0, 1e-6])
    upper = np.array([10.0, 10.0, 0.999, float(m.max()) + 1.0, 5.0])

    def residual(theta: np.ndarray) -> np.ndarray:
        a, b, rho, m0, sigma = theta
        w = svi_total_variance(m, a, b, rho, m0, sigma)
        return np.sqrt(np.maximum(w, 1e-12) / T_years) - iv_obs

    # max_nfev: raw-SVI least squares routinely needs a few thousand evals to
    # thread the wing parameters; the default budget ends mid-descent.
    result = least_squares(residual, init, bounds=(lower, upper), max_nfev=10000)
    if not result.success:
        raise ValueError(
            f"SVI fit failed to converge for T={T_years:.4f}: "
            f"{result.message}"
        )

    a, b, rho, m0, sigma = (float(v) for v in result.x)
    fitted = np.sqrt(
        np.maximum(svi_total_variance(m, a, b, rho, m0, sigma), 1e-12) / T_years
    )
    resid = fitted - iv_obs
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    wings_ok = bool(b * (1.0 + abs(rho)) <= 4.0 / T_years + _WINGS_TOL)
    return SviFit(
        expiry_days=0.0,  # filled by the caller, which owns the bucketing
        T_years=T_years,
        n_quotes=len(strikes),
        params={"a": a, "b": b, "rho": rho, "m0": m0, "sigma": sigma},
        rmse=rmse,
        max_residual=float(np.max(np.abs(resid))),
        wings_ok=wings_ok,
    )


def smile_quotes(
    spot: float,
    expiry_days: Sequence[float],
    strikes: Sequence[float],
    atm_iv: float,
    skew: float = 0.0,
    curvature: float = 0.0,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> list[Dict[str, float]]:
    """Fabricate a stylised IV smile over a strike/expiry grid.

    The generated IV is a quadratic in log-moneyness around the forward::

        iv(m) = atm_iv * (1 + skew * m + curvature * m**2)

    clamped to :data:`_SYNTHETIC_IV_FLOOR` on the low side and
    :data:`_MAX_IV` on the high side. This is an *explicitly synthetic* quote
    generator — it exists so the demo, CLI and tests share one deterministic
    source of quotes and the SVI fit has something well-behaved to fit.

    Args:
        spot: Underlying spot price; must be positive.
        expiry_days: One or more expirations in calendar days.
        strikes: Strike grid.
        atm_iv: Annualised IV at the forward (moneyness 0), e.g. 0.25.
        skew: Linear log-moneyness coefficient (negative = puts priced above
            calls, the usual equity skew).
        curvature: Quadratic log-moneyness coefficient.
        risk_free_rate: Continuously compounded annual rate.
        dividend_yield: Continuous dividend yield.

    Returns:
        List of ``{"strike", "expiry_days", "implied_volatility"}`` dicts, one
        per (expiry, strike) combination, ordered by expiry then strike.

    Raises:
        ValueError: If ``spot``, ``atm_iv`` or any expiry/strike is not
            positive and finite.
    """
    if spot <= 0 or not np.isfinite(spot):
        raise ValueError(f"spot must be positive and finite, got {spot}")
    if atm_iv <= 0 or not np.isfinite(atm_iv):
        raise ValueError(f"atm_iv must be positive and finite, got {atm_iv}")
    expiries = _as_float_array(list(expiry_days), "expiry_days")
    strikes_arr = _as_float_array(list(strikes), "strikes")
    if np.any(expiries <= 0):
        raise ValueError("expiry_days must be positive")
    if np.any(strikes_arr <= 0):
        raise ValueError("strikes must be positive")

    quotes: list[Dict[str, float]] = []
    for T_days in expiries:
        T_years = float(T_days) / 365.0
        forward = _forward(spot, T_years, risk_free_rate, dividend_yield)
        for K in strikes_arr:
            m = _moneyness(float(K), forward)
            iv = atm_iv * (1.0 + skew * m + curvature * m ** 2)
            iv = min(max(iv, _SYNTHETIC_IV_FLOOR), _MAX_IV)
            quotes.append({
                "strike": float(K),
                "expiry_days": float(T_days),
                "implied_volatility": float(iv),
            })
    return quotes


@dataclass(frozen=True)
class SurfaceExpiry:
    """A fitted expiry inside a :class:`VolSurface`."""

    expiry_days: float
    T_years: float
    fit: SviFit


class VolSurface:
    """A fitted implied-volatility surface.

    Built from ``(strike, expiry_days, implied_volatility)`` quotes: quotes are
    bucketed by expiry, each bucket is fitted with raw SVI, and time
    interpolation runs in total-variance space. Not part of ``__all__`` — the
    pure-compute surface (``quantlib_call``) exposes the functions, the REST
    route and the Web UI use this class.
    """

    def __init__(
        self,
        quotes: Sequence[Dict[str, float]],
        spot: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ) -> None:
        """Fit a surface from quote dicts.

        Args:
            quotes: Sequence of ``{"strike", "expiry_days",
                "implied_volatility"}`` dicts. At least one expiry with at
                least :data:`_MIN_QUOTES_PER_EXPIRY` valid quotes is required.
            spot: Underlying spot price.
            risk_free_rate: Continuously compounded annual rate.
            dividend_yield: Continuous dividend yield.

        Raises:
            ValueError: On empty/invalid input or when no expiry has enough
                quotes to fit.
        """
        if spot <= 0 or not np.isfinite(spot):
            raise ValueError(f"spot must be positive and finite, got {spot}")

        buckets: dict[float, list[Tuple[float, float]]] = {}
        for quote in quotes:
            try:
                strike = float(quote["strike"])
                expiry_days = float(quote["expiry_days"])
                iv = float(quote["implied_volatility"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed surface quote {quote!r}: {exc}") from exc
            if not np.isfinite(strike) or strike <= 0:
                continue
            if not np.isfinite(iv) or iv <= 0:
                continue
            if not np.isfinite(expiry_days) or expiry_days <= 0:
                continue
            key = round(expiry_days, _EXPIRY_ROUND)
            buckets.setdefault(key, []).append((strike, iv))

        if not buckets:
            raise ValueError("no valid quotes to build a surface from")

        self.spot = float(spot)
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yield = float(dividend_yield)
        self.expiries: list[SurfaceExpiry] = []
        for expiry_days in sorted(buckets):
            T_years = expiry_days / 365.0
            fit = fit_svi_expiry(
                buckets[expiry_days], T_years, self.spot,
                self.risk_free_rate, self.dividend_yield,
            )
            self.expiries.append(
                SurfaceExpiry(expiry_days=expiry_days, T_years=T_years, fit=fit)
            )

    @property
    def fitted_expiry_days(self) -> list[float]:
        """Sorted expiration days of the fitted expiries."""
        return [e.expiry_days for e in self.expiries]

    def _fitted_iv(self, expiry: SurfaceExpiry, strike: float) -> float:
        """Annualised IV of one fitted smile at a strike."""
        forward = _forward(self.spot, expiry.T_years, self.risk_free_rate,
                           self.dividend_yield)
        m = _moneyness(strike, forward)
        p = expiry.fit.params
        w = svi_total_variance(m, p["a"], p["b"], p["rho"], p["m0"], p["sigma"])
        return float(np.sqrt(max(w, 1e-12) / expiry.T_years))

    def iv(self, strike: float, expiry_days: float) -> float:
        """Annualised IV at an arbitrary (strike, expiry).

        Between fitted expiries, total variance is interpolated linearly in
        time; outside the fitted range the nearest fitted expiry is used (flat
        extrapolation, reported by :meth:`limitations`).

        Args:
            strike: Target strike.
            expiry_days: Target expiration in calendar days.

        Returns:
            Annualised implied volatility.
        """
        if strike <= 0 or not np.isfinite(strike):
            raise ValueError(f"strike must be positive and finite, got {strike}")
        if expiry_days <= 0 or not np.isfinite(expiry_days):
            raise ValueError(
                f"expiry_days must be positive and finite, got {expiry_days}"
            )
        T_target = expiry_days / 365.0

        if len(self.expiries) == 1:
            return self._fitted_iv(self.expiries[0], strike)

        # Outside the fitted range: hold flat at the nearest fitted expiry.
        # (Interpolating with a clamped weight would divide that expiry's
        # total variance by the wrong T and report a distorted IV.)
        if T_target <= self.expiries[0].T_years:
            return self._fitted_iv(self.expiries[0], strike)
        if T_target >= self.expiries[-1].T_years:
            return self._fitted_iv(self.expiries[-1], strike)

        # Locate the bracketing fitted expiries (strictly between them now).
        idx = 0
        while (
            idx < len(self.expiries) - 1
            and self.expiries[idx + 1].T_years < T_target
        ):
            idx += 1
        lo = self.expiries[idx]
        hi = self.expiries[idx + 1]

        w_lo = self._fitted_iv(lo, strike) ** 2 * lo.T_years
        w_hi = self._fitted_iv(hi, strike) ** 2 * hi.T_years
        span = hi.T_years - lo.T_years
        if span <= 0:
            return self._fitted_iv(lo, strike)
        t = min(max((T_target - lo.T_years) / span, 0.0), 1.0)
        w = w_lo + (w_hi - w_lo) * t
        return float(np.sqrt(max(w, 1e-12) / T_target))

    def greeks(self, strike: float, expiry_days: float,
               option_type: str = "call") -> Dict[str, float]:
        """Black-Scholes Greeks priced at the *surface* IV for this strike.

        The five Greeks come from :func:`src.quantlib.options.bs_greeks` with
        ``sigma`` set to :meth:`iv` — so a skew that prices puts above calls
        shows up in the delta/gamma curves instead of being flattened by one
        ATM vol. Conventions are ``bs_greeks``' native ones: theta per calendar
        day, vega/rho per 1 percentage point.

        Args:
            strike: Target strike.
            expiry_days: Target expiration in calendar days.
            option_type: ``"call"`` or ``"put"`` (case-insensitive).

        Returns:
            Dict with ``delta``, ``gamma``, ``theta``, ``vega``, ``rho``.
        """
        sigma = self.iv(strike, expiry_days)
        return bs_greeks(
            S=self.spot,
            K=strike,
            T=expiry_days / 365.0,
            r=self.risk_free_rate,
            sigma=sigma,
            option_type=option_type,
            q=self.dividend_yield,
        )

    def grid(self, strikes: Sequence[float],
             expiries_days: Sequence[float]) -> list[list[float]]:
        """Evaluate the surface on a strike × expiry grid.

        Args:
            strikes: Strike axis.
            expiries_days: Expiration axis in calendar days.

        Returns:
            Rows of annualised IV, one row per expiry, each aligned with
            ``strikes``.
        """
        strikes_arr = _as_float_array(list(strikes), "strikes")
        expiries_arr = _as_float_array(list(expiries_days), "expiries_days")
        if np.any(strikes_arr <= 0):
            raise ValueError("strikes must be positive")
        if np.any(expiries_arr <= 0):
            raise ValueError("expiries_days must be positive")
        return [
            [self.iv(float(K), float(T_days))
             for K in strikes_arr]
            for T_days in expiries_arr
        ]

    def limitations(self) -> list[str]:
        """Human-readable limits of this surface's construction.

        Currently only reports when time extrapolation is in effect, because
        flat extrapolation is the one place the surface silently invents
        numbers outside the observed data.
        """
        if len(self.expiries) < 2:
            return ["only one expiry was fitted; the surface is flat in time"]
        return [
            "volatility outside the fitted expiry range is held flat at the "
            "nearest fitted expiry",
        ]
