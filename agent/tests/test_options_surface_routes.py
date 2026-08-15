"""API tests for ``POST /options/surface`` + ``POST /options/surface/synthetic``.

Deterministic by construction — no network anywhere: both routes run the pure
math in ``src.quantlib.volsurface`` on caller-supplied quotes or fabricated
smile quotes, so happy paths, validation and error mapping are exercised end
to end. Loopback ``TestClient`` (127.0.0.1) bypasses dev-mode auth, matching
the convention in ``test_options_routes.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import api_server
from src.quantlib.volsurface import smile_quotes

_QUOTES = smile_quotes(100.0, [30, 60, 90], [70.0, 85.0, 100.0, 115.0, 130.0],
                       0.25, skew=-0.2, curvature=0.1)


def _client() -> TestClient:
    return TestClient(api_server.app, client=("127.0.0.1", 50000))


@pytest.fixture(autouse=True)
def _dev_mode_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_AUTH_KEY", raising=False)
    monkeypatch.setattr(api_server, "_API_KEY", "")


def _synthetic_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "spot": 100.0,
        "expiry_days": [30, 60, 90],
        "strike_min": 70.0,
        "strike_max": 130.0,
        "strike_points": 25,
        "atm_iv": 0.25,
        "skew": -0.2,
        "curvature": 0.1,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.0,
    }
    body.update(over)
    return body


def _quotes_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "spot": 100.0,
        "quotes": [dict(q) for q in _QUOTES],
        "risk_free_rate": 0.05,
        "dividend_yield": 0.0,
    }
    body.update(over)
    return body


# ── POST /options/surface/synthetic — happy path ───────────────────────────


def test_synthetic_happy_path() -> None:
    r = _client().post("/options/surface/synthetic", json=_synthetic_body())

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["spot"] == 100.0

    assert [e["expiry_days"] for e in body["expiries"]] == [30.0, 60.0, 90.0]
    for e in body["expiries"]:
        assert e["n_quotes"] == 25
        assert e["rmse"] < 1e-3
        assert set(e["params"]) == {"a", "b", "rho", "m0", "sigma"}

    grid = body["grid"]
    assert len(grid["strikes"]) == 25
    assert grid["expiries"] == [30.0, 60.0, 90.0]
    assert len(grid["iv"]) == 3
    assert all(len(row) == 25 for row in grid["iv"])
    assert all(v > 0 for row in grid["iv"] for v in row)

    # No greeks requested ⇒ the field is explicitly null, not absent.
    assert body["greeks"] is None
    assert isinstance(body["limitations"], list)


def test_synthetic_negative_skew_fits_negative_rho() -> None:
    r = _client().post(
        "/options/surface/synthetic", json=_synthetic_body(skew=-0.3)
    )
    assert r.status_code == 200
    body = r.json()
    assert all(e["params"]["rho"] < 0 for e in body["expiries"])


def test_synthetic_greeks_curves_when_requested() -> None:
    r = _client().post(
        "/options/surface/synthetic",
        json=_synthetic_body(greeks_expiry_days=60.0),
    )
    assert r.status_code == 200
    greeks = r.json()["greeks"]
    assert greeks is not None
    assert greeks["expiry_days"] == 60.0
    assert len(greeks["strikes"]) == 25
    for key in ("delta", "gamma", "theta", "vega", "rho"):
        assert len(greeks[key]) == 25, key
    # ATM call delta sits between 0 and 1; near-the-money gamma is positive.
    atm_idx = min(range(len(greeks["strikes"])),
                  key=lambda i: abs(greeks["strikes"][i] - 100.0))
    assert 0.0 < greeks["delta"][atm_idx] < 1.0
    assert greeks["gamma"][atm_idx] > 0.0


# ── POST /options/surface — explicit quotes ────────────────────────────────


def test_quotes_mode_round_trip() -> None:
    r = _client().post("/options/surface", json=_quotes_body())

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert [e["expiry_days"] for e in body["expiries"]] == [30.0, 60.0, 90.0]
    # Grid spans the quote strikes by default.
    grid = body["grid"]
    assert grid["strikes"][0] == pytest.approx(70.0)
    assert grid["strikes"][-1] == pytest.approx(130.0)
    assert grid["expiries"] == [30.0, 60.0, 90.0]
    assert all(v > 0 for row in grid["iv"] for v in row)


def test_quotes_mode_honours_grid_overrides() -> None:
    r = _client().post(
        "/options/surface",
        json=_quotes_body(
            strike_min=80.0,
            strike_max=120.0,
            strike_points=9,
            expiries=[30.0, 90.0],
        ),
    )
    assert r.status_code == 200
    grid = r.json()["grid"]
    assert len(grid["strikes"]) == 9
    assert grid["strikes"][0] == pytest.approx(80.0)
    assert grid["strikes"][-1] == pytest.approx(120.0)
    assert grid["expiries"] == [30.0, 90.0]


def test_quotes_mode_rejects_strike_min_above_max() -> None:
    r = _client().post(
        "/options/surface",
        json=_quotes_body(strike_min=130.0, strike_max=70.0),
    )
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "error"
    assert "strike_min" in body["error"]


# ── Validation (FastAPI 422) ───────────────────────────────────────────────


def test_synthetic_rejects_nonpositive_spot() -> None:
    r = _client().post(
        "/options/surface/synthetic", json=_synthetic_body(spot=0.0)
    )
    assert r.status_code == 422


def test_synthetic_rejects_nonpositive_atm_iv() -> None:
    r = _client().post(
        "/options/surface/synthetic", json=_synthetic_body(atm_iv=0.0)
    )
    assert r.status_code == 422


def test_synthetic_rejects_negative_expiry() -> None:
    r = _client().post(
        "/options/surface/synthetic", json=_synthetic_body(expiry_days=[30, -5])
    )
    assert r.status_code == 422


def test_quotes_mode_rejects_too_few_quotes() -> None:
    r = _client().post(
        "/options/surface",
        json=_quotes_body(quotes=[dict(q) for q in _QUOTES[:2]]),
    )
    assert r.status_code == 422


def test_quotes_mode_rejects_nonpositive_iv() -> None:
    bad = [dict(q) for q in _QUOTES[:4]]
    bad[0]["implied_volatility"] = 0.0
    r = _client().post("/options/surface", json=_quotes_body(quotes=bad))
    assert r.status_code == 422


def test_quotes_mode_rejects_too_small_strike_points() -> None:
    r = _client().post(
        "/options/surface", json=_quotes_body(strike_points=3)
    )
    assert r.status_code == 422


# ── Curated 400 mapping (per-expiry fit failures) ──────────────────────────


def test_quotes_mode_maps_per_expiry_fit_failure_to_400() -> None:
    # 4 quotes total passes the pydantic floor but splits 2+2 across expiries,
    # so neither expiry has enough to fit an SVI smile → curated 400.
    sparse = [
        {"strike": 90.0, "expiry_days": 30.0, "implied_volatility": 0.2},
        {"strike": 110.0, "expiry_days": 30.0, "implied_volatility": 0.25},
        {"strike": 90.0, "expiry_days": 60.0, "implied_volatility": 0.22},
        {"strike": 110.0, "expiry_days": 60.0, "implied_volatility": 0.27},
    ]
    r = _client().post("/options/surface", json=_quotes_body(quotes=sparse))

    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "error"
    assert "at least 4 valid quotes per expiry" in body["error"]
