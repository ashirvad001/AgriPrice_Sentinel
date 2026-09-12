"""
tests/test_input_validation.py
──────────────────────────────
Malicious / malformed input tests for every API endpoint.

Ensures that SQL-injection-style strings, oversized payloads, wrong types,
extra fields, and XSS payloads are cleanly rejected with 422 (not 500 or 200).
"""

import importlib
import os
import sys
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Environment & module reload helpers ──────────────────────────────────────

_BASE_ENV = {
    "SECRET_KEY": "a" * 64,
    "DATABASE_URL": "sqlite+aiosqlite:///./test_validation.db",
}

_MODULES_TO_CLEAR = [
    "app.api.routes_prices",
    "app.api.routes_forecast",
    "app.api.routes_shap",
    "app.api.routes_alerts",
    "app.api.routes_whatsapp",
    "app.api.auth",
    "app.api.deps",
    "app.api.schemas",
    "app.database",
    "app.logger",
    "app.config",
    "app.services.forecast_service",
    "app.repositories.forecast_repository",
]


def _clear_modules() -> None:
    for mod in _MODULES_TO_CLEAR:
        sys.modules.pop(mod, None)


def _build_test_app() -> FastAPI:
    """Build a FastAPI app with all route routers attached and DB overridden."""
    with patch.dict(os.environ, _BASE_ENV, clear=False):
        _clear_modules()
        deps = importlib.import_module("app.api.deps")
        prices = importlib.import_module("app.api.routes_prices")
        forecast = importlib.import_module("app.api.routes_forecast")
        shap = importlib.import_module("app.api.routes_shap")
        alerts = importlib.import_module("app.api.routes_alerts")
        auth = importlib.import_module("app.api.auth")
        whatsapp = importlib.import_module("app.api.routes_whatsapp")

        app = FastAPI()
        app.include_router(prices.router)
        app.include_router(forecast.router)
        app.include_router(shap.router)
        app.include_router(alerts.router)
        app.include_router(auth.router)
        app.include_router(whatsapp.router)

        async def _override_db():
            yield AsyncMock()

        app.dependency_overrides[deps.get_db] = _override_db

        # Override get_current_user for alert tests
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.phone = "9876543210"

        async def _override_current_user():
            return mock_user

        app.dependency_overrides[deps.get_current_user] = _override_current_user

        return app


@pytest.fixture(scope="module")
def client():
    app = _build_test_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
#  SQL INJECTION PAYLOADS — must all return 422
# ═══════════════════════════════════════════════════════════════════════════════

SQL_INJECTION_PAYLOADS = [
    "' OR 1=1 --",
    "'; DROP TABLE raw_prices; --",
    "1 UNION SELECT * FROM users --",
    "admin'--",
    "' OR ''='",
    "1; SELECT * FROM users",
    "' AND 1=CONVERT(int, (SELECT TOP 1 phone FROM users))--",
]


class TestSQLInjectionInPathParams:
    """SQL injection strings in crop/mandi path params must be rejected."""

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_prices_crop_sql_injection(self, client, payload):
        resp = client.get(f"/api/v1/prices/{payload}/Azadpur")
        assert resp.status_code == 422, f"Expected 422 for SQL payload in crop, got {resp.status_code}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_prices_mandi_sql_injection(self, client, payload):
        resp = client.get(f"/api/v1/prices/wheat/{payload}")
        assert resp.status_code == 422, f"Expected 422 for SQL payload in mandi, got {resp.status_code}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_forecast_crop_sql_injection(self, client, payload):
        resp = client.get(f"/api/v1/forecast/{payload}/Azadpur")
        assert resp.status_code == 422, f"Expected 422 for SQL payload in crop, got {resp.status_code}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_forecast_mandi_sql_injection(self, client, payload):
        resp = client.get(f"/api/v1/forecast/wheat/{payload}")
        assert resp.status_code == 422, f"Expected 422 for SQL payload in mandi, got {resp.status_code}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_shap_crop_sql_injection(self, client, payload):
        resp = client.get(f"/api/v1/shap/{payload}")
        assert resp.status_code == 422, f"Expected 422 for SQL payload in crop, got {resp.status_code}"


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERSIZED PAYLOADS — must all return 422
# ═══════════════════════════════════════════════════════════════════════════════

class TestOversizedPayloads:
    """Strings exceeding max_length limits must be rejected."""

    def test_prices_oversized_crop(self, client):
        huge_crop = "a" * 200
        resp = client.get(f"/api/v1/prices/{huge_crop}/Azadpur")
        assert resp.status_code == 422

    def test_prices_oversized_mandi(self, client):
        huge_mandi = "a" * 200
        resp = client.get(f"/api/v1/prices/wheat/{huge_mandi}")
        assert resp.status_code == 422

    def test_forecast_oversized_crop(self, client):
        resp = client.get(f"/api/v1/forecast/{'x' * 200}/Azadpur")
        assert resp.status_code == 422

    def test_shap_oversized_crop(self, client):
        resp = client.get(f"/api/v1/shap/{'x' * 200}")
        assert resp.status_code == 422

    def test_register_oversized_password(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "phone": "9876543210",
            "password": "p" * 500,
        })
        assert resp.status_code == 422

    def test_register_oversized_full_name(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "phone": "9876543210",
            "password": "securepass",
            "full_name": "N" * 500,
        })
        assert resp.status_code == 422

    def test_alert_oversized_crop(self, client):
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "c" * 200,
            "mandi": "Azadpur",
            "threshold_price": 100.0,
        })
        assert resp.status_code == 422

    def test_alert_oversized_mandi(self, client):
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat",
            "mandi": "m" * 200,
            "threshold_price": 100.0,
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
#  WRONG TYPES — must all return 422
# ═══════════════════════════════════════════════════════════════════════════════

class TestWrongTypes:
    """Sending wrong types (e.g. string where number expected) must be rejected."""

    def test_prices_days_string(self, client):
        resp = client.get("/api/v1/prices/wheat/Azadpur?days=not_a_number")
        assert resp.status_code == 422

    def test_prices_days_negative(self, client):
        resp = client.get("/api/v1/prices/wheat/Azadpur?days=-10")
        assert resp.status_code == 422

    def test_prices_days_too_large(self, client):
        resp = client.get("/api/v1/prices/wheat/Azadpur?days=99999")
        assert resp.status_code == 422

    def test_forecast_horizon_string(self, client):
        resp = client.get("/api/v1/forecast/wheat/Azadpur?horizon=abc")
        assert resp.status_code == 422

    def test_forecast_horizon_too_large(self, client):
        resp = client.get("/api/v1/forecast/wheat/Azadpur?horizon=200")
        assert resp.status_code == 422

    def test_forecast_horizon_zero(self, client):
        resp = client.get("/api/v1/forecast/wheat/Azadpur?horizon=0")
        assert resp.status_code == 422

    def test_alert_threshold_price_string(self, client):
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat",
            "mandi": "Azadpur",
            "threshold_price": "not_a_number",
        })
        assert resp.status_code == 422

    def test_alert_threshold_price_negative(self, client):
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat",
            "mandi": "Azadpur",
            "threshold_price": -100.0,
        })
        assert resp.status_code == 422

    def test_alert_threshold_price_zero(self, client):
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat",
            "mandi": "Azadpur",
            "threshold_price": 0.0,
        })
        assert resp.status_code == 422

    def test_alert_threshold_price_too_large(self, client):
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat",
            "mandi": "Azadpur",
            "threshold_price": 2_000_000.0,
        })
        assert resp.status_code == 422

    def test_register_phone_non_digits(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "phone": "abc-def-ghij",
            "password": "securepass123",
        })
        assert resp.status_code == 422

    def test_register_phone_too_short(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "phone": "12345",
            "password": "securepass123",
        })
        assert resp.status_code == 422

    def test_login_phone_non_digits(self, client):
        resp = client.post("/api/v1/auth/login", json={
            "phone": "not-a-phone!",
            "password": "securepass123",
        })
        assert resp.status_code == 422

    def test_login_password_empty(self, client):
        resp = client.post("/api/v1/auth/login", json={
            "phone": "9876543210",
            "password": "",
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRA FIELDS (extra="forbid") — must all return 422
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtraFieldsRejected:
    """Request bodies with unknown fields must be rejected with extra='forbid'."""

    def test_register_extra_field(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "phone": "9876543210",
            "password": "securepass123",
            "is_admin": True,  # extra field
        })
        assert resp.status_code == 422
        assert "extra" in resp.text.lower() or "not permitted" in resp.text.lower()

    def test_login_extra_field(self, client):
        resp = client.post("/api/v1/auth/login", json={
            "phone": "9876543210",
            "password": "securepass123",
            "role": "admin",  # extra field
        })
        assert resp.status_code == 422

    def test_alert_extra_field(self, client):
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat",
            "mandi": "Azadpur",
            "threshold_price": 2500.0,
            "notify_email": "hacker@evil.com",  # extra field
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
#  XSS / SPECIAL CHARACTER PAYLOADS — must return 422
# ═══════════════════════════════════════════════════════════════════════════════

XSS_PAYLOADS = [
    "<script>alert(1)</script>",
    "javascript:alert(1)",
    '"><img src=x onerror=alert(1)>',
    "{{7*7}}",
    "${7*7}",
]


class TestXSSPayloads:
    """XSS / template injection payloads in path params must be rejected."""

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_prices_crop_xss(self, client, payload):
        resp = client.get(f"/api/v1/prices/{payload}/Azadpur")
        assert resp.status_code in (422, 404)

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_shap_crop_xss(self, client, payload):
        resp = client.get(f"/api/v1/shap/{payload}")
        assert resp.status_code in (422, 404)


# ═══════════════════════════════════════════════════════════════════════════════
#  EMPTY / BOUNDARY VALUES — must return 422
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryValues:
    """Edge cases: empty strings, whitespace-only, single chars."""

    def test_prices_empty_crop(self, client):
        """Empty crop in path → 404 (no matching route) or 422."""
        resp = client.get("/api/v1/prices//Azadpur")
        # FastAPI may return 404 for empty path segments (route not matched)
        assert resp.status_code in (404, 422)

    def test_register_password_too_short(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "phone": "9876543210",
            "password": "ab",  # min_length=6
        })
        assert resp.status_code == 422

    def test_alert_crop_starts_with_number(self, client):
        """Crop names must start with a letter."""
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "123wheat",
            "mandi": "Azadpur",
            "threshold_price": 100.0,
        })
        assert resp.status_code == 422

    def test_alert_crop_special_chars(self, client):
        """Crop names must not contain special chars like @#$."""
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat@#$",
            "mandi": "Azadpur",
            "threshold_price": 100.0,
        })
        assert resp.status_code == 422

    def test_alert_mandi_special_chars(self, client):
        """Mandi names must not contain special chars like ;'\"."""
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat",
            "mandi": "Azadpur'; DROP TABLE--",
            "threshold_price": 100.0,
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
#  VALID INPUTS SHOULD STILL PASS VALIDATION (not blocked by overly strict rules)
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidInputsAccepted:
    """Sanity checks: well-formed inputs pass validation and reach business logic."""

    def test_prices_valid_crop_mandi_passes_validation(self, client):
        """A valid request should not get 422. It may 404/500 since DB is mocked."""
        resp = client.get("/api/v1/prices/wheat/Azadpur?days=30")
        # Should not be 422 (validation passed), but may be 500 due to mock DB
        assert resp.status_code != 422

    def test_forecast_valid_params_passes_validation(self, client):
        resp = client.get("/api/v1/forecast/rice/Karnal?horizon=7")
        assert resp.status_code != 422

    def test_shap_valid_crop_passes_validation(self, client):
        resp = client.get("/api/v1/shap/wheat")
        assert resp.status_code != 422

    def test_register_valid_body_passes_validation(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "phone": "9876543210",
            "password": "securepass123",
            "full_name": "Ramesh Kumar",
        })
        # Should not be 422; may fail at DB layer
        assert resp.status_code != 422

    def test_login_valid_body_passes_validation(self, client):
        resp = client.post("/api/v1/auth/login", json={
            "phone": "9876543210",
            "password": "securepass123",
        })
        assert resp.status_code != 422

    def test_alert_valid_body_passes_validation(self, client):
        resp = client.post("/api/v1/alerts/subscribe", json={
            "crop": "wheat",
            "mandi": "Azadpur",
            "threshold_price": 2500.0,
        })
        assert resp.status_code != 422

    def test_crop_with_hyphen_accepted(self, client):
        """Crop names like 'Tur-Dal' should be accepted."""
        resp = client.get("/api/v1/prices/Tur-Dal/Azadpur")
        assert resp.status_code != 422

    def test_crop_with_space_accepted(self, client):
        """Crop names like 'Tur Dal' should be accepted (URL-encoded space)."""
        resp = client.get("/api/v1/prices/Tur%20Dal/Azadpur")
        assert resp.status_code != 422

    def test_mandi_with_dot_accepted(self, client):
        """Mandi names like 'S.A.S. Nagar' should be accepted."""
        resp = client.get("/api/v1/prices/wheat/S.A.S.%20Nagar")
        assert resp.status_code != 422
