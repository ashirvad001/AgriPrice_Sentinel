import importlib
import os
import sys
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


_BASE_ENV = {
    "SECRET_KEY": "a" * 64,
    "DATABASE_URL": "sqlite+aiosqlite:///./test.db",
}


def _load_main_app_module():
    with patch.dict(os.environ, _BASE_ENV, clear=False):
        sys.modules.pop("app.app", None)
        return importlib.import_module("app.app")


def test_readiness_endpoint_returns_503_when_startup_probe_fails():
    app_module = _load_main_app_module()

    with patch("alembic.config.main", return_value=None), patch.object(
        app_module,
        "init_db",
        new=AsyncMock(return_value=None),
    ), patch.object(
        app_module,
        "init_redis",
        new=AsyncMock(return_value=None),
    ), patch.object(
        app_module,
        "close_redis",
        new=AsyncMock(return_value=None),
    ), patch.object(
        app_module.ForecastService,
        "probe_model_readiness",
        return_value=(False, "Cannot load canary model for wheat/Azadpur"),
    ):
        with TestClient(app_module.app) as client:
            response = client.get("/healthz/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["reason"] == "Cannot load canary model for wheat/Azadpur"


def test_readiness_endpoint_returns_200_when_startup_probe_passes():
    app_module = _load_main_app_module()

    with patch("alembic.config.main", return_value=None), patch.object(
        app_module,
        "init_db",
        new=AsyncMock(return_value=None),
    ), patch.object(
        app_module,
        "init_redis",
        new=AsyncMock(return_value=None),
    ), patch.object(
        app_module,
        "close_redis",
        new=AsyncMock(return_value=None),
    ), patch.object(
        app_module.ForecastService,
        "probe_model_readiness",
        return_value=(True, "wheat/Azadpur"),
    ):
        with TestClient(app_module.app) as client:
            response = client.get("/healthz/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["canary_model"] == "wheat/Azadpur"
    assert "reason" not in payload
