import importlib
import os
import sys
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient


_BASE_ENV = {
    "SECRET_KEY": "a" * 64,
    "DATABASE_URL": "sqlite+aiosqlite:///./test.db",
}

_FORECAST_MODULES = [
    "app.api.routes_forecast",
    "app.api.deps",
    "app.api.schemas",
    "app.database",
    "app.logger",
    "app.config",
    "app.services.forecast_service",
    "app.repositories.forecast_repository",
]


class DummyModel:
    input_shape = (None, 60, 53)
    output_shape = (None, 30)


def _clear_forecast_modules() -> None:
    for module_name in _FORECAST_MODULES:
        sys.modules.pop(module_name, None)


def _build_forecast_test_app() -> FastAPI:
    with patch.dict(os.environ, _BASE_ENV, clear=False):
        _clear_forecast_modules()
        deps = importlib.import_module("app.api.deps")
        routes = importlib.import_module("app.api.routes_forecast")

        app = FastAPI()
        app.include_router(routes.router)

        async def _override_db():
            yield object()

        app.dependency_overrides[deps.get_db] = _override_db
        return app


def _make_records(base_price: float) -> list[dict]:
    start = date(2026, 1, 1)
    records = []
    for offset in range(120):
        price = base_price + offset * 2
        records.append(
            {
                "date": start + timedelta(days=offset),
                "modal_price": price,
                "min_price": price - 10,
                "max_price": price + 10,
                "arrivals_tonnes": 100 + offset,
                "rainfall_mm": float(offset % 7),
                "max_temp": 30.0,
                "min_temp": 18.0,
                "freight_index": 100.0,
                "futures_price": price + 5,
                "msp": 2275.0,
            }
        )
    return records


async def _fake_fetch_historical_prices(self, crop: str, mandi: str, days: int = 365):
    return _make_records(1900.0)


def _fake_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = 70
    last_price = float(df["modal_price"].iloc[-1])
    feature_matrix = [[last_price for _ in range(53)] for _ in range(rows)]
    return pd.DataFrame(feature_matrix)


def test_forecast_endpoint_returns_503_when_model_cannot_load(capsys):
    app = _build_forecast_test_app()

    with patch(
        "app.repositories.forecast_repository.ForecastRepository.get_cached_forecast",
        new=AsyncMock(return_value=None),
    ), patch(
        "app.repositories.forecast_repository.ForecastRepository.set_cached_forecast",
        new=AsyncMock(return_value=None),
    ), patch(
        "app.repositories.forecast_repository.ForecastRepository.fetch_historical_prices",
        _fake_fetch_historical_prices,
    ), patch(
        "app.services.forecast_service.ForecastService.load_crop_model",
        return_value=None,
    ):
        with TestClient(app) as client:
            response = client.get("/api/v1/forecast/wheat/Azadpur?horizon=30")

    stdout = capsys.readouterr().out
    assert response.status_code == 503
    assert "trained model could not be loaded" in response.json()["detail"]
    assert "Forecast model unavailable for wheat/Azadpur" in stdout


def test_forecast_endpoint_returns_503_when_model_inference_fails(capsys):
    app = _build_forecast_test_app()

    with patch(
        "app.repositories.forecast_repository.ForecastRepository.get_cached_forecast",
        new=AsyncMock(return_value=None),
    ), patch(
        "app.repositories.forecast_repository.ForecastRepository.set_cached_forecast",
        new=AsyncMock(return_value=None),
    ), patch(
        "app.repositories.forecast_repository.ForecastRepository.fetch_historical_prices",
        _fake_fetch_historical_prices,
    ), patch(
        "app.services.forecast_service.ForecastService.load_crop_model",
        return_value=DummyModel(),
    ), patch(
        "app.services.forecast_service.ForecastService.load_scaler",
        return_value=None,
    ), patch(
        "app.feature_engineering.engineer_features",
        side_effect=_fake_engineer_features,
    ), patch(
        "app.forecast_model.get_mc_dropout_predictions",
        side_effect=RuntimeError("boom"),
    ):
        with TestClient(app) as client:
            response = client.get("/api/v1/forecast/wheat/Azadpur?horizon=30")

    stdout = capsys.readouterr().out
    assert response.status_code == 503
    assert "Forecast inference unavailable for wheat/Azadpur" in response.json()["detail"]
    assert "MC Dropout prediction failed: RuntimeError" in response.json()["detail"]
    assert "Forecast inference unavailable for wheat/Azadpur" in stdout
